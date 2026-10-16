/*
 * r9700-pw-camera: publish frames as a PipeWire camera (media.class=Video/Source, media.role=Camera).
 *
 * No kernel module, no root. OBS's "Video Capture Device (PipeWire)" source and browsers using the
 * camera portal can open it. Offers BGRA (alpha preserved in OBS), BGRx and YUY2.
 *
 * Frames come from a shared-memory triple buffer written by the Python app (see pwcam.py):
 *   header (64 B): magic, version, width, height, slot_bytes, fps, seq(u64), slot(u32)
 *   then 3 slots of width*height*4 BGRA bytes.
 * Each byte arriving on stdin means "a new frame is ready": we trigger one graph cycle (the stream
 * is the driver, so the consumer runs at exactly our frame rate). EOF on stdin = quit.
 *
 * usage: r9700-pw-camera <shm-path> <width> <height> <fps> [node-description]
 */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <pipewire/pipewire.h>
#include <spa/param/video/format-utils.h>
#include <spa/param/buffers.h>
#include <spa/buffer/meta.h>

#define MAGIC 0x52393742u /* "R97B" */

struct shm_header {
	uint32_t magic, version, width, height, slot_bytes, fps;
	_Atomic uint64_t seq;
	_Atomic uint32_t slot;
	uint8_t pad[64 - 6 * 4 - 8 - 4];
};

struct data {
	struct pw_main_loop *loop;
	struct pw_stream *stream;
	struct spa_source *in_src;
	struct spa_video_info_raw format;
	int negotiated;
	uint32_t width, height, fps;
	uint8_t *shm;
	size_t shm_size;
	uint64_t last_seq;
	uint64_t frames;
	uint64_t triggers, no_buffer;
};

static inline uint8_t clamp8(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

static void bgra_to_yuy2(const uint8_t *src, uint8_t *dst, uint32_t w, uint32_t h, int dst_stride)
{
	for (uint32_t y = 0; y < h; y++) {
		const uint8_t *s = src + (size_t)y * w * 4;
		uint8_t *d = dst + (size_t)y * dst_stride;
		for (uint32_t x = 0; x + 1 < w; x += 2, s += 8, d += 4) {
			int b0 = s[0], g0 = s[1], r0 = s[2], b1 = s[4], g1 = s[5], r1 = s[6];
			int y0 = ((66 * r0 + 129 * g0 + 25 * b0 + 128) >> 8) + 16;
			int y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;
			int r = (r0 + r1) >> 1, g = (g0 + g1) >> 1, b = (b0 + b1) >> 1;
			int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
			int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
			d[0] = clamp8(y0); d[1] = clamp8(u); d[2] = clamp8(y1); d[3] = clamp8(v);
		}
	}
}

static void on_process(void *userdata)
{
	struct data *data = userdata;
	struct shm_header *hdr = (struct shm_header *)data->shm;
	struct pw_buffer *b;
	struct spa_buffer *buf;

	if ((b = pw_stream_dequeue_buffer(data->stream)) == NULL) {
		data->no_buffer++;
		return;
	}
	buf = b->buffer;
	uint8_t *dst = buf->datas[0].data;
	if (dst == NULL) {
		pw_stream_queue_buffer(data->stream, b);
		return;
	}
	uint32_t slot = atomic_load(&hdr->slot) % 3;
	const uint8_t *src = data->shm + sizeof(struct shm_header) + (size_t)slot * hdr->slot_bytes;
	uint32_t w = data->width, h = data->height;
	int stride;

	int need = (data->format.format == SPA_VIDEO_FORMAT_YUY2 ? SPA_ROUND_UP_N(w * 2, 4) : (int)w * 4) * h;
	if (buf->datas[0].maxsize < (uint32_t)need) {
		static int warned;
		if (!warned++)
			fprintf(stderr, "r9700-pw-camera: buffer too small (%u < %d), skipping\n",
				buf->datas[0].maxsize, need);
		pw_stream_queue_buffer(data->stream, b);
		return;
	}
	if (data->format.format == SPA_VIDEO_FORMAT_YUY2) {
		stride = SPA_ROUND_UP_N(w * 2, 4);
		bgra_to_yuy2(src, dst, w, h, stride);
	} else { /* BGRA / BGRx share the byte layout */
		stride = w * 4;
		memcpy(dst, src, (size_t)stride * h);
	}
	struct spa_meta_header *mh = spa_buffer_find_meta_data(buf, SPA_META_Header, sizeof(*mh));
	if (mh) {
		mh->flags = 0;
		mh->offset = 0;
		mh->seq = data->frames;
		mh->pts = -1;
		mh->dts_offset = 0;
	}
	buf->datas[0].chunk->offset = 0;
	buf->datas[0].chunk->size = stride * h;
	buf->datas[0].chunk->stride = stride;
	data->frames++;
	pw_stream_queue_buffer(data->stream, b);
}

static void on_param_changed(void *userdata, uint32_t id, const struct spa_pod *param)
{
	struct data *data = userdata;
	uint8_t params_buffer[1024];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(params_buffer, sizeof(params_buffer));
	const struct spa_pod *params[2];

	if (param == NULL || id != SPA_PARAM_Format)
		return;
	if (spa_format_video_raw_parse(param, &data->format) < 0)
		return;
	int bpp = data->format.format == SPA_VIDEO_FORMAT_YUY2 ? 2 : 4;
	int stride = SPA_ROUND_UP_N(data->width * bpp, 4);
	fprintf(stderr, "r9700-pw-camera: negotiated %s %ux%u\n",
		data->format.format == SPA_VIDEO_FORMAT_YUY2 ? "YUY2" :
		data->format.format == SPA_VIDEO_FORMAT_BGRA ? "BGRA" : "BGRx",
		data->width, data->height);

	params[0] = spa_pod_builder_add_object(&b,
		SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
		SPA_PARAM_BUFFERS_buffers, SPA_POD_CHOICE_RANGE_Int(8, 2, 16),
		SPA_PARAM_BUFFERS_blocks,  SPA_POD_Int(1),
		SPA_PARAM_BUFFERS_size,    SPA_POD_Int(stride * data->height),
		SPA_PARAM_BUFFERS_stride,  SPA_POD_Int(stride));
	params[1] = spa_pod_builder_add_object(&b,
		SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
		SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Header),
		SPA_PARAM_META_size, SPA_POD_Int(sizeof(struct spa_meta_header)));
	pw_stream_update_params(data->stream, params, 2);
	data->negotiated = 1;
}

static void on_state_changed(void *userdata, enum pw_stream_state old, enum pw_stream_state state,
			     const char *error)
{
	fprintf(stderr, "r9700-pw-camera: %s%s%s\n", pw_stream_state_as_string(state),
		error ? ": " : "", error ? error : "");
}

static const struct pw_stream_events stream_events = {
	PW_VERSION_STREAM_EVENTS,
	.process = on_process,
	.param_changed = on_param_changed,
	.state_changed = on_state_changed,
};

static void on_stdin(void *userdata, int fd, uint32_t mask)
{
	struct data *data = userdata;
	char tmp[256];
	ssize_t n = read(fd, tmp, sizeof(tmp));
	if (n == 0 || (n < 0 && errno != EAGAIN && errno != EINTR) || (mask & SPA_IO_HUP)) {
		pw_main_loop_quit(data->loop);   /* producer closed stdin (or died): unpublish the camera */
		return;
	}
	if (n < 0)
		return;
	struct shm_header *hdr = (struct shm_header *)data->shm;
	uint64_t seq = atomic_load(&hdr->seq);
	if (seq != data->last_seq) {
		data->last_seq = seq;
		data->triggers++;
		pw_stream_trigger_process(data->stream);
		if (getenv("R9700_PWCAM_STATS") && data->triggers % 120 == 0)
			fprintf(stderr, "stats: triggers %lu frames %lu no_buffer %lu\n",
				(unsigned long)data->triggers, (unsigned long)data->frames,
				(unsigned long)data->no_buffer);
	}
}

static void do_quit(void *userdata, int signal_number)
{
	struct data *data = userdata;
	pw_main_loop_quit(data->loop);
}

static struct spa_pod *format_param(struct spa_pod_builder *b, enum spa_video_format f,
				    uint32_t w, uint32_t h, uint32_t fps)
{
	return spa_pod_builder_add_object(b,
		SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
		SPA_FORMAT_mediaType,       SPA_POD_Id(SPA_MEDIA_TYPE_video),
		SPA_FORMAT_mediaSubtype,    SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
		SPA_FORMAT_VIDEO_format,    SPA_POD_Id(f),
		SPA_FORMAT_VIDEO_size,      SPA_POD_Rectangle(&SPA_RECTANGLE(w, h)),
		SPA_FORMAT_VIDEO_framerate, SPA_POD_Fraction(&SPA_FRACTION(fps, 1)));
}

int main(int argc, char *argv[])
{
	struct data data = { 0 };
	uint8_t buffer[2048];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
	const struct spa_pod *params[3];

	if (argc < 5) {
		fprintf(stderr, "usage: %s <shm-path> <width> <height> <fps> [description]\n", argv[0]);
		return 2;
	}
	data.width = atoi(argv[2]);
	data.height = atoi(argv[3]);
	data.fps = atoi(argv[4]);
	const char *desc = argc > 5 ? argv[5] : "R9700 Broadcast Camera";

	int fd = open(argv[1], O_RDONLY);
	if (fd < 0) { perror("open shm"); return 1; }
	struct stat st;
	fstat(fd, &st);
	data.shm_size = st.st_size;
	data.shm = mmap(NULL, data.shm_size, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);
	if (data.shm == MAP_FAILED) { perror("mmap"); return 1; }
	struct shm_header *hdr = (struct shm_header *)data.shm;
	if (hdr->magic != MAGIC || hdr->width != data.width || hdr->height != data.height) {
		fprintf(stderr, "bad shm header\n");
		return 1;
	}

	pw_init(&argc, &argv);
	data.loop = pw_main_loop_new(NULL);
	pw_loop_add_signal(pw_main_loop_get_loop(data.loop), SIGINT, do_quit, &data);
	pw_loop_add_signal(pw_main_loop_get_loop(data.loop), SIGTERM, do_quit, &data);

	char fps_s[32];
	snprintf(fps_s, sizeof(fps_s), "1/%u", data.fps);
	data.stream = pw_stream_new_simple(
		pw_main_loop_get_loop(data.loop), "r9700-broadcast-camera",
		pw_properties_new(
			PW_KEY_MEDIA_TYPE, "Video",
			PW_KEY_MEDIA_CATEGORY, "Capture",
			PW_KEY_MEDIA_ROLE, "Camera",
			PW_KEY_MEDIA_CLASS, "Video/Source",
			PW_KEY_NODE_NAME, "r9700_broadcast_camera",
			PW_KEY_NODE_DESCRIPTION, desc,
			PW_KEY_NODE_NICK, desc,
			PW_KEY_NODE_LATENCY, fps_s,
			PW_KEY_NODE_RATE, fps_s,
			NULL),
		&stream_events, &data);

	params[0] = format_param(&b, SPA_VIDEO_FORMAT_BGRA, data.width, data.height, data.fps);
	params[1] = format_param(&b, SPA_VIDEO_FORMAT_BGRx, data.width, data.height, data.fps);
	params[2] = format_param(&b, SPA_VIDEO_FORMAT_YUY2, data.width, data.height, data.fps);

	pw_stream_connect(data.stream, PW_DIRECTION_OUTPUT, PW_ID_ANY,
			  PW_STREAM_FLAG_DRIVER | PW_STREAM_FLAG_MAP_BUFFERS,
			  params, 3);

	fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);
	data.in_src = pw_loop_add_io(pw_main_loop_get_loop(data.loop), 0, SPA_IO_IN | SPA_IO_HUP, false,
				     on_stdin, &data);

	fprintf(stderr, "r9700-pw-camera: publishing '%s' %ux%u@%u\n", desc, data.width, data.height, data.fps);
	pw_main_loop_run(data.loop);

	pw_stream_destroy(data.stream);
	pw_main_loop_destroy(data.loop);
	munmap(data.shm, data.shm_size);
	pw_deinit();
	return 0;
}
