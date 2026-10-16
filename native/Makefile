CFLAGS ?= -O3 -march=x86-64-v3 -Wall
../bin/r9700-pw-camera: pw_camera.c
	$(CC) $(CFLAGS) -o $@ $< $(shell pkg-config --cflags --libs libpipewire-0.3)
