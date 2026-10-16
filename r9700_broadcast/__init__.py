"""Virtual Greenscreen R9700 Broadcast Studio - AI green screen without a green screen, for Radeon AI PRO R9700."""
import os

# hipBLASLt in the ROCm 10 build intermittently faults (HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION in a
# Cijk_*_HHS_BH_Bias kernel) on gfx1201 when two processes run BiRefNet concurrently. rocBLAS is
# just as fast for these small GEMMs and has been stable.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
# fp16 Linear layers with a bias go through hipBLASLt's fused gemm+bias ("Cijk_*_HHS_BH_Bias_*")
# regardless of the preferred BLAS library; that kernel page-faults on gfx1201 in MatAnyone 2's
# transformer. Route addmm through plain rocBLAS + a separate bias add instead.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "0")   # rocBLAS on gfx12 otherwise forwards to the same kernels
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
# CPU thread pools: the heavy work is on the GPU, the CPU only runs small per-frame steps. By default
# torch/OpenMP and OpenCV start one thread per logical CPU, and on a 128-thread CPU those pools spin
# after every tiny op: ~18 cores busy doing nothing. Four threads are as fast and use ~1.7 cores.
_CPU_THREADS = str(min(4, os.cpu_count() or 4))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _CPU_THREADS)

# expandable_segments + HIP graphs on ROCm: when the caching allocator empties its cache (e.g. MIOpen
# retrying a workspace allocation) it unmaps segments a captured graph still points at, and the next
# replay dies with HSA_STATUS_ERROR_MEMORY_FAULT. Strip it even if a parent launcher set it.
_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" in _conf:
    _kept = [p for p in _conf.split(",") if p and not p.startswith("expandable_segments")]
    if _kept:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(_kept)
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")

# the env var only works if set before torch is imported; enforce it either way
import torch as _torch  # noqa: E402

_torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
try:
    import cv2 as _cv2
    _cv2.setNumThreads(int(os.environ["OMP_NUM_THREADS"]))
except ImportError:
    pass

if os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT") == "0" and _torch.version.hip:
    _torch.backends.cuda.preferred_blas_library("cublas")

__version__ = "0.2.0"
