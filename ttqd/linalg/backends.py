#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#


r"""
Selects the backend for the openms-package.
The `openms` allows to choose a backend. The ``numpy`` backend is the
default one, but there are also several additional backends:

"""

from __future__ import annotations
from functools import wraps

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union, Callable
from ttqd.linalg._tamm import (
    PYTAMM_AVAILABLE,
    PYTAMM_GPU_AVAILABLE,
    PYTAMM_MPI_AVAILABLE,
    PYTAMM_INFO,
)

Shape = Union[int, Sequence[int], Tuple[int, ...]]
DTypeLike = Union[str, Any]  # "float32" or np.float32/torch.float32/jnp.float32

# ------------------------------------
# check available backends
# ------------------------------------
_AVAILABLE_BACKENDS = ["numpy"]
_AVAILABLE_DEVICE = {"numpy": ["cpu"]}


if PYTAMM_AVAILABLE:
    _AVAILABLE_BACKENDS.append("tamm")
    if PYTAMM_GPU_AVAILABLE:
        _AVAILABLE_DEVICE["tamm"] = ["cpu", "gpu"]
    else:
        _AVAILABLE_DEVICE["tamm"] = ["cpu"]


try:
    import cupy

    CUPY_AVAILABLE = True
    _AVAILABLE_BACKENDS.append("cupy")
    _AVAILABLE_DEVICE["cupy"] = []  # GPUs - populated by cupy
except ImportError:
    CUPY_AVAILABLE = False


try:
    import torch

    torch.set_default_dtype(torch.float64)  # we need high precision
    try:  # we don't need gradients (for now)
        torch._C.set_grad_enabled(False)  # type: ignore
    except AttributeError:
        torch._C._set_grad_enabled(False)
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()

    _AVAILABLE_BACKENDS.append("torch")
    _AVAILABLE_DEVICE["torch"] = ["cpu"]
    if TORCH_CUDA_AVAILABLE:
        _AVAILABLE_BACKENDS.append("torch_cuda")
        _AVAILABLE_BACKENDS.append("torch_gpu")
        _AVAILABLE_DEVICE["torch_cuda"] = ["cuda"]
        _AVAILABLE_DEVICE["torch_gpu"] = ["cuda"]
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False


# TiledArray Backends (and flags)
try:
    import tiledarray as TA

    TA_AVAILABLE = True
    _AVAILABLE_BACKENDS.append("tiledarray")
    _AVAILABLE_BACKENDS.append("ta")
    _AVAILABLE_DEVICE["tiledarray"] = []  # TBA
    _AVAILABLE_DEVICE["ta"] = []  # TBA
    # TA_CUDA_AVAILABLE = TA.cuda_available() # (todo)
except ImportError:
    TA_AVAILABLE = False
    TA_CUDA_AVAILABLE = False

try:
    import jax

    JAX_AVAILABLE = True
    _AVAILABLE_BACKENDS.append("jax")
    if JAX_AVAILABLE:
        # print("jax devices = ", jax.devices())
        _AVAILABLE_DEVICE["jax"] = [str(d) for d in jax.devices()]
        if "gpu" in [d.platform for d in jax.devices()]:
            _AVAILABLE_BACKENDS.append("jax_gpu")
            _AVAILABLE_DEVICE["jax_gpu"] = [
                str(d) for d in jax.devices() if d.platform == "gpu"
            ]
except ImportError:
    JAX_AVAILABLE = False

# -------------------------
# Helpers
# -------------------------


def _normalize_shape(shape: Shape) -> Tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(x) for x in shape)


def _dtype_name(dtype: DTypeLike) -> str:
    # Normalize dtype to a string key like "float32"
    if dtype is None:
        return "float32"
    if isinstance(dtype, str):
        return dtype.lower()
    # numpy dtype / torch dtype / jax dtype objects
    s = str(dtype).lower()
    # common formats: "<class 'numpy.float32'>", "torch.float32", "float32"
    for key in ("float16", "bfloat16", "float32", "float64", "int32", "int64", "bool"):
        if key in s:
            return key
    raise ValueError(f"Unsupported dtype: {dtype} (parsed as {s})")


def _ensure_seed_int(seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    if not isinstance(seed, int):
        raise TypeError("seed must be int or None")
    return seed


def _replace_float(func):
    """replace the default dtype a function is called with"""

    @wraps(func)
    def new_func(self, *args, **kwargs):
        result = func(*args, **kwargs)
        if result.dtype in numpy_float_dtypes:
            result = numpy.asarray(result, dtype=self.float)
        return result

    return new_func


# -------------------------
# Backend base interface
# -------------------------


class _BackendBase:
    name: str

    # TODO: rename attribute (torch, np, jax) to xp
    #       and then implement all the common functions in the base
    #       class

    # other constants? TBA/TBD

    # Device/dtype mapping
    def dtype(self, dtype: DTypeLike):
        raise NotImplementedError

    def asarray(self, x, dtype: DTypeLike = None):
        raise NotImplementedError

    # Creation
    def zeros(self, shape: Shape, dtype: DTypeLike = "float32"):
        raise NotImplementedError

    def ones(self, shape: Shape, dtype: DTypeLike = "float32"):
        raise NotImplementedError

    def eye(self, n, dtype=None):
        if self.name == "torch":
            return self.xp.eye(n, dtype=dtype)
        return self.xp.eye(n, dtype=dtype)

    def real(self, x):
        return x.real if self.name != "torch" else self.xp.real(x)

    def abs(self, x):
        return self.xp.abs(x)

    def conj(self, x):
        return self.xp.conj(x)

    def empty(self, shape: Shape, dtype: DTypeLike = "float32"):
        raise NotImplementedError

    def arange(self, start, stop=None, step=1, dtype: DTypeLike = "int64"):
        raise NotImplementedError

    # Ops
    def reshape(self, x, shape: Shape):
        raise NotImplementedError

    def transpose(self, x, axes=None):
        raise NotImplementedError

    def sum(self, x):
        return self.xp.sum(x)

    def trace(self, a):
        return self.xp.trace(a)

    def vdot(self, a, b):
        return self.xp.vdot(a.reshape(-1), b.reshape(-1))

    def norm(self, x):
        return self.xp.sqrt(self.xp.real(self.vdot(x, x)))

    def mean(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def dot(self, a, b):
        raise NotImplementedError

    def matmul(self, a, b):
        return self.dot(a, b)

    # Random namespace
    @property
    def random(self):
        raise NotImplementedError

    # Interop helpers
    def to_numpy(self, x):
        raise NotImplementedError

    def from_numpy(self, x, dtype=None):
        raise NotImplementedError

    def synchronize(self):
        # optional no-op for numpy/jax-cpu
        return None

    def einsum(self, subscripts: str, *operands):
        return self.xp.einsum(subscripts, *operands)

    def qr(self, a):
        return self._linalg.qr(a)

    def eigh(self, a):
        return self._linalg.eigh(a)

    def eigvalsh(self, a):
        return self._linalg.eigvalsh(a)

    def exp(self, x):
        return self.xp.exp(x)

    def svd(self, a, full_matrices=False):
        return self._linalg.svd(a, full_matrices=full_matrices)

    def available_devices(self):
        """Return a list of device identifiers available to this backend."""
        raise NotImplementedError


# -------------------------
# NumPy backend
# -------------------------


class _NumpyRandom:
    def __init__(self):
        import numpy as np

        self.np = np
        self._rng = np.random.default_rng()

    def seed(self, seed: Optional[int]):
        seed = _ensure_seed_int(seed)
        if seed is None:
            self._rng = self.np.random.default_rng()
        else:
            self._rng = self.np.random.default_rng(seed)

    def normal(self, shape: Shape, mean=0.0, std=1.0, dtype="float32"):
        shape = _normalize_shape(shape)
        dt = self.np.dtype(_dtype_name(dtype))
        x = self._rng.normal(loc=mean, scale=std, size=shape)
        return x.astype(dt, copy=False)

    def rand(self, *args):
        return self.np.random.rand(*args)

    def uniform(self, shape: Shape, low=0.0, high=1.0, dtype="float32"):
        shape = _normalize_shape(shape)
        dt = self.np.dtype(_dtype_name(dtype))
        x = self._rng.uniform(low=low, high=high, size=shape)
        return x.astype(dt, copy=False)

    def randint(
        self, shape: Shape, low: int, high: Optional[int] = None, dtype="int64"
    ):
        shape = _normalize_shape(shape)
        dt = self.np.dtype(_dtype_name(dtype))
        x = self._rng.integers(low=low, high=high, size=shape, endpoint=False)
        return x.astype(dt, copy=False)


class NumpyBackend(_BackendBase):
    name = "numpy"

    def __init__(self):
        import numpy as np

        self.np = np
        self.xp = np
        self._random = _NumpyRandom()
        self._linalg = self.xp.linalg

    def available_devices(self):
        return ["cpu"]

    def dtype(self, dtype: DTypeLike):
        return self.xp.dtype(_dtype_name(dtype))

    def asarray(self, x, dtype: DTypeLike = None):
        dt = self.dtype(dtype) if dtype is not None else None
        return self.xp.asarray(x, dtype=dt)

    def zeros(self, shape: Shape, dtype: DTypeLike = "float32"):
        return self.xp.zeros(_normalize_shape(shape), dtype=self.dtype(dtype))

    # def ones(self, shape: Shape, dtype: DTypeLike = "float32"):
    #    return self.xp.ones(_normalize_shape(shape), dtype=self.dtype(dtype))
    def ones(self, *args, dtype=None):
        return self.xp.ones(*args, dtype=dtype)

    def empty(self, shape: Shape, dtype: DTypeLike = "float32"):
        return self.xp.empty(_normalize_shape(shape), dtype=self.dtype(dtype))

    def arange(self, start, stop=None, step=1, dtype: DTypeLike = "int64"):
        return self.xp.arange(start, stop, step, dtype=self.dtype(dtype))

    def reshape(self, x, shape: Shape):
        return self.xp.reshape(x, _normalize_shape(shape))

    def transpose(self, x, axes=None):
        return self.xp.transpose(x, axes=axes)

    def sum(self, x, axis=None, keepdims=False):
        return self.xp.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return self.xp.mean(x, axis=axis, keepdims=keepdims)

    def dot(self, a, b):
        # numpy.dot has vector special-cases; for matrices prefer @
        return a @ b

    def log(self, x):
        return self.xp.log(x)

    def clip(self, x, a_min, a_max):
        return self.xp.clip(x, a_min=a_min, a_max=a_max)

    def allclose(self, *args, **kwargs):
        return self.xp.allclose(*args, **kwargs)

    def tensordot(self, *args, **kwargs):
        return self.xp.tensordot(*args, **kwargs)

    @property
    def random(self):
        return self._random

    # conversion
    def to_numpy(self, x):
        return x

    def from_numpy(self, x, dtype=None):
        # Ensure it's a numpy array; cast only if requested
        arr = self.xp.asarray(x)
        if dtype is None:
            return arr
        return arr.astype(self.dtype(dtype), copy=False)


# -------------------------
# Torch backend
# -------------------------


class _TorchRandom:
    def __init__(self, torch_mod, device: str):
        self.xp = self.torch = torch_mod
        self.device = device
        self._gen = None  # torch.Generator per-device

    def seed(self, seed: Optional[int]):
        seed = _ensure_seed_int(seed)
        if seed is None:
            self._gen = None
            return
        g = self.torch.Generator(device=self.device)
        g.manual_seed(seed)
        self._gen = g

    def normal(self, shape: Shape, mean=0.0, std=1.0, dtype="float32"):
        shape = _normalize_shape(shape)
        dt = _TORCH_DTYPE_MAP[_dtype_name(dtype)]
        return self.torch.normal(
            mean=self.torch.tensor(mean, device=self.device, dtype=dt),
            std=self.torch.tensor(std, device=self.device, dtype=dt),
            size=shape,
            generator=self._gen,
            device=self.device,
            dtype=dt,
        )

    def uniform(self, shape: Shape, low=0.0, high=1.0, dtype="float32"):
        shape = _normalize_shape(shape)
        dt = _TORCH_DTYPE_MAP[_dtype_name(dtype)]
        x = self.torch.rand(shape, generator=self._gen, device=self.device, dtype=dt)
        return x * (high - low) + low

    def randint(
        self, shape: Shape, low: int, high: Optional[int] = None, dtype="int64"
    ):
        shape = _normalize_shape(shape)
        dt = _TORCH_DTYPE_MAP[_dtype_name(dtype)]
        if high is None:
            raise ValueError("torch randint requires high")
        return self.torch.randint(
            low=low,
            high=high,
            size=shape,
            generator=self._gen,
            device=self.device,
            dtype=dt,
        )


_TORCH_DTYPE_MAP = {
    "float16": None,
    "bfloat16": None,
    "float32": None,
    "float64": None,
    "int32": None,
    "int64": None,
    "bool": None,
}


def _init_torch_dtype_map():
    import torch

    _TORCH_DTYPE_MAP.update(
        {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
    )


class TorchBackend(_BackendBase):
    name = "torch"

    def __init__(self, device: str = "cpu"):
        import torch

        if _TORCH_DTYPE_MAP["float32"] is None:
            _init_torch_dtype_map()
        self.torch = torch
        self.device = self._normalize_device(device)
        self._random = _TorchRandom(torch, self.device)
        self._linalg = self.torch.linalg

        # optional: allow TF32 on ampere+ for speed
        if self.device.startswith("cuda"):
            self.torch.backends.cuda.matmul.allow_tf32 = True

    def available_devices(self):
        devs = ["cpu"]
        if self.torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(self.torch.cuda.device_count())]
        return devs

    def _normalize_device(self, device: str) -> str:
        d = (device or "cpu").lower()
        if d in ("gpu", "cuda"):
            if not self.torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested but torch.cuda.is_available() is False"
                )
            # pick current device index
            idx = int(
                os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
            )
            return f"cuda:{idx}"
        if d.startswith("cuda"):
            return d
        return "cpu"

    def dtype(self, dtype: DTypeLike):
        return _TORCH_DTYPE_MAP[_dtype_name(dtype)]

    def asarray(self, x, dtype: DTypeLike = None):
        dt = self.dtype(dtype) if dtype is not None else None
        return self.torch.as_tensor(x, device=self.device, dtype=dt)

    def zeros(self, shape: Shape, dtype: DTypeLike = "float32"):
        return self.torch.zeros(
            _normalize_shape(shape), device=self.device, dtype=self.dtype(dtype)
        )

    def ones(self, shape: Shape, dtype: DTypeLike = "float32"):
        return self.torch.ones(
            _normalize_shape(shape), device=self.device, dtype=self.dtype(dtype)
        )

    def empty(self, shape: Shape, dtype: DTypeLike = "float32"):
        return self.torch.empty(
            _normalize_shape(shape), device=self.device, dtype=self.dtype(dtype)
        )

    def arange(self, start, stop=None, step=1, dtype: DTypeLike = "int64"):
        return self.torch.arange(
            start, stop, step, device=self.device, dtype=self.dtype(dtype)
        )

    def reshape(self, x, shape: Shape):
        return x.reshape(_normalize_shape(shape))

    def transpose(self, x, axes=None):
        if axes is None:
            return x.transpose(-2, -1)
        # axes can be a tuple; use permute
        return x.permute(*axes)

    def sum(self, x, axis=None, keepdims=False):
        return x.sum(dim=axis, keepdim=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return x.mean(dim=axis, keepdim=keepdims)

    def dot(self, a, b):
        return a @ b

    def log(self, x):
        return self.torch.log(x)

    def clip(self, x, a_min, a_max):
        return self.torch.clamp(x, min=a_min, max=a_max)

    @property
    def random(self):
        return self._random

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def from_numpy(self, x, dtype=None):
        # Ensure ndarray (handles lists / scalars)
        x = np.asarray(x)

        # torch.from_numpy shares CPU memory; moving to GPU makes a copy
        t = self.torch.from_numpy(x)

        # Only cast if dtype is explicitly requested
        if dtype is not None:
            t = t.to(dtype=self.dtype(dtype))
        # Move to this backend's device (cpu or cuda:<local_rank>)
        return t.to(self.device, non_blocking=(self.device.startswith("cuda")))

    def synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def einsum(self, subscripts: str, *operands):
        return self.torch.einsum(subscripts, *operands)


# -------------------------
# JAX backend
# -------------------------


class _JaxRandom:
    def __init__(self, jax_mod, jnp_mod, key, device):
        self.jax = jax_mod
        self.jnp = jnp_mod
        self.key = key
        self.device = device

    def seed(self, seed: Optional[int]):
        seed = _ensure_seed_int(seed)
        if seed is None:
            seed = 0
        self.key = self.jax.random.PRNGKey(seed)

    def _split(self):
        self.key, sub = self.jax.random.split(self.key)
        return sub

    def normal(self, shape: Shape, mean=0.0, std=1.0, dtype="float32"):
        shape = _normalize_shape(shape)
        dt = getattr(self.jnp, _dtype_name(dtype))
        sub = self._split()
        x = self.jax.random.normal(sub, shape, dtype=dt)
        return x * std + mean

    def uniform(self, shape: Shape, low=0.0, high=1.0, dtype="float32"):
        shape = _normalize_shape(shape)
        dt = getattr(self.jnp, _dtype_name(dtype))
        sub = self._split()
        x = self.jax.random.uniform(sub, shape, dtype=dt, minval=low, maxval=high)
        return x

    def randint(self, shape: Shape, low: int, high: int, dtype="int32"):
        shape = _normalize_shape(shape)
        # jax randint dtype must be integer
        dt = getattr(self.jnp, _dtype_name(dtype))
        sub = self._split()
        return self.jax.random.randint(sub, shape, low, high, dtype=dt)


class JaxBackend(_BackendBase):
    name = "jax"

    def __init__(self, device: str = "cpu"):
        import jax

        # Enable float64 support (must be done before creating arrays)
        jax.config.update("jax_enable_x64", True)

        import jax.numpy as jnp

        self.jax = jax
        self.xp = self.jnp = jnp
        self.device = self._normalize_device(device)
        # place PRNG key on host; arrays will be placed via device_put
        self._random = _JaxRandom(jax, jnp, jax.random.PRNGKey(0), self.device)
        self._linalg = self.jnp.linalg

    def available_devices(self):
        # Examples: "cpu:0", "gpu:0", "tpu:0"
        return [f"{d.platform}:{d.id}" for d in self.jax.devices()]

    def _normalize_device(self, device: str):
        d = (device or "cpu").lower()
        if d in ("gpu", "cuda"):
            # pick local GPU if multiple; JAX uses device indices too
            idx = int(
                os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
            )
            gpus = [dev for dev in self.jax.devices() if dev.platform == "gpu"]
            if not gpus:
                raise RuntimeError(
                    "JAX GPU requested but no GPU devices found (check jaxlib build)."
                )
            if idx >= len(gpus):
                idx = 0
            return gpus[idx]
        # CPU device
        cpus = [dev for dev in self.jax.devices() if dev.platform == "cpu"]
        return cpus[0] if cpus else self.jax.devices()[0]

    def dtype(self, dtype: DTypeLike):
        return getattr(self.jnp, _dtype_name(dtype))

    def asarray(self, x, dtype: DTypeLike = None):
        dt = self.dtype(dtype) if dtype is not None else None
        arr = self.jnp.asarray(x, dtype=dt)
        return self.jax.device_put(arr, self.device)

    def zeros(self, shape: Shape, dtype: DTypeLike = "float32"):
        arr = self.jnp.zeros(_normalize_shape(shape), dtype=self.dtype(dtype))
        return self.jax.device_put(arr, self.device)

    def ones(self, shape: Shape, dtype: DTypeLike = "float32"):
        arr = self.jnp.ones(_normalize_shape(shape), dtype=self.dtype(dtype))
        return self.jax.device_put(arr, self.device)

    def empty(self, shape: Shape, dtype: DTypeLike = "float32"):
        # JAX has no true uninitialized empty; use zeros as a safe stand-in.
        arr = self.jnp.zeros(_normalize_shape(shape), dtype=self.dtype(dtype))
        return self.jax.device_put(arr, self.device)

    def arange(self, start, stop=None, step=1, dtype: DTypeLike = "int32"):
        arr = self.jnp.arange(start, stop, step, dtype=self.dtype(dtype))
        return self.jax.device_put(arr, self.device)

    def reshape(self, x, shape: Shape):
        return self.jnp.reshape(x, _normalize_shape(shape))

    def transpose(self, x, axes=None):
        return self.jnp.transpose(x, axes=axes)

    def sum(self, x, axis=None, keepdims=False):
        return self.jnp.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return self.jnp.mean(x, axis=axis, keepdims=keepdims)

    def dot(self, a, b):
        return a @ b

    @property
    def random(self):
        return self._random

    def to_numpy(self, x):
        import numpy as np

        return np.asarray(x)

    def to_numpy(self, x, dtype=None):
        arr = np.asarray(x)
        if dtype is None:
            return arr
        return arr.astype(np.dtype(dtype), copy=False)

    def from_numpy(self, x, dtype=None):
        x = np.asarray(x)
        target_dtype = self.dtype(dtype) if dtype is not None else x.dtype
        arr = self.jnp.asarray(x, dtype=target_dtype)

        return self.jax.device_put(arr, self.device)

    def synchronize(self):
        # Force completion
        self.jax.block_until_ready(self.zeros((1,), dtype="float32"))

    def einsum(self, subscripts: str, *operands):
        # jnp.einsum exists and returns a JAX array
        return self.jnp.einsum(subscripts, *operands)


# -------------------------
# Global backend selection
# -------------------------


def _warn(msg: str) -> None:
    # Keep it simple; may replace with logging
    print(f"\nWARNING: [backends] {msg}", flush=True)


_backend: _BackendBase = NumpyBackend()


def set_backend(name: str, device: str = "cpu") -> None:
    """
    Set the global backend.

    If requested backend (torch/jax) is unavailable, falls back to NumPy.

    name: "numpy" | "torch" | "jax" | "tamm"
    device:
      - numpy: ignored (always CPU)
      - torch: "cpu" or "gpu"/"cuda" or "cuda:0"
      - jax:   "cpu" or "gpu"/"cuda" (requires GPU-enabled jaxlib)
      - tamm:  "cpu" or "gpu"/"cuda" (if TAMM was built with GPU support)
    """
    global _backend
    name = (name or "numpy").lower()

    if name == "numpy":
        _backend = NumpyBackend()
        return

    if name == "torch":
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            _warn("Torch not installed; falling back to NumPy.")
            _backend = NumpyBackend()
            return
        except Exception as e:
            _warn(f"Failed to import torch ({e}); falling back to NumPy.")
            _backend = NumpyBackend()
            return

        # If user asked for GPU but CUDA not available, downgrade to cpu or fall back to numpy
        if (device or "").lower() in ("gpu", "cuda") and not torch.cuda.is_available():
            _warn("Torch CUDA not available; using TorchBackend(device='cpu') instead.")
            try:
                _backend = TorchBackend(device="cpu")
                return
            except Exception as e:
                _warn(
                    f"Failed to initialize TorchBackend on CPU ({e}); falling back to NumPy."
                )
                _backend = NumpyBackend()
                return

        try:
            _backend = TorchBackend(device=device)
            return
        except Exception as e:
            _warn(f"Failed to initialize TorchBackend ({e}); falling back to NumPy.")
            _backend = NumpyBackend()
            return

    if name == "jax":
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp  # noqa: F401
        except ModuleNotFoundError:
            _warn("JAX not installed; falling back to NumPy.")
            _backend = NumpyBackend()
            return
        except Exception as e:
            _warn(f"Failed to import jax ({e}); falling back to NumPy.")
            _backend = NumpyBackend()
            return

        # If user asked for GPU but JAX has no GPU devices, fall back
        want_gpu = (device or "").lower() in ("gpu", "cuda")
        if want_gpu:
            try:
                has_gpu = any(d.platform == "gpu" for d in jax.devices())
            except Exception:
                has_gpu = False
            if not has_gpu:
                _warn("JAX GPU not available; using JaxBackend(device='cpu') instead.")
                try:
                    _backend = JaxBackend(device="cpu")
                    return
                except Exception as e:
                    _warn(
                        f"Failed to initialize JaxBackend on CPU ({e}); falling back to NumPy."
                    )
                    _backend = NumpyBackend()
                    return

        try:
            _backend = JaxBackend(device=device)
            return
        except Exception as e:
            _warn(f"Failed to initialize JaxBackend ({e}); falling back to NumPy.")
            _backend = NumpyBackend()
            return

    if name == "tamm":
        try:
            from ttqd.linalg._tamm import TammBackend

            _backend = TammBackend(device=device)
            return
        except Exception as e:
            _warn(f"Failed to initialize TammBackend ({e}); falling back to NumPy.")
            _backend = NumpyBackend()
            return

    _warn(f"Unknown backend '{name}'; falling back to NumPy.")
    _backend = NumpyBackend()


def get_backend() -> str:
    return _backend.name


class _LinalgProxy:
    """Expose backend.linalg.<fn> mapped to np.linalg / torch.linalg / jnp.linalg.
    Always proxies to the currently active _backend, not a cached one."""

    def __getattr__(self, item):
        # Always look up the current global _backend for linalg methods
        linalg_mod = getattr(_backend, "_linalg", None)
        if linalg_mod is None:
            raise AttributeError(
                f"Active backend '{getattr(_backend, 'name', 'unknown')}' has no linalg module configured."
            )
        if hasattr(linalg_mod, item):
            return getattr(linalg_mod, item)
        raise AttributeError(
            f"backend.linalg has no attribute '{item}' for backend '{getattr(_backend, 'name', 'unknown')}'."
        )


# Expose the selected backend as a module-like object
class _BackendProxy:
    """
    Proxy that forwards all attribute access to the current global _backend.
    This allows set_backend() to dynamically change the backend at runtime.
    """

    def __getattr__(self, item):
        # Always look up the current global _backend, not a stored reference
        return getattr(_backend, item)

    def __repr__(self):
        return f"<BackendProxy -> {_backend.name}>"


backend = _BackendProxy()
backend.linalg = _LinalgProxy()
