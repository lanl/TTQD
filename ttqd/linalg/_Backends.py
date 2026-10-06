from __future__ import annotations
from functools import wraps

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

# deprecated

# ============================================================
# Backend abstraction
# ============================================================
# will be replaced with openms backend class
class Backend:
    """
    Minimal backend wrapper to support numpy/jax/torch arrays.
    we can further extend this with (TODO):
      - jit/vmap (jax)
      - device placement (torch)
      - custom contraction engines
    """
    def __init__(self, name: str = "numpy"):
        self.name = name.lower()
        if self.name == "numpy":
            import numpy as xp  # type: ignore
            self.xp = xp
            self.linalg = xp.linalg
        elif self.name == "jax":
            import jax.numpy as xp
            self.xp = xp
            self.linalg = xp.linalg
        elif self.name == "torch":
            import torch
            self.xp = torch
            self.linalg = torch.linalg
        else:
            raise ValueError(f"Unknown backend {name}")

    # --- core ops ---
    def asarray(self, x):
        if self.name == "torch":
            return self.xp.as_tensor(x)
        return self.xp.asarray(x)

    def zeros(self, *args, dtype=None):
        return self.xp.zeros(*args, dtype=dtype)

    def ones(self, *args, dtype=None):
        return self.xp.ones(*args, dtype=dtype)

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

    def transpose(self, x, axes):
        return x.transpose(axes)

    def reshape(self, x, shape):
        return x.reshape(shape)

    def tensordot(self, a, b, axes):
        return self.xp.tensordot(a, b, axes=axes)

    def einsum(self, subscripts: str, *operands):
        return self.xp.einsum(subscripts, *operands)

    def qr(self, a):
        #if self.name == "torch":
        #    return self.linalg.qr(a)
        return self.linalg.qr(a)

    def svd(self, a, full_matrices=False):
        # returns U, S, Vh
        #if self.name == "torch":
        #    return self.linalg.svd(a, full_matrices=full_matrices)
        return self.linalg.svd(a, full_matrices=full_matrices)

    def eigh(self, a): return self.linalg.eigh(a)

    def eigvalsh(self, a):
        #if self.name == "torch":
        #    return self.linalg.eigvalsh(a)
        return self.linalg.eigvalsh(a)

    def exp(self, x): return self.xp.exp(x)

    def log(self, x):
        return self.xp.log(x)

    def clip(self, x, a_min, a_max):
        if self.name == "torch":
            return self.xp.clamp(x, min=a_min, max=a_max)
        return self.xp.clip(x, a_min=a_min, a_max=a_max)

    def sum(self, x): return self.xp.sum(x)

    def trace(self, a):
        return self.xp.trace(a)

    def vdot(self, a, b):
        return self.xp.vdot(a.reshape(-1), b.reshape(-1))

    def norm(self, x):
        return self.xp.sqrt(self.xp.real(self.vdot(x, x)))

    def to_numpy(self, x):
        # handy for debugging / printing
        if self.name == "numpy":
            return x
        if self.name == "jax":
            import numpy as np
            return np.asarray(x)
        if self.name == "torch":
            return x.detach().cpu().numpy()
        raise ValueError("Unknown backend")
