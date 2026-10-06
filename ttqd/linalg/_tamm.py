from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

PYTAMM_AVAILABLE = False
PYTAMM_GPU_AVAILABLE = False
PYTAMM_MPI_AVAILABLE = False
PYTAMM_INFO: Dict[str, Any] = {}

pytamm = None
so_path: Optional[Path] = None
prefix: Optional[Path] = None

_INITIALIZED = False
_DEFAULT_PROC_GROUP = None
_DEFAULT_EXECUTION_CONTEXT = None
_PYTAMM_EXTENSIONS = ("pytamm*.so", "pytamm*.pyd", "pytamm*.dll")


def _split_paths(value: str) -> List[Path]:
    return [Path(p).expanduser() for p in value.split(os.pathsep) if p]


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out = []
    for path in paths:
        try:
            key = str(path.expanduser().resolve())
        except OSError:
            key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(Path(key))
    return out


def _prefix_python_dirs(root: Path) -> List[Path]:
    root = root.expanduser()
    dirs = [
        root,
        root / "lib",
        root / "lib64",
        root / "python",
    ]
    dirs.extend((root / "lib").glob("python*/site-packages"))
    dirs.extend((root / "lib64").glob("python*/site-packages"))
    return [d for d in dirs if d.is_dir()]


def _candidate_prefixes() -> List[Path]:
    roots = []

    for name in ("TTQD_TAMM_PREFIX", "TAMM_PREFIX", "TAMM_ROOT"):
        value = os.environ.get(name)
        if value:
            roots.extend(_split_paths(value))

    tamm_dir = os.environ.get("TAMM_DIR")
    if tamm_dir:
        # TAMM_DIR is often share/cmake/tamm. Walk back to the install prefix.
        p = Path(tamm_dir).expanduser()
        roots.append(p)
        for parent in p.parents:
            if parent.name in {"share", "cmake"}:
                continue
            if (parent / "lib").is_dir() or (parent / "lib64").is_dir():
                roots.append(parent)
                break

    package_root = Path(__file__).resolve().parents[1]
    local_lib = package_root / "lib"
    roots.extend(
        [
            local_lib / "tamm-install",
            local_lib / "tamm" / "install",
            local_lib / "tamm",
            local_lib / "build" / "tamm-install",
            local_lib / "build" / "_deps" / "tamm-install",
        ]
    )
    roots.extend(sorted(local_lib.glob("build*/tamm-install")))
    roots.extend(sorted(local_lib.glob("*/tamm-install")))

    return _dedupe_paths(root for root in roots if root.exists())


def _candidate_python_dirs(extra_paths: Optional[Iterable[os.PathLike[str] | str]] = None) -> List[Path]:
    dirs: List[Path] = []

    if extra_paths:
        dirs.extend(Path(p).expanduser() for p in extra_paths)

    for name in ("TTQD_TAMM_PYTHONPATH", "PYTHONPATH"):
        value = os.environ.get(name)
        if value:
            dirs.extend(_split_paths(value))

    for root in _candidate_prefixes():
        dirs.extend(_prefix_python_dirs(root))

    dirs.extend(Path(p).expanduser() for p in sys.path if p)
    return _dedupe_paths(d for d in dirs if d.is_dir())


def _prepend_runtime_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        try:
            os.add_dll_directory(str(path))
        except (AttributeError, FileNotFoundError, OSError):
            pass
        return

    var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    parts = os.environ.get(var, "").split(os.pathsep) if os.environ.get(var) else []
    spath = str(path)
    if spath not in parts:
        os.environ[var] = os.pathsep.join([spath] + parts)


def _find_pytamm_extension(paths: Iterable[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    for directory in paths:
        for pattern in _PYTAMM_EXTENSIONS:
            matches = sorted(directory.glob(pattern))
            if not matches:
                continue
            ext = matches[0].resolve()
            root = _infer_prefix_from_extension(ext)
            return ext, root
    return None, None


def _infer_prefix_from_extension(ext: Path) -> Path:
    parent = ext.parent
    if parent.name in {"lib", "lib64", "python"}:
        return parent.parent
    for ancestor in [parent, *parent.parents]:
        if (ancestor / "share" / "cmake" / "tamm" / "tamm-config.cmake").exists():
            return ancestor
    return parent


def _read_build_flags(root: Path) -> Optional[Dict[str, Optional[bool]]]:
    candidates = [
        root / "share" / "cmake" / "tamm" / "tamm-config.cmake",
        root / "share" / "cmake" / "TAMM" / "TAMMConfig.cmake",
        root / "share" / "cmake" / "TAMM" / "tamm-config.cmake",
    ]
    cfg = next((p for p in candidates if p.exists()), None)
    if cfg is None:
        return None

    txt = cfg.read_text(errors="ignore")

    def flag(name: str) -> Optional[bool]:
        patterns = (
            rf"\bTAMM_HAS_{name}\s+([A-Za-z0-9_]+)",
            rf"\btamm_HAS_{name}\s+([A-Za-z0-9_]+)",
            rf"\bHAS_{name}\s+([A-Za-z0-9_]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, txt)
            if match:
                value = match.group(1).upper()
                if value in {"1", "ON", "TRUE", "YES"}:
                    return True
                if value in {"0", "OFF", "FALSE", "NO"}:
                    return False
        return None

    return {
        "cuda": flag("CUDA"),
        "hip": flag("HIP"),
        "dpcpp": flag("DPCPP"),
        "python": flag("PYTHON"),
    }


def _linked_libs(path: Path) -> str:
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["otool", "-L", str(path)], text=True)
        elif sys.platform.startswith("win"):
            return ""
        else:
            out = subprocess.check_output(["ldd", str(path)], text=True)
        return out.lower()
    except Exception:
        return ""


def _prepare_import_paths(paths: Iterable[Path]) -> None:
    for directory in reversed(list(paths)):
        sdir = str(directory)
        if sdir not in sys.path:
            sys.path.insert(0, sdir)

    for root in _candidate_prefixes():
        for libdir in (root / "lib", root / "lib64"):
            if libdir.is_dir():
                _prepend_runtime_path(libdir)


def _try_import(extra_paths: Optional[Iterable[os.PathLike[str] | str]] = None):
    global pytamm, so_path, prefix

    paths = _candidate_python_dirs(extra_paths)
    _prepare_import_paths(paths)

    try:
        module = importlib.import_module("pytamm")
    except Exception as exc:
        ext, root = _find_pytamm_extension(paths)
        so_path = ext
        prefix = root
        PYTAMM_INFO["import_error"] = repr(exc)
        if ext is not None:
            PYTAMM_INFO["extension_path"] = str(ext)
        return None

    pytamm = module
    so_path = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
    prefix = _infer_prefix_from_extension(so_path) if so_path is not None else None
    PYTAMM_INFO.pop("import_error", None)
    if so_path is not None:
        PYTAMM_INFO["extension_path"] = str(so_path)
    if prefix is not None:
        PYTAMM_INFO["prefix"] = str(prefix)
    return module


def _refresh_capabilities() -> None:
    global PYTAMM_AVAILABLE, PYTAMM_GPU_AVAILABLE, PYTAMM_MPI_AVAILABLE

    PYTAMM_AVAILABLE = pytamm is not None
    PYTAMM_GPU_AVAILABLE = False
    PYTAMM_MPI_AVAILABLE = False

    if prefix is not None:
        flags = _read_build_flags(prefix)
        if flags:
            PYTAMM_INFO["build_flags"] = flags
            PYTAMM_GPU_AVAILABLE = bool(flags.get("cuda") or flags.get("hip") or flags.get("dpcpp"))

    if so_path is not None:
        libs = _linked_libs(so_path)
        PYTAMM_INFO["linked_libs_checked"] = bool(libs)
        PYTAMM_MPI_AVAILABLE = "libmpi" in libs or "mpi" in libs
        if any(x in libs for x in ("libcuda", "libcudart", "libcublas", "libhip", "libamdhip64", "libsycl")):
            PYTAMM_GPU_AVAILABLE = True


def load(extra_paths: Optional[Iterable[os.PathLike[str] | str]] = None):
    """Import and return the ``pytamm`` extension module.

    The loader searches normal Python import paths, ``TTQD_TAMM_PREFIX`` /
    ``TAMM_PREFIX``, and the default ``ttqd/lib/build/tamm-install`` layout
    produced by ``ttqd/lib/CMakeLists.txt``.
    """

    global pytamm
    if pytamm is None:
        pytamm = _try_import(extra_paths)
        _refresh_capabilities()

    if pytamm is None:
        hint = (
            "pytamm could not be imported. Build TAMM with "
            "`cmake -S ttqd/lib -B ttqd/lib/build && cmake --build ttqd/lib/build --target pytamm`, "
            "or set TTQD_TAMM_PREFIX to the TAMM install prefix."
        )
        raise ImportError(hint)

    return pytamm


def is_available() -> bool:
    return PYTAMM_AVAILABLE


def initialize(args: Optional[Sequence[str]] = None, is_mpi_tm: bool = False) -> None:
    """Initialize TAMM/MPI once for this Python process."""

    global _INITIALIZED
    if _INITIALIZED:
        return
    module = load()
    module.initialize(list(args or ["python"]), is_mpi_tm)
    _INITIALIZED = True


def finalize(tamm_mpi_finalize: bool = True) -> None:
    """Finalize TAMM if this module initialized it."""

    global _INITIALIZED, _DEFAULT_PROC_GROUP, _DEFAULT_EXECUTION_CONTEXT
    if not _INITIALIZED:
        return
    module = load()
    module.finalize(tamm_mpi_finalize)
    _INITIALIZED = False
    _DEFAULT_PROC_GROUP = None
    _DEFAULT_EXECUTION_CONTEXT = None


@contextmanager
def runtime(args: Optional[Sequence[str]] = None, is_mpi_tm: bool = False,
            finalize_mpi: bool = True) -> Iterator[Any]:
    """Context manager that initializes TAMM and yields the ``pytamm`` module."""

    initialize(args=args, is_mpi_tm=is_mpi_tm)
    try:
        yield load()
    finally:
        finalize(finalize_mpi)


def default_execution_context():
    """Return a cached TAMM world execution context."""

    global _DEFAULT_PROC_GROUP, _DEFAULT_EXECUTION_CONTEXT
    initialize()
    module = load()
    if _DEFAULT_EXECUTION_CONTEXT is None:
        _DEFAULT_PROC_GROUP = module.ProcGroup.create_world_coll()
        _DEFAULT_EXECUTION_CONTEXT = module.ExecutionContext(
            _DEFAULT_PROC_GROUP,
            module.DistributionKind.nw,
            module.MemoryManagerKind.ga,
        )
    return _DEFAULT_EXECUTION_CONTEXT


def from_numpy(array: Any, ec: Any = None, tilesize: int = 32, spaces: Any = None):
    """Create an allocated TAMM tensor from a NumPy-compatible array."""

    module = load()
    ec = default_execution_context() if ec is None else ec
    if spaces is None:
        return module.from_numpy(array, ec, tilesize=tilesize)
    return module.from_numpy(array, ec, tilesize=tilesize, spaces=spaces)


def to_numpy(tensor: Any, dtype: Any = None):
    """Convert a TAMM tensor or labeled tensor to a NumPy array."""

    module = load()
    if dtype is None:
        return module.to_numpy(tensor)
    return module.to_numpy(tensor, dtype=dtype)


def _parse_einsum(equation: str, noperands: int) -> Tuple[List[str], str]:
    if "..." in equation:
        raise NotImplementedError("TAMM einsum wrapper does not support ellipsis")

    lhs, sep, rhs = equation.replace(" ", "").partition("->")
    inputs = lhs.split(",") if lhs else []
    if len(inputs) != noperands:
        raise ValueError(f"einsum expected {len(inputs)} operands, got {noperands}")

    for labels in inputs:
        if len(set(labels)) != len(labels):
            raise NotImplementedError("TAMM einsum wrapper does not support repeated labels within one operand")

    if sep:
        output = rhs
    else:
        counts: Dict[str, int] = {}
        for labels in inputs:
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        output = "".join(sorted(label for label, count in counts.items() if count == 1))

    return inputs, output


def _is_tamm_tensor(obj: Any) -> bool:
    return callable(getattr(obj, "tiled_index_spaces", None)) and callable(obj)


def _deallocate_if_needed(*tensors: Any) -> None:
    for tensor in tensors:
        try:
            if tensor is not None and tensor.is_allocated():
                tensor.deallocate()
        except Exception:
            pass


def einsum(equation: str, *operands: Any, ec: Any = None, tilesize: int = 32,
           return_tensor: bool = False):
    """Evaluate a simple explicit einsum through TAMM.

    Operands may be NumPy arrays or already allocated TAMM tensors. The result
    is converted back to NumPy by default; pass ``return_tensor=True`` to keep
    the allocated TAMM result tensor.
    """

    import numpy as np

    module = load()
    ec = default_execution_context() if ec is None else ec
    inputs, output = _parse_einsum(equation, len(operands))

    tensors = []
    owned = []
    numpy_operands = []
    for operand in operands:
        if _is_tamm_tensor(operand):
            tensors.append(operand)
        else:
            arr = np.asarray(operand)
            numpy_operands.append(arr)
            tensor = from_numpy(arr, ec=ec, tilesize=tilesize)
            tensors.append(tensor)
            owned.append(tensor)

    label_spaces: Dict[str, Any] = {}
    for labels, tensor in zip(inputs, tensors):
        spaces = tuple(tensor.tiled_index_spaces())
        if len(labels) != len(spaces):
            raise ValueError(f"labels '{labels}' do not match tensor rank {len(spaces)}")
        for label, space in zip(labels, spaces):
            label_spaces.setdefault(label, space)

    try:
        result_spaces = [label_spaces[label] for label in output]
    except KeyError as exc:
        raise ValueError(f"output label {exc.args[0]!r} does not appear in the inputs") from exc

    result_dtype = np.result_type(*numpy_operands) if numpy_operands else np.dtype("float64")
    result_cls = module.TensorComplexDouble if np.issubdtype(result_dtype, np.complexfloating) else module.TensorDouble
    result = result_cls(result_spaces)

    scheduler = module.Scheduler(ec)
    scheduler.allocate(result).execute()

    labeled_operands = [tensor(*tuple(labels)) for tensor, labels in zip(tensors, inputs)]
    expr = labeled_operands[0]
    for labeled in labeled_operands[1:]:
        expr = expr * labeled

    module.Scheduler(ec)(result(*tuple(output)), "=", expr).execute()
    ec.flush_and_sync()

    if return_tensor:
        _deallocate_if_needed(*owned)
        return result

    try:
        return result.to_numpy()
    finally:
        _deallocate_if_needed(result, *owned)


class _TammRandom:
    def __init__(self, np_module):
        self.np = np_module
        self._rng = np_module.random.default_rng()

    def seed(self, seed: Optional[int]) -> None:
        self._rng = self.np.random.default_rng(seed)

    def normal(self, shape, mean=0.0, std=1.0, dtype="float32"):
        return self._rng.normal(mean, std, size=shape).astype(self.np.dtype(dtype), copy=False)

    def uniform(self, shape, low=0.0, high=1.0, dtype="float32"):
        return self._rng.uniform(low, high, size=shape).astype(self.np.dtype(dtype), copy=False)

    def randint(self, shape, low: int, high: Optional[int] = None, dtype="int64"):
        return self._rng.integers(low, high, size=shape).astype(self.np.dtype(dtype), copy=False)


class TammBackend:
    """NumPy-compatible backend that routes ``einsum`` through TAMM."""

    name = "tamm"

    def __init__(self, device: str = "cpu", tilesize: int = 32):
        import numpy as np

        normalized_device = (device or "cpu").lower()
        if normalized_device not in {"cpu", "gpu", "cuda"}:
            raise ValueError(f"Unsupported TAMM device {device!r}")
        if normalized_device in {"gpu", "cuda"} and not PYTAMM_GPU_AVAILABLE:
            raise RuntimeError("TAMM GPU device requested, but pytamm was not built with CUDA/HIP/DPCPP support")
        self.np = self.xp = np
        self.device = device
        self.tilesize = int(tilesize)
        self._random = _TammRandom(np)
        self._linalg = np.linalg
        load()
        initialize()

    def available_devices(self):
        return ["cpu", "gpu"] if PYTAMM_GPU_AVAILABLE else ["cpu"]

    def dtype(self, dtype):
        return self.np.dtype("float32" if dtype is None else dtype)

    def asarray(self, x, dtype=None):
        return self.np.asarray(x, dtype=None if dtype is None else self.dtype(dtype))

    def zeros(self, shape, dtype="float32"):
        return self.np.zeros(shape, dtype=self.dtype(dtype))

    def ones(self, shape, dtype="float32"):
        return self.np.ones(shape, dtype=self.dtype(dtype))

    def empty(self, shape, dtype="float32"):
        return self.np.empty(shape, dtype=self.dtype(dtype))

    def arange(self, start, stop=None, step=1, dtype="int64"):
        return self.np.arange(start, stop, step, dtype=self.dtype(dtype))

    def reshape(self, x, shape):
        return self.np.reshape(x, shape)

    def transpose(self, x, axes=None):
        return self.np.transpose(x, axes=axes)

    def sum(self, x, axis=None, keepdims=False):
        return self.np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return self.np.mean(x, axis=axis, keepdims=keepdims)

    def dot(self, a, b):
        return a @ b

    def matmul(self, a, b):
        return a @ b

    def tensordot(self, *args, **kwargs):
        return self.np.tensordot(*args, **kwargs)

    def einsum(self, subscripts: str, *operands):
        return einsum(subscripts, *operands, tilesize=self.tilesize)

    def to_numpy(self, x):
        return x

    def from_numpy(self, x, dtype=None):
        arr = self.np.asarray(x)
        if dtype is None:
            return arr
        return arr.astype(self.dtype(dtype), copy=False)

    def tamm_from_numpy(self, x, ec=None, tilesize: Optional[int] = None):
        return from_numpy(x, ec=ec, tilesize=self.tilesize if tilesize is None else tilesize)

    @property
    def random(self):
        return self._random

    def synchronize(self):
        default_execution_context().flush_and_sync()


_try_import()
_refresh_capabilities()
