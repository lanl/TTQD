

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Tuple
import string
import uuid, itertools
from ttqd.linalg import backend

# ============================================================
# Tensor + named-index contraction
# ============================================================

Ix = str


def _prod(iterable):
    """Compute product of elements in iterable (like math.prod)."""
    prod = 1
    for x in iterable:
        prod *= x
    return prod


# Single-character index generator (prefer short names for better readability)
_SINGLE_CHAR_INDICES = (
    list(string.ascii_lowercase) +              # a-z
    list(string.ascii_uppercase) +              # A-Z
    ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ',    # Greek lowercase
     'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π',
     'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω'] +
    ['Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ',    # Greek uppercase
     'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π',
     'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω']
)

_RAND_PREFIX = str(uuid.uuid4())[:5]
# but then make the list orderable to help contraction caching
_RAND_ALPHABET = string.ascii_uppercase + string.ascii_lowercase
RAND_UUIDS = map(
    "".join,
    itertools.chain.from_iterable(
        itertools.product(_RAND_ALPHABET, repeat=repeat)
        for repeat in itertools.count(5)
    ),
)

class IndexContainer:
    """Container to track used indices and generate new unique ones."""

    def __init__(self):
        self.used_indices = set()

    def add(self, index: str) -> None:
        """Mark an index as used."""
        self.used_indices.add(index)

    def add_many(self, indices) -> None:
        """Mark multiple indices as used."""
        self.used_indices.update(indices)

    def get_uid(self, base: str = "") -> str:
        r"""
        Generate a unique index name.

        First tries single characters (a-z, A-Z, Greek), then falls back
        to longer strings if necessary.

        Parameters
        ----------
        base : str, optional
            Base prefix for the generated index (e.g., 'bond_', 'phys_').

        Returns
        -------
        str
            A unique index name not in the container's used set.
        """
        # Try single characters first
        for char in _SINGLE_CHAR_INDICES:
            candidate = f"{base}{char}" if base else char
            if candidate not in self.used_indices:
                self.used_indices.add(candidate)
                return candidate

        # Fall back to numbered suffixes if all single chars are exhausted
        i = 0
        while True:
            candidate = f"{base}_idx_{i}" if base else f"idx_{i}"
            if candidate not in self.used_indices:
                self.used_indices.add(candidate)
                return candidate
            i += 1


# Global index container for backward compatibility
_GLOBAL_INDEX_CONTAINER = IndexContainer()


def get_uid(base: str = "", container: "IndexContainer" = None) -> str:
    """
    Generate a unique index name.

    First tries single characters (a-z, A-Z, Greek), then falls back
    to longer strings if necessary.

    Parameters
    ----------
    base : str, optional
        Base prefix for the generated index (e.g., 'bond_;, 'phys_').
    container : IndexContainer, optional
        Container to track used indices. If None, uses global container.

    Returns
    -------
    str
        A unique index name.
    """
    return f"{base}_{_RAND_PREFIX}{next(RAND_UUIDS)}"
    if container is None:
        container = _GLOBAL_INDEX_CONTAINER
    return container.get_uid(base)


def add_bond(
    T1: "Tensor",
    T2: "Tensor",
    size=1,
    name=None,
    axis1=0,
    axis2=0,
    container: "IndexContainer" = None,
):
    """Inplace addition of a new bond between tensors ``T1`` and ``T2``. The
    size of the new bond can be specified, in which case the new array parts
    will be filled with zeros.

    Parameters
    ----------
    T1 : Tensor
        First tensor to modify.
    T2 : Tensor
        Second tensor to modify.
    size : int, optional
        Size of the new dimension.
    name : str, optional
        Name for the new index.
    axis1 : int, optional
        Position on the first tensor for the new dimension.
    axis2 : int, optional
        Position on the second tensor for the new dimension.
    container : IndexContainer, optional
        Container to track used indices. If None, uses global container.
    """
    if container is None:
        container = _GLOBAL_INDEX_CONTAINER

    # Register existing indices with the container
    container.add_many(T1.inds)
    container.add_many(T2.inds)

    if name is None:
        name = get_uid(container=container)
    # print(f"Adding bond with name '{name}' between tensors with indices {T1.inds} and {T2.inds}")
    # add new index to each tensor; add_index handles inplace mutation
    T1.add_index(name, size=size, pos=axis1)
    T2.add_index(name, size=size, pos=axis2)
    # function modifies T1 and T2 in place and does not return a value

@dataclass
class Tensor:
    r"""Lightweight tensor container with *named* indices.

    The :class:`Tensor` holds an array-like ``data`` object together with a
    tuple of index names (``inds``) that identify each axis.  Tags can be
    attached as a :class:`frozenset` of strings for bookkeeping or selection
    purposes.  The class is intentionally simple: operations such as
    contraction, reindexing, and transposition are provided in a minimal
    form suitable for building up more sophisticated tensor network code.

    The implementation is backend-agnostic in that the stored ``data`` may
    be a NumPy, JAX, or PyTorch array; the helper ``Backend`` wrapper is used
    to perform operations without committing to a specific numerical library.
    """
    data: Any
    inds: Tuple[Ix, ...]
    tags: FrozenSet[str] = frozenset()

    def __post_init__(self):
        # ensure consistency between data dimensionality and index list
        nd = getattr(self.data, "ndim", None)
        if nd is not None and nd != len(self.inds):
            raise ValueError(f"data.ndim={nd} does not match len(inds)={len(self.inds)}")
        # enforce frozen tags
        if not isinstance(self.tags, frozenset):
            object.__setattr__(self, "tags", frozenset(self.tags))

    # convenience properties
    @property
    def shape(self):
        return getattr(self.data, "shape", None)

    @property
    def ndim(self):
        return getattr(self.data, "ndim", None)

    @property
    def size(self):
        shp = self.shape
        return None if shp is None else int(_prod(shp))

    @property
    def dtype(self):
        return getattr(self.data, "dtype", None)

    def __repr__(self):
        return (
            f"Tensor(shape={self.shape}, inds={self.inds}, tags={list(self.tags)})"
        )

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        if self.inds != other.inds or self.tags != other.tags:
            return False
        # compare data numerically if possible
        try:
            import numpy as _np
            return _np.allclose(self.data, other.data)
        except Exception:
            return self.data == other.data

    # ------------------------------------------------------------------
    # convenience mutation helpers
    # ------------------------------------------------------------------
    def with_data(self, data: Any) -> "Tensor":
        """Return a new ``Tensor`` with identical metadata but updated data."""
        return Tensor(data, self.inds, self.tags)

    def set_data(self, data: Any) -> None:
        """In-place update of the underlying array.

        This method exists primarily for algorithms that repeatedly modify a
        tensor in-place, e.g. during DMRG sweeps.  The original class was a
        frozen dataclass which raised ``FrozenInstanceError`` when setting
        ``tensor.data`` directly.
        """
        object.__setattr__(self, "data", data)

    def add_index(self, label, size, pos=None, inplace=True):
        r"""Add new index

        Parameters
        ----------
        label : str
            Name of the new index to attach.
        size : int
            Length of the new axis.
        pos : Optional[int]
            Where to insert the new axis in the existing order.  If ``None``
            (default) the axis is appended at the end.

        Returns
        -------
        Tensor
            New tensor with the added index unless ``inplace`` is ``True``
            (default) in which case the original tensor is mutated and
            returned.  If ``label`` already exists, the original tensor is
            returned unchanged regardless of the ``inplace`` flag.
        """
        # if index already present, do nothing
        if label in self.inds:
            return self

        be = backend

        arr = be.asarray(self.data)
        # compute new shape and index position
        old_shape = list(arr.shape)
        new_axis = arr.ndim if pos is None else int(pos)
        assert 0 <= new_axis <= arr.ndim, "pos out of range"
        new_shape = old_shape.copy()
        new_shape.insert(new_axis, int(size))

        new_data = be.zeros(tuple(new_shape), dtype=arr.dtype)

        # build slice tuple: slices for every old dim, then 0 at new axis
        idx = [slice(None)] * (arr.ndim + 1)
        idx[new_axis] = 0
        new_data[tuple(idx)] = arr

        new_inds = list(self.inds)
        new_inds.insert(new_axis, label)
        if inplace:
            # Tensor is a frozen dataclass, so we cannot assign attributes directly.
            # use object.__setattr__ to mutate in-place when requested.
            object.__setattr__(self, "data", new_data)
            object.__setattr__(self, "inds", tuple(new_inds))
            return self
        else:
            return Tensor(new_data, tuple(new_inds), self.tags)

    def add_tags(self, *tags: str) -> "Tensor":
        r"""Return a new tensor with additional tags.

        Parameters
        ----------
        *tags : str
            Tags to add to the existing tag set.  Tags are stored as a
            :class:`frozenset` and duplicate tags are ignored.

        Returns
        -------
        Tensor
            A new :class:`Tensor` instance sharing the same data and indices
            but with the expanded tag set.
        """
        return Tensor(self.data, self.inds, self.tags.union(tags))

    def reindex(self, mapping: Dict[Ix, Ix]) -> "Tensor":
        r"""Change the names of existing indices.

        Parameters
        ----------
        mapping : Dict[Ix, Ix]
            Dictionary mapping old index names to new names.  Any index not
            present in the mapping is left unchanged.

        Returns
        -------
        Tensor
            A new tensor with the same data and tags but with renamed indices.
        """
        return Tensor(self.data, tuple(mapping.get(ix, ix) for ix in self.inds), self.tags)

    def transpose_to(self, new_inds: Tuple[Ix, ...]) -> "Tensor":
        r"""Return a permuted tensor with a new index ordering.

        Parameters
        ----------
        new_inds : Tuple[Ix, ...]
            Desired ordering of the indices.  This must be a permutation of
            the current ``self.inds``; otherwise :class:`ValueError` will be
            raised by the attempt to compute ``index()``.

        Returns
        -------
        Tensor
            Tensor with axes permuted to match ``new_inds`` and the same
            data type and tags as the original.  If ``new_inds`` is identical
            to ``self.inds`` the original tensor is returned unchanged.
        """
        if new_inds == self.inds:
            return self
        perm = [self.inds.index(ix) for ix in new_inds]
        return Tensor(backend.transpose(self.data, perm), new_inds, self.tags)


def contract_two(t1: Tensor, t2: Tensor) -> Tensor:
    r"""Contract two tensors by summing over shared index names.

    The routine identifies all index names appearing in both ``t1`` and
    ``t2`` and contracts the corresponding axes using the supplied
    ``backend``'s ``tensordot`` implementation.  The ordering of the
    remaining open axes in the returned tensor is: all axes from ``t1`` that
    were not contracted followed by the remaining axes from ``t2``.

    Parameters
    ----------
    t1, t2 : Tensor
        Input tensors to contract.

    Returns
    -------
    Tensor
        Resulting tensor after contraction.  Tags from both inputs are
        unioned.  If the two tensors share no indices, an outer product is
        performed and the indices simply concatenate.
    """
    shared = [ix for ix in t1.inds if ix in t2.inds]
    if not shared:
        # outer product
        data = backend.tensordot(t1.data, t2.data, axes=0)
        inds = t1.inds + t2.inds
        return Tensor(data, inds, t1.tags.union(t2.tags))

    axes1 = [t1.inds.index(ix) for ix in shared]
    axes2 = [t2.inds.index(ix) for ix in shared]
    data = backend.tensordot(t1.data, t2.data, axes=(axes1, axes2))
    new_inds = tuple(ix for ix in t1.inds if ix not in shared) + tuple(ix for ix in t2.inds if ix not in shared)
    return Tensor(data, new_inds, t1.tags.union(t2.tags))


# simple unittest runner
if __name__ == "__main__":
    import os, sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, root)
    import unittest
    loader = unittest.defaultTestLoader
    suite = loader.discover(start_dir=os.path.join(root, "tests"), pattern="test_tensor.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())

    # test reindex
    D = A.reindex({"i":"x","j":"y"})
    assert D.inds == ("x","y")
    assert np.allclose(D.data, A.data)

    # test transpose_to
    T = A.transpose_to(("j","i"))
    assert T.inds == ("j","i")
    assert T.data.shape == (3,2)
    assert np.allclose(T.data, A.data.T)

    # test contract_two
    A1 = Tensor(np.ones((2,2)), ("a","b"))
    A2 = Tensor(np.arange(6).reshape(2,3), ("b","c"))
    C1 = contract_two(A1, A2)
    _print("C1 inds", C1.inds)
    _print("C1 data", C1.data)
    # resulting shape should be (2,3) contracting over b
    assert C1.inds == ("a","c")
    assert C1.data.shape == (2,3)

    print("All tensor tests passed.")

