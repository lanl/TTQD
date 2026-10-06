r"""
Matrix Product operator (MPO) module
------------------------------------

This module provides the lightweight :class:`MPO` container used by the chain
solvers and a two-site cache for TDVP-style updates.
"""

from __future__ import annotations
from typing import Any, Callable, Iterator, Sequence

from ttqd.linalg import backend
from ttqd.operator.spinoperator import build_h2_list_xxz, build_mpo_xxz, paulis

class MPO:
    r"""
    Matrix product operator container.

    An MPO is stored as a sequence of rank-four tensors
    ``W[i][w_left, w_right, p_out, p_in]``.  The virtual dimensions must match
    between neighboring tensors:

    .. math::

        W_i.shape[1] = W_{i+1}.shape[0].

    Diagrammatically:

    .. math::

        & p_0\quad\quad\;\; p_1 \quad\quad\;\;  p_2\quad\quad\;\;  \cdots\quad\quad\; p_N \\
        & \;|\quad\quad\quad|\quad\quad\quad\; |\quad\quad\quad\; |\quad\quad\quad | \\
        & M_0 --          M_1 --            M_2 --         \cdots -- M_N \\
        & \;|\quad\quad\quad|\quad\quad\quad\; |\quad\quad\quad\; |\quad\quad\quad | \\
        & q_0\quad\quad\;\; q_1 \quad\quad\;\;  q_2\quad\quad\;\;  \cdots\quad\quad\; q_N

    Parameters
    ----------
    Ws : Sequence[Any]
        Local MPO tensors. Elements are converted to the configured TTQD
        backend with ``backend.asarray``.
    validate : bool, optional
        If ``True``, check that the sequence is non-empty, every tensor is
        rank four, and adjacent virtual bond dimensions match.
    """

    def __init__(self, Ws: Sequence[Any], validate: bool = True):
        self.be = backend
        self.Ws = [self.be.asarray(Wi) for Wi in Ws]
        self.L = len(self.Ws)
        if validate:
            self.validate()

    def validate(self) -> None:
        """Validate tensor ranks and virtual-bond compatibility."""
        if self.L == 0:
            raise ValueError("MPO requires at least one tensor")

        for i, W in enumerate(self.Ws):
            if W.ndim != 4:
                raise ValueError(
                    f"MPO tensor {i} must have rank 4, got shape {tuple(W.shape)}"
                )

        for i, (Wl, Wr) in enumerate(zip(self.Ws[:-1], self.Ws[1:])):
            if Wl.shape[1] != Wr.shape[0]:
                raise ValueError(
                    "MPO virtual-bond mismatch between tensors "
                    f"{i} and {i + 1}: {Wl.shape[1]} != {Wr.shape[0]}"
                )

    @property
    def phys_dim(self) -> int:
        """Physical output dimension of the first site."""
        return int(self.Ws[0].shape[2])

    @property
    def phys_dims(self) -> tuple[tuple[int, int], ...]:
        """Physical ``(d_out, d_in)`` dimensions for every site."""
        return tuple((int(W.shape[2]), int(W.shape[3])) for W in self.Ws)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        """Virtual bond dimensions, including left and right boundaries."""
        return (int(self.Ws[0].shape[0]),) + tuple(int(W.shape[1]) for W in self.Ws)

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Tensor shapes as plain Python tuples."""
        return tuple(tuple(int(dim) for dim in W.shape) for W in self.Ws)

    @property
    def is_open_boundary(self) -> bool:
        """Whether the first left and last right virtual bonds are dimension 1."""
        return self.Ws[0].shape[0] == 1 and self.Ws[-1].shape[1] == 1

    def __len__(self) -> int:
        return self.L

    def __iter__(self) -> Iterator[Any]:
        return iter(self.Ws)

    def __getitem__(self, item):
        return self.Ws[item]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(L={self.L}, "
            f"bond_dims={self.bond_dims}, phys_dims={self.phys_dims})"
        )


class TwoSiteMPOCache:
    r"""
    Cache two-site MPO tensors.

    The cached object is

    .. math::

        W^{[i,i+1]}_{xzqspr}
        = \sum_y W^{[i]}_{xyqp} W^{[i+1]}_{yzsr}.

    This is useful in two-site TDVP/DMRG sweeps, where the same pair tensor may
    be applied many times inside Krylov iterations.
    """

    def __init__(self, mpo: MPO, contract: Callable):
        self.mpo = mpo
        self.contract = contract
        self._W2: dict[int, Any] = {}

    def get(self, i: int) -> Any:
        """Return the contracted two-site MPO tensor for sites ``i`` and ``i+1``."""
        if not (0 <= i < self.mpo.L - 1):
            raise IndexError(f"two-site MPO index {i} out of range for L={self.mpo.L}")
        if i in self._W2:
            return self._W2[i]
        W0 = self.mpo.Ws[i]
        W1 = self.mpo.Ws[i + 1]
        W2 = self.contract("xyqp,yzsr->xzqspr", W0, W1)
        self._W2[i] = W2
        return W2

    def clear(self) -> None:
        """Drop cached tensors, for example after replacing ``mpo.Ws`` in place."""
        self._W2.clear()


if __name__ == "__main__":
    L = 8
    h2_list = build_h2_list_xxz(L, Jx=1.0, Jy=1.0, Jz=1.0, hz=0.3)
    print(len(h2_list), h2_list[0].shape)  # (L-1, (4,4))

    Ws = build_mpo_xxz(L, Jx=1.0, Jy=1.0, Jz=1.0, hz=0.3)
    print(len(Ws), Ws[0].shape, Ws[1].shape, Ws[-1].shape)
    # -> (L, (1,5,2,2), (5,5,2,2), (5,1,2,2))
