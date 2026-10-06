r"""
Tree Tensor Network and Matrix Product State
============================================

(Refactored MPS module with BaseTreeTensorNetwork architecture)

The MPS is a special case of a general tree tensor network,
enabling flexible tree-based tensor network implementations.

MPS is:

.. math::
    &q_0 \qquad q_1 \qquad q_2 \qquad q_3 \qquad q_j \quad \cdots \qquad q_N\\
    &| \;\;\qquad| \;\; \qquad| \qquad\;\;| \qquad\;\;| \qquad\;| \qquad\;\;|\\
    &A_0\!-\!-\!A_1\!-\!-\!A_2\!-\!-\;A_3\!-\!-\!A_j\!-\!-\cdots\!-\!\!-A_N

which is stored as :math:`[A_0, A_1, \cdots, A_j, \cdots, A_N]`, and ech A is a 3-index tensor (1 physical
index and two virtual indices). In this package, the MPS tensor is ordered as
(vL, p, vR), i.e., left, physical and right virtual indices.

A compact tensor network skeleton with:
- Named indices + tags
- Base TensorNetwork container
- MPS with efficient 1-site and 2-site RDMs, entropy, mutual information, expvals
- TreeTensorNetwork (TTN) with exact message-passing RDMs on trees

This is a *framework skeleton* meant to be extended (canonical forms, compression, MPOs,
cotengra paths, etc.). It’s already useful for observables and entanglement on MPS/TTN.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union
import sys
import numpy as np
from ttqd.tensors import Tensor, contract_two
from ttqd.lib import logger
from ttqd.linalg import backend

# print("backend in mps module is:", backend)
# print("type of the array is: ", type(backend.ones(10)))

# ============================================================
# Entropy helpers
# ============================================================

def _as_density_matrix(rdm) -> Any:
    """
    Ensure rdm is 2D (matrix). If already 2D, return.
    If 4D (d,d,d,d), reshape to (d^2, d^2).
    """
    if getattr(rdm, "ndim", None) == 2:
        return rdm
    if getattr(rdm, "ndim", None) == 4:
        d1, d2, d3, d4 = rdm.shape
        if d1 != d3 or d2 != d4:
            raise ValueError("Expected (d,d,d,d) layout for 2-site RDM.")
        return backend.reshape(rdm, (d1 * d2, d1 * d2))
    raise ValueError(f"Unsupported rdm ndim={getattr(rdm, 'ndim', None)}")


def von_neumann_entropy(rdm, base: float = 2.0, eps: float = 1e-12) -> Any:
    r"""Compute the von Neumann entropy from density matrix:

    .. math::
        S(\rho) = -Tr \rho log \rho,

    computed from eigenvalues of :math:`\rho`
    """
    rho = _as_density_matrix(rdm)
    # symmetrize for numerical stability
    rhoH = 0.5 * (rho + backend.conj(backend.transpose(rho, (1, 0))))
    evals = backend.eigvalsh(rhoH)
    evals = backend.real(evals)
    # keep eigenvalues in physical range; zero modes should not contribute
    # to entropy, so mask them out rather than clipping them to eps.
    evals = backend.clip(evals, 0.0, 1.0)
    mask = evals > eps
    log_arg = backend.clip(evals, eps, 1.0)
    if base == 2.0:
        coeff = backend.log(log_arg) / backend.log(backend.asarray(2.0))
    else:
        coeff = backend.log(log_arg) / backend.log(backend.asarray(base))
    return -backend.sum((evals * coeff) * mask)


def mutual_information(rho_i, rho_j, rho_ij, base: float = 2.0) -> Any:
    return von_neumann_entropy(rho_i, base=base) + \
           von_neumann_entropy(rho_j, base=base) - \
           von_neumann_entropy(rho_ij, base=base)



# ============================================================
# Base TensorNetwork container
# ============================================================

class TensorNetwork:
    def __init__(self, **kwargs):
        self.bc = kwargs.get("boundary", "finite")
        self.backend = self.be = backend
        self.tensors: Dict[Any, Tensor] = {}
        self.threshold = kwargs.get("threshold", 1.e-10)
        self.stdout = sys.stdout
        self.verbose = kwargs.get("verbose", 3)

    def add_tensor(self, key: Any, tensor: Tensor) -> None:
        self.tensors[key] = tensor

    def __getitem__(self, key: Any) -> Tensor:
        return self.tensors[key]

    def __setitem__(self, key: Any, tensor: Tensor) -> None:
        self.tensors[key] = tensor

    def copy(self) -> "TensorNetwork":
        tn = self.__class__()
        tn.tensors = dict(self.tensors)
        return tn

    def select(self, tag: str) -> Dict[Any, Tensor]:
        return {k: t for k, t in self.tensors.items() if tag in t.tags}

    def all_inds(self) -> List[Ix]:
        out: List[Ix] = []
        for t in self.tensors.values():
            out.extend(t.inds)
        return out

    def open_inds(self) -> List[Ix]:
        c: Dict[Ix, int] = {}
        for ix in self.all_inds():
            c[ix] = c.get(ix, 0) + 1
        return [ix for ix, k in c.items() if k == 1]

    def contract_all(self) -> Tensor:
        """
        Naive sequential contraction. For high performance, replace this with:
          - opt_einsum / cotengra path finder
          - tree decomposition, slicing, etc.
        """
        ts = list(self.tensors.values())
        if not ts:
            raise ValueError("Empty network.")
        cur = ts[0]
        for t in ts[1:]:
            cur = contract_two(cur, t, self.backend)
        return cur

    def expval_bonds(self, op: List[Any]):
        r"""Calculate the expectation value of local operators along the bond"""
        raise NotImplementedError("Should be implemented in inherited class")

    def to_dense(self) -> Array:
        """Debug helper: contract tensor network state into a dense state vector."""
        raise NotImplementedError("Should be implemented in inherited class")

    def norm(self) -> Array:
        """contract tensor network state norm <TNS|TNS>"""
        raise NotImplementedError("Should be implemented in inherited class")

    def normalize(self) -> Array:
        """normalize tensor network states"""
        raise NotImplementedError("Should be implemented in inherited class")


#--------------------------------------------------
# MPS initilization functions for certain models
#--------------------------------------------------

# for ising model
def init_FM_MPS(L, d=2, bc='finite'):
    """Return a ferromagnetic MPS (= product state with all spins up)"""
    A = np.zeros([1, d, 1], dtype=float)
    A[0, 0, 0] = 1.
    S = np.ones([1], dtype=float)
    As = [A.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return MPS(As, bouncary=bc) #Ss

def init_AFM_MPS(L, d=2, bc='finite'):
    """Return a anti-ferromagnetic MPS (= product state with alternating spins up/dn)"""
    assert L % 2 == 0
    Aup = np.zeros([1, d, 1], dtype=float)
    Adn = np.zeros([1, d, 1], dtype=float)
    Aup[0, 0, 0] = 1.
    Adn[0, 1, 0] = 1.
    S = np.ones([1], dtype=float)
    As = [Aup.copy() if i % 2 == 0 else Adn.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return MPS(As, bouncary=bc) #Ss


def init_Neel_MPS(L, d=2, bc='finite'):
    """Return a Neel state MPS (= product state with alternating spins up  down up down... )"""
    S = np.ones([1], dtype=float)
    As = []
    for i in range(L):
        A = np.zeros([1, d, 1], dtype=float)
        if i % 2 == 0:
            A[0, 0, 0] = 1.
        else:
            A[0, -1, 0] = 1.
        As.append(A)
    Ss = [S.copy() for i in range(L)]
    return MPS(As, Ss, boundary=bc)


def basis_state(dim: int, n: int, dtype=np.complex128):
    """Return the computational basis vector ``|n>`` in a local space of size ``dim``."""
    if not (0 <= n < dim):
        raise ValueError(f"basis index {n} is outside the valid range [0, {dim})")
    vec = np.zeros(dim, dtype=dtype)
    vec[n] = 1.0
    return vec


def init_product_MPS(local_states: Sequence[Any], bc: str = "finite"):
    """Return a bond-dimension-1 MPS built from local state vectors.

    This helper supports mixed local dimensions, which is useful for chains such
    as ``[electronic, reaction-coordinate, bath_0, bath_1, ...]``.
    """
    As = []
    for i, state in enumerate(local_states):
        vec = np.asarray(state)
        if vec.ndim != 1:
            raise ValueError(f"local_states[{i}] must be a 1D state vector")
        As.append(vec.reshape(1, vec.shape[0], 1))
    return MPS(As, boundary=bc)




def split_truncate_theta(theta, chi_max, eps):
    """Split and truncate a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled ``vC``).

    Parameters
    ----------
    theta : np.Array[ndim=4]
        Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
    chi_max : int
        Maximum number of singular values to keep
    eps : float
        Discard any singular values smaller than that.

    Returns
    -------
    A : np.Array[ndim=3]
        Left-canonical matrix on site i, with legs ``vL, i, vC``
    S : np.Array[ndim=1]
        Singular/Schmidt values.
    B : np.Array[ndim=3]
        Right-canonical matrix on site j, with legs ``vC, j, vR``
    """
    import scipy

    chivL, dL, dR, chivR = theta.shape
    theta = np.reshape(theta, [chivL * dL, dR * chivR])
    X, Y, Z = scipy.linalg.svd(theta, full_matrices=False)
    # truncate
    chivC = min(chi_max, np.sum(Y > eps))
    assert chivC >= 1
    piv = np.argsort(Y)[::-1][:chivC]  # keep the largest `chivC` singular values
    X, Y, Z = X[:, piv], Y[piv], Z[piv, :]
    # renormalize
    S = Y / np.linalg.norm(Y)  # == Y/sqrt(sum(Y**2))
    # split legs of X and Z
    A = np.reshape(X, [chivL, dL, chivC])
    B = np.reshape(Z, [chivC, dR, chivR])
    return A, S, B


# ============================================================
# TTN Node for tree structures
# ============================================================

@dataclass
class TTNNode:
    """
    Tensor + metadata for tree tensor network nodes.
    - tensor: Tensor with named indices
    - phys_inds: physical index names (often 1 at leaves, none internal)
    - bonds are inferred by shared index names between connected nodes
    """
    tensor: Tensor
    phys_inds: Tuple[Ix, ...] = ()


# ============================================================
# BaseTreeTensorNetwork: General tree tensor network infrastructure
# ============================================================

class BaseTreeTensorNetwork(TensorNetwork):
    """
    Abstract base class for tree tensor networks.

    Provides common infrastructure for acyclic tensor networks with parent/child hierarchy:
    - Node storage and management
    - Tree topology (parent/children mappings)
    - Message-passing RDM computation on double-layer
    - Tree traversal utilities

    Key Methods
    -----------
    rdm(target_leaves, normalize=True)
        Compute reduced density matrix for target nodes via message passing.
    make_rdm1(leaf)
        1-site RDM.
    make_rdm2(leaf_a, leaf_b)
        2-site RDM.
    mutual_information_leaves(leaf_a, leaf_b, base=2.0)
        Mutual information between two leaves.

    Subclasses should:
    - Call super().__init__() with nodes, parent, children, root
    - Implement problem-specific methods (canonicalization, observables, etc.)
    - _build_tree() : construct parent/children/root from problem-specific topology
    - _build_nodes() : construct TTNNode objects
    """

    def __init__(
        self,
        nodes: Dict[Any, TTNNode],
        parent: Dict[Any, Optional[Any]],
        children: Dict[Any, Tuple[Any, ...]],
        root: Any,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nodes = dict(nodes)
        self.parent = dict(parent)
        self.children = {k: tuple(v) for k, v in children.items()}
        self.root = root

        # register tensors in base container
        for k, n in self.nodes.items():
            self.add_tensor(k, n.tensor)

        # --- attach a dictionary of Schmidt values for every edge in the tree ---
        # for an N-node tree there are (N-1) bonds; we represent each bond by an
        # unordered frozenset of the two end‑point node keys.  the value is the
        # vector of singular values living on that link.  this mirrors the simple
        # list used by MPS but generalises it to an arbitrary graph topology.
        self._init_edge_singulars()

    def _shared_bond_inds(self, a: Any, b: Any) -> List[Ix]:
        """Find index names shared between two nodes."""
        ta = self.nodes[a].tensor.inds
        tb = self.nodes[b].tensor.inds
        return [ix for ix in ta if ix in tb]

    # ------------------------------------------------------------------
    # bond / edge utilities (used by subclasses that manage Schmidt values)
    # ------------------------------------------------------------------
    def _get_adj(self, i):
        r"""return adjacent notes of site i"""
        parent = self.parent.get(i, ())
        children = self.children.get(i, ())
        if len(children) == 0: children = (-1)
        if parent is None: parent = (-1)
        parent = parent if isinstance(parent, tuple) else (parent, )
        children = children if isinstance(children, tuple) else (children, )
        return parent + children


    def _bond_edges(self):
        """Return all undirected edges (parent‑child bonds) as frozensets,
        one for every edge in the tree

        Uses :attr:`children` mapping, so parent–child pairs are only counted
        once.
        """
        edges = set()
        for u, ch_list in self.children.items():
            for v in ch_list:
                edges.add(frozenset({u, v}))
        return list(edges)

    def _init_edge_singulars(self):
        """Initialise a dictionary of Schmidt vectors keyed by edge frozensets.

        The storage is **not** assigned to ``self.Ss`` here so that subclasses can
        choose their own attribute name.  We simply return the dict and let the
        caller store it however they wish (e.g. ``self.Ss`` in the TTN class).
        """
        be = self.backend
        Smap: Dict[frozenset, Any] = {}
        for edge in self._bond_edges():
            u, v = tuple(edge)
            shared = self._shared_bond_inds(u, v)
            if not shared:
                raise ValueError(f"no shared index between nodes {u} and {v}")
            name = shared[0]
            A = self.nodes[u].tensor.data
            idx = self.nodes[u].tensor.inds.index(name)
            dim = int(A.shape[idx])
            Smap[edge] = be.ones((dim,), dtype=float)
        self.Smap = Smap
        return self.Smap


    # def init_envs_map(self):
    #     Lenv: Dict[frozenset, Any] = {}
    #     Renv: Dict[frozenset, Any] = {}
    #     for edge in self._bond_edges():
    #         print("edge = ", edge)
    #         u, v = tuple(edge)
    #         shared = self._shared_bond_inds(u, v)
    #         if not shared:
    #             raise ValueError(f"no shared index between nodes {u} and {v}")
    #         Lenv[edge] = None
    #         Renv[edge] = None
    #     self.Lenv = Lenv
    #     self.Renv = Renv


    def init_envs_map(self):
        r"""Initialize directed environment maps for holding left/right environments on edges.

        Uses **directed** edges (i, j) as keys instead of undirected frozenset({i, j}).
        This enables proper tracking of environment flow direction in the tensor network.
        """
        self.Lenv = {}  # left environments keyed by directed edge (u, v)
        self.Renv = {}  # right environments keyed by directed edge (u, v)

    def get_S(self, u, v):
        """Return Schmidt values living on the bond connecting ``u`` and ``v``.

        The order of ``u``/``v`` does not matter; a frozenset of the two keys is
        used internally.  A ``KeyError`` will be raised if the pair is not
        adjacent in the tree.
        """
        return self.Smap[frozenset({u, v})]

    def set_S(self, u, v, S):
        """Store a new vector of singular values for bond ``{u,v}``.

        Modifying the array in place is fine, but the assignment allows the
        caller to replace the entire vector (e.g. after truncation).
        """
        self.Smap[frozenset({u, v})] = S

    def _upward_order_org(self) -> List[Any]:
        r"""Return node IDs in upward traversal (leaves to root)."""
        order: List[Any] = []
        seen = set()
        def dfs(u: Any):
            if u in seen:
                return
            seen.add(u)
            for v in self.children.get(u, ()):
                dfs(v)
            order.append(u)
        dfs(self.root)
        return order

    def _upward_order(self) -> List[Any]:
        """Return node IDs in upward traversal (leaves to root).

        This is simply the reverse of the *continuous* downward contraction path
        produced by :meth:`_downward_order`.  Every pair of consecutive entries
        in the returned list are adjacent in the tree, making it useful for
        sweeping algorithms that may traverse bonds in reverse order.
        """
        # reuse the downward contraction path and reverse it
        return list(self._downward_order())[::-1]


    def get_As(self, i, j=None, direction="left"):
        r"""Return a reduced 3-index tensor for node ``i`` when contracting
        toward a neighbouring node ``j``.

        The returned array has shape ``(R, d, D_ij)`` where ``D_ij`` is the
        bond dimension on the ``i``–``j`` link, ``d`` is the product of all
        physical dimensions attached to node ``i`` and ``R`` is the product of
        the remaining virtual bond dimensions.  This mirrors the behavior of
        ``MPS.As`` entries for a chain and allows one to apply MPS‑style
        two‑site updates along an arbitrary tree edge.
        """
        # ensure i and j share a bond
        logger.debug(self, f"\nDebug: constructing effective As on edge ({i}, {j})")
        shared = self._shared_bond_inds(i, j)
        if not shared:
            raise ValueError(f"nodes {i} and {j} are not adjacent")
        bond_name = shared[0]

        node = self.nodes[i]
        A = node.tensor.data
        inds = list(node.tensor.inds)
        phys = set(node.phys_inds)

        # locate axes
        axis_j = inds.index(bond_name)
        phys_axes = [ax for ax, name in enumerate(inds) if name in phys]
        other_axes = [ax for ax, name in enumerate(inds)
                      if name not in phys and ax != axis_j]

        logger.debug(self, f"A[{i}].shape   : {A.shape}")
        logger.debug(self, f"inds      : {inds}")
        logger.debug(self, f"axis_j    : {axis_j}")
        logger.debug(self, f"phys_axes : {phys_axes}")
        logger.debug(self, f"other_axes: {other_axes}")

        # permute so that other virtuals come first, then phys legs, then the
        # bond-to-j last
        if direction == "left":
            perm = other_axes + phys_axes + [axis_j]
        else:
            perm = [axis_j] + phys_axes + other_axes
        logger.debug(self, f"perm    : {perm}")
        A_perm = np.transpose(A, perm)

        shape = A.shape
        R = int(np.prod([shape[ax] for ax in other_axes])) if other_axes else 1
        d = int(np.prod([shape[ax] for ax in phys_axes])) if phys_axes else 1
        Dj = shape[axis_j]
        logger.debug(self, f"A.shape after reduction : {A_perm.reshape(R, d, Dj).shape}")
        if direction == "left":
            return A_perm.reshape(R, d, Dj)
        else:
            return A_perm.reshape(Dj, d, R)


    def set_As(self, i, j, Ai, direction="left"):
        r"""Update the tensor at node ``i`` given a reshaped 3-index tensor.

        This reverses the reshaping done by :meth:`get_As`.  The input tensor
        ``Ai`` has shape ``(R, d, D_ij_new)`` where the bond dimension
        ``D_ij_new`` may differ from the original (e.g. after SVD truncation).

        The function:
        1. Reshapes ``Ai`` from ``(R, d, D_ij_new)`` back to the original index structure with the new bond dimension.
        2. Permutes the axes back to the original order.
        3. Updates the tensor stored in ``self.nodes[i]``.

        Parameters
        ----------
        i : key
            Node key for the tensor to update.
        j : key
            Neighboring node (defines which virtual bond is the "special" one).
        Ai : array
            Updated 3-index tensor with shape ``(R, d, D_ij_new)``.
        """
        logger.debug(self, f"\nUpdating tensors via set_As({i}, {j}) function")
        # ensure i and j share a bond
        shared = self._shared_bond_inds(i, j)
        if not shared:
            raise ValueError(f"nodes {i} and {j} are not adjacent")
        bond_name = shared[0]

        node = self.nodes[i]
        A_old = node.tensor.data
        inds = list(node.tensor.inds)
        phys = set(node.phys_inds)

        # locate axes in original tensor
        axis_j = inds.index(bond_name)
        phys_axes = [ax for ax, name in enumerate(inds) if name in phys]
        other_axes = [ax for ax, name in enumerate(inds)
                      if name not in phys and ax != axis_j]

        logger.debug(self, f"old A.shape : {A_old.shape}")
        logger.debug(self, f"reduced A shape : {Ai.shape}")
        logger.debug(self, f"inds        : {inds}")
        logger.debug(self, f"axis_j      : {axis_j}")
        logger.debug(self, f"phys_axes   : {phys_axes}")
        logger.debug(self, f"other_axes  : {other_axes}")

        # permutation used in get_As
        if direction == "left":
            perm = other_axes + phys_axes + [axis_j]
        else:
            perm = [axis_j] + phys_axes + other_axes
        logger.debug(self, f"perm    : {perm}")

        # inverse permutation: map from permuted axes back to original axes
        inv_perm = [0] * len(perm)
        for orig_idx, perm_idx in enumerate(perm):
            inv_perm[perm_idx] = orig_idx

        other_shapes = [A_old.shape[ax] for ax in other_axes]
        phys_shapes = [A_old.shape[ax] for ax in phys_axes]

        # reshape Ai from the reduced tensor layout back to the tensor layout
        # used by get_As before the inverse permutation.
        if direction == "left":
            R, d, D_ij_new = Ai.shape
            target_shape = tuple(other_shapes) + tuple(phys_shapes) + (D_ij_new,)
        else:
            D_ij_new, d, R = Ai.shape
            target_shape = (D_ij_new,) + tuple(phys_shapes) + tuple(other_shapes)
        logger.debug(self, f"New bond dimension: {D_ij_new}")
        logger.debug(self, f"target reshape    : {target_shape}")
        assert Ai.size == np.prod(target_shape)

        A_reshaped = np.reshape(Ai, target_shape)

        logger.debug(self, f"intermediate shape after reshape: {A_reshaped.shape}")

        # permute back to original index order using inverse permutation
        A_new = np.transpose(A_reshaped, inv_perm)

        logger.debug(self, f"final A_new.shape : {A_new.shape}")

        # update the tensor in the network
        self.nodes[i].tensor.data = A_new
        return A_new


    def update_bond(self, i, j, Ai, Sj, Bj):
        r"""Update bonds according to new SVD values
        """
        print("\nDebug: in update_bond of BaseTreeTensorNetwork")
        print("Ai.shape :", Ai.shape)
        print("Sj.shape :", Sj.shape)
        print("Bj.shape :", Bj.shape)

        Si = self.get_S(i, j)
        Gi = Ai # np.tensordot(np.diag(Si**(-1)), Ai, axes=(1, 0))
        newAi = np.tensordot(Gi, np.diag(Sj), axes=(2, 0))
        print("Gi.shape     = ", Gi.shape)
        print("New Sj.shape = ", Sj.shape)
        self.set_S(i, j, Sj)
        self.set_As(i, j, newAi) #, Bj)
        self.set_As(j, i, Bj, direction="right")
        # raise NotImplementedError("update bond not implemented for general ttn!")

    def _downward_order(self) -> List[Any]:
        """Return a continuous contraction path (root-to-leaves with backtracking).

        Unlike a simple topological ordering, this method records each step of a
        depth-first walk including returns to parent nodes.  For a tree with N
        nodes the path length is ``a*N-1`` and every consecutive pair of nodes
        is connected by a bond.  Example (binary tree):

            [0,1,3,7,3,8,3,1,4,9,4,1,0,2,5,2,6]
        """
        path: List[Any] = []
        seencount = 0
        total = len(self.nodes)
        def dfs(u: Any):
            nonlocal seencount
            path.append(u)
            seencount += 1
            for v in self.children.get(u, ()):
                dfs(v)
                # only backtrack if there are still unseen nodes remaining
                if seencount < total:
                    path.append(u)
                else:
                    return
        dfs(self.root)
        return path


    def _double_tensor(self, node_id: Any, target_phys: FrozenSet[Any]) -> Tensor:
        """
        Build double-layer tensor :math:`D = A \otimes A*` for RDM computation.

        - For target physical indices: left open as (p_k, p_b)
        - For non-target physical indices: contracted (trace over)
        - Bond indices: left open as (b_k, b_b)
        """
        be = self.backend
        node = self.nodes[node_id]
        A = node.tensor.data
        inds = node.tensor.inds
        phys = node.phys_inds

        # suffix convention
        ket_inds = tuple(f"{ix}__k" for ix in inds)
        bra_inds = tuple(f"{ix}__b" for ix in inds)

        # axes of physical legs in this tensor
        phys_axes = [inds.index(pix) for pix in phys] if phys else []

        if (node_id in target_phys) and phys:
            # keep physical legs open: outer product, no contraction over phys
            data = be.tensordot(A, be.conj(A), axes=0)
            out_inds = ket_inds + bra_inds
            return Tensor(data, out_inds, node.tensor.tags.union({"DOUBLE"}))

        # Contract non-target phys legs
        if phys_axes:
            data = be.tensordot(A, be.conj(A), axes=(phys_axes, phys_axes))
            # determine remaining inds (bonds only)
            rem = [ix for ix in range(len(inds)) if ix not in phys_axes]
            out_k = tuple(ket_inds[i] for i in rem)
            out_b = tuple(bra_inds[i] for i in rem)
            out_inds = out_k + out_b
            return Tensor(data, out_inds, node.tensor.tags.union({"DOUBLE"}))

        # no physical indices to contract
        data = be.tensordot(A, be.conj(A), axes=0)
        out_inds = ket_inds + bra_inds
        return Tensor(data, out_inds, node.tensor.tags.union({"DOUBLE"}))

    def rdm(self, target_leaves: Sequence[Any], normalize: bool = True) -> Any:
        """
        Reduced density matrix for a small set of target nodes (typically leaves with phys inds).
        Exact via message passing on the double-layer (tree => no approximation).

        Output:
          - for one target with phys dim d: (d, d)
          - for two targets: (d1, d2, d1, d2) in order of target_leaves
          - for k targets: tensor with ket phys legs then bra phys legs
        """
        be = self.backend
        targets = list(target_leaves)
        target_set: FrozenSet[Any] = frozenset(targets)

        # Build local double tensors
        D: Dict[Any, Tensor] = {u: self._double_tensor(u, target_set) for u in self.nodes.keys()}

        # Upward messages: msg[u] is tensor leaving the parent bond legs open (if u!=root)
        msg: Dict[Any, Tensor] = {}
        # FIXME: new upward_order needs to error
        # print("self._upward_order()     in rdm is", self._upward_order())
        # print("self._upward_order_org() in rdm is", self._upward_order_org())
        for u in self._upward_order_org():
            cur = D[u]
            # contract in child messages
            for c in self.children.get(u, ()):
                cur = contract_two(cur, msg[c])
            # root: cur contains only open phys legs (and maybe none)
            msg[u] = cur

        # At root, msg[root] is the full contraction with open target phys legs
        out = msg[self.root]

        # Identify desired output ordering of physical indices:
        # for each target node, take its phys_inds (often length 1), and use suffixed __k, __b
        ket_phys: List[Ix] = []
        bra_phys: List[Ix] = []
        for t in targets:
            for pix in self.nodes[t].phys_inds:
                ket_phys.append(f"{pix}__k")
                bra_phys.append(f"{pix}__b")

        desired = tuple(ket_phys + bra_phys)
        out = out.transpose_to(desired).data

        # reshape to matrix for normalization + optionally return structured tensor
        if len(ket_phys) == 1:
            rho = out  # (d, d)
            if normalize:
                rho = rho / be.trace(rho)
            return rho

        # For k>=2, out is (d1,d2,..., d1,d2,...)
        # Normalize by trace after reshaping to (D,D)
        dims_ket = out.shape[: len(ket_phys)]
        Ddim = 1
        for d in dims_ket:
            Ddim *= int(d)
        rho_mat = be.reshape(out, (Ddim, Ddim))
        if normalize:
            rho_mat = rho_mat / be.trace(rho_mat)
        # return as rank-2k tensor again (nice for operator contractions)
        return be.reshape(rho_mat, out.shape)

    def make_rdm1(self, leaf: Any, normalize: bool = True) -> Any:
        """1-site reduced density matrix for a leaf node."""
        return self.rdm([leaf], normalize=normalize)

    def make_rdm2(self, leaf_a: Any, leaf_b: Any, normalize: bool = True) -> Any:
        """2-site reduced density matrix for two leaf nodes."""
        return self.rdm([leaf_a, leaf_b], normalize=normalize)

    def von_neumann_entropy_1body(
        self,
        leaf: Any,
        base: float = 2.0,
        eps: float = 1e-12,
        normalize: bool = True,
    ) -> Any:
        """Von Neumann entropy of a one-body reduced density matrix.

        Parameters
        ----------
        leaf : key
            Target node key.
        base : float
            Logarithm base in entropy definition.
        eps : float
            Lower clip for eigenvalues for numerical stability.
        normalize : bool
            Whether to normalize the one-body RDM before entropy evaluation.
        """
        rho = self.make_rdm1(leaf, normalize=normalize)
        return von_neumann_entropy(rho, base=base, eps=eps)

    def von_neumann_entropy_2body(
        self,
        leaf_a: Any,
        leaf_b: Any,
        base: float = 2.0,
        eps: float = 1e-12,
        normalize: bool = True,
    ) -> Any:
        """Von Neumann entropy of a two-body reduced density matrix."""
        rho = self.make_rdm2(leaf_a, leaf_b, normalize=normalize)
        return von_neumann_entropy(rho, base=base, eps=eps)

    # concise aliases
    entropy_1body = von_neumann_entropy_1body
    entropy_2body = von_neumann_entropy_2body

    def mutual_information_leaves(self, leaf_a: Any, leaf_b: Any, base: float = 2.0) -> Any:
        """Mutual information between two leaf nodes."""
        rho_a = self.make_rdm1(leaf_a, normalize=True)
        rho_b = self.make_rdm1(leaf_b, normalize=True)
        rho_ab = self.make_rdm2(leaf_a, leaf_b, normalize=True)
        return mutual_information(rho_a, rho_b, rho_ab, base=base)

    def _to_networkx_graph(self):
        """Return a :mod:`networkx` graph representing the TTN.

        Nodes are the same as ``self.nodes``; edges carry a ``weight`` equal
        to the bond dimension (size of the shared index).  The graph is
        undirected.  This method works for any tree tensor network.

        Nodes are the same as ``self.nodes``; edges carry a ``weight`` equal
        to the bond dimension (size of the shared index).  The graph is
        undirected.  This method works for any tree tensor network.
        """
        try:
            import networkx as nx
        except ImportError:  # pragma: no cover
            raise ImportError("networkx must be installed to convert to graph")

        G = nx.Graph()
        for n in self.nodes:
            G.add_node(n)
        # add edges from parent-child relationships
        for u, childs in self.children.items():
            for v in childs:
                shared = self._shared_bond_inds(u, v)
                if shared:
                    idx = shared[0]
                    size = self.nodes[u].tensor.shape[
                        self.nodes[u].tensor.inds.index(idx)
                    ]
                else:
                    size = 1
                G.add_edge(u, v, weight=size)
        return G

    def visualize(self, figsize: Tuple[int, int] = (10, 8), show: bool = True):
        """Visualize the tree tensor network structure using networkx.

        Creates a plot of the network graph with node labels and edge weights
        representing bond dimensions.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size for the plot (width, height).
        show : bool
            If True, display the plot immediately. If False, return the
            matplotlib figure object.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The matplotlib figure object if show=False, otherwise None.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
        except ImportError:  # pragma: no cover
            raise ImportError(
                "networkx and matplotlib must be installed for visualization"
            )

        G = self._to_networkx_graph()
        fig, ax = plt.subplots(figsize=figsize)

        # try to create hierarchical layout top-down (root at top)
        pos = None
        try:
            # prefer graphviz dot layout if available
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            # fallback to custom hierarchy_pos or spring
            def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
                """If there is a cycle that is reachable from root, then this will see infinite recursion."""
                if pos is None:
                    pos = {root: (xcenter, vert_loc)}
                else:
                    pos[root] = (xcenter, vert_loc)
                neighbors = list(G.neighbors(root))
                if parent is not None and parent in neighbors:
                    neighbors.remove(parent)
                if len(neighbors) != 0:
                    dx = width / len(neighbors)
                    nextx = xcenter - width / 2 - dx / 2
                    for neighbor in neighbors:
                        nextx += dx
                        pos = hierarchy_pos(
                            G,
                            neighbor,
                            width=dx,
                            vert_gap=vert_gap,
                            vert_loc=vert_loc - vert_gap,
                            xcenter=nextx,
                            pos=pos,
                            parent=root,
                        )
                return pos

            if hasattr(self, "root") and self.root in G:
                pos = hierarchy_pos(G, self.root)
            else:
                pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

        # draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color="lightblue", node_size=900, ax=ax
        )

        # draw edges with weights
        nx.draw_networkx_edges(G, pos, width=2.5, ax=ax)

        # draw labels
        nx.draw_networkx_labels(G, pos, font_size=14, ax=ax)

        # draw edge weights
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, ax=ax)

        # ax.set_title("Tree Tensor Network Structure")
        ax.axis("off")
        fig.tight_layout()

        if show:
            plt.show()
            return None
        else:
            return fig



# ============================================================
# MPS: Matrix Product State as a tree tensor network
# ============================================================

class MPS(BaseTreeTensorNetwork):
    r"""
    Matrix Product State as a special case of tree tensor network.

    MPS tensors A[i] with shape (Dl, d, Dr), where d is the dize of local physical space
    Dl/Dr are the virtual bond dimensions. We choose this order of indices because it
    make the canonicalization/SVD steps natural: e.g. reshape :math:`(D_L\cdot d, D_R)`
    or :math:`(D_L, d\cdot D_R)` when pushing orthogonality center to the left/right.

    Efficient:
    - 1-site RDM: :math:`O(\chi^3 d)`
    - 2-site RDM (any :math:`i<j`): :math:`O(|i-j| \chi^3 d)`
    - entanglement entropy across cut: :math:`O(L \chi^3 d)` via local canonicalization around cut

    Parameters
    ----------
    arrays : Sequence[Any]
        List of MPS tensors, each shape (Dl, d, Dr)
    tags_prefix : str
        Prefix for tensor tags
    boundary : str
        "finite" or "periodic"

    Attributes
    ----------
    As : List[Any]
        List of MPS tensors for backward compatibility
    Ss : List[Any]
        Schmidt values at each bond
    L : int
        Number of sites
    d : int
        Physical dimension

    Inherited from BaseTreeTensorNetwork
    -----------------------------------
    nodes, parent, children, root
        Tree topology (linear chain for MPS)
    rdm, make_rdm1, make_rdm2
        Message-passing RDM computation
    """

    def __init__(
        self,
        arrays: Sequence[Any],
        tags_prefix: str = "I",
        **kwargs
    ):
        self.nsites = self.L = len(arrays)
        self.tags_prefix = tags_prefix
        be = backend
        # self.As: List[Any] = [self.backend.asarray(a) for a in arrays] # (vL, p, vR)
        self.As: List[Any] = [be.asarray(a) for a in arrays]
        self.bc = kwargs.get("boundary", "finite")
        self.num_bonds = self.L - 1 if self.bc == 'finite' else self.L

        # Schmidt values at bonds
        S = np.ones([1], dtype=float)
        self.Ss: List[Any] = [S] * self.L #TODO: remove this list, replaced with Smap

        # Build linear chain tree topology: 0 <- 1 <- 2 <- ... <- L-1
        # with named inds (bond names are internal bookkeeping)
        # bonds: b(i-1) --[site i]-- b(i)
        nodes = {}
        parent = {}
        children = {}

        for i, A in enumerate(self.As):
            li = f"b{i-1}" if i > 0 else f"l{i}"
            ri = f"b{i}" if i < self.L - 1 else f"r{i}"
            pi = f"k{i}"
            # print(f"node {i} labels =", li, pi, ri)
            nodes[i] = TTNNode(
                Tensor(A, (li, pi, ri), frozenset({f"{tags_prefix}{i}"})),
                phys_inds=(pi,)
            )
            parent[i] = i - 1 if i > 0 else None
            children[i] = (i + 1,) if i < self.L - 1 else ()
            # self.add_tensor(i, Tensor(A, (li, pi, ri), frozenset({f"{tags_prefix}{i}"})))

        # Initialize tree from node 0 (left-most)
        super().__init__(nodes, parent, children, root=0, **kwargs)

    @staticmethod
    def rand(L: int, d: int, chi: int, seed: Optional[int] = None,
             **kwargs) -> "MPS":
        """Generate random MPS."""
        if backend.name == "numpy":
            rng = np.random.default_rng(seed)
            As = []
            for i in range(L):
                Dl = 1 if i == 0 else chi
                Dr = 1 if i == L - 1 else chi
                A = rng.normal(size=(Dl, d, Dr)) + 1j * rng.normal(size=(Dl, d, Dr))
                A /= np.linalg.norm(A)
                As.append(A)
            return MPS(As, **kwargs)
        # for torch/jax: user can pass arrays already on device; this is a simple fallback
        raise NotImplementedError("For non-numpy random init, pass arrays explicitly.")

    def _downward_order(self) -> List[Any]:
        path = super()._downward_order()
        # if self.bc == 'infinite':
        #    path.append(self.root)
        return path

    def get_As(self, i, j=None):
        r"""return the tensor at node i"""
        return self.As[i]

    def set_As(self, i, Ai):
        r"""return the tensor at node i"""
        self.As[i] = Ai

    def get_S(self, i, j=None):
        """Return Schmidt values living on the bond connecting ``u`` and ``v``.

        The order of ``u``/``v`` does not matter; a frozenset of the two keys is
        used internally.  A ``KeyError`` will be raised if the pair is not
        adjacent in the tree.
        """
        # FIXME: using old Ss list first, change to Smap later
        return self.Ss[i]
        # return self.Smap[frozenset({u, v})]

    def set_S(self, i, j, Sj):
        """Store a new vector of singular values for bond ``{i,j}``.

        Modifying the array in place is fine, but the assignment allows the
        caller to replace the entire vector (e.g. after truncation).
        """
        # FIXME: using old Ss list first, change to Smap later
        self.Ss[j] = Sj
        # self.Smap[frozenset({u, v})] = S

    def update_bond(self, i, j, Ai, Sj, Bj):
        r"""Update bonds according to new SVD values
        """
        Si = self.get_S(i, j)
        Gi = np.tensordot(np.diag(Si**(-1)), Ai, axes=(1, 0))  # vL [vL*], [vL] i vC
        newAi = np.tensordot(Gi, np.diag(Sj), axes=(2, 0))
        self.set_S(i, j, Sj)
        self.set_As(i, newAi)
        self.set_As(j, Bj)

    # ---------- canonicalization primitives ----------
    def make_mixed_canonical(self, center: int) -> None:
        """
        Bring into mixed canonical form with orthogonality center at `center`.
        O(L chi^3 d). Uses QR sweeps.
        """
        be = self.be
        # Left-canonize [0..center-1]
        for i in range(0, center):
            A = self.As[i]
            Dl, d, Dr = A.shape
            M = be.reshape(A, (Dl * d, Dr))
            Q, R = be.qr(M)
            chi = Q.shape[1]
            self.As[i] = be.reshape(Q, (Dl, d, chi))
            # absorb R into next
            self.As[i + 1] = be.einsum("ab,bpc->apc", R, self.As[i + 1])

        # Right-canonize [L-1..center+1]
        for i in range(self.L - 1, center, -1):
            A = self.As[i]
            Dl, d, Dr = A.shape
            M = be.reshape(A, (Dl, d * Dr))
            Mt = be.transpose(M, (1, 0))  # (d*Dr, Dl)
            Q, R = be.qr(Mt)              # Mt = Q R
            Rt = be.transpose(R, (1, 0))
            Qt = be.transpose(Q, (1, 0))
            # Qt has shape (k, d*Dr) => reshape to (k,d,Dr)
            k = Qt.shape[0]
            self.As[i] = be.reshape(Qt, (k, d, Dr))
            # absorb Rt into previous
            self.As[i - 1] = be.einsum("apb,bc->apc", self.As[i - 1], Rt)

    # ---------------environments for <MPS|H|MPS> ----------------
    def update_left(self, i, Li, Wi):
        r"""Compute left environment for <MPS|H|MPs> tensor contraction
        """
        be = self.backend
        Ai = self.nodes[i].tensor.data
        Ac = be.conj(Ai)      # (Dl, d, Dr)
        return update_left(Wi, Ai, Li, Ac)

    def update_right(self, i, Ri, Wi):
        r"""Compute right environment for <MPS|H|MPs> tensor contraction
        """
        be = self.backend
        Bi = self.nodes[i].tensor.data
        Bc = be.conj(Bi)
        return update_right(Wi, Bi, Ri, Bc)


    # ---------- environments for RDMs ----------

    def left_envs(self) -> List[Any]:
        r"""Compute left environment for RDM

        .. math::
            L^i_{bd} = \sum_{p,ac} L^{i-1}_{ac} A^{i}_{p,ab} A^\dagger_{p, cd}

        Ls[i] is environment up to (but excluding) site i, living on the bond left of site i:
        shape (Dl_i, Dl_i) in the double-layer.
        Ls[0] is (1,1).
        Assumes A[i] has shape (d, Dl, Dr).
        """
        be = self.backend
        Ls: List[Any] = [be.eye(1, dtype=self.As[0].dtype)]
        for i in range(self.L):
            A = self.As[i]       # (Dl, d, Dr)
            Ac = be.conj(A)      # (Dl, d, Dr)
            # L_next[Dr,Dr] = sum_{Dl,Dl,d} L[Dl,Dl] A[Dl,d,Dr] Ac[Dl,d,Dr]
            L_next = be.einsum("ac,apb,cpd->bd", Ls[-1], A, Ac)
            Ls.append(L_next)
        return Ls

    def right_envs(self) -> List[Any]:
        r"""Compute right environment for RDM

        .. math::
            R^i_{ac} = \sum_{p,bd} R^{i+1}_{bd} A^{i}_{p,ab} A^\dagger_{p, cd}

        R_env[i] is environment from (but excluding) site i to the end, on bond right of site i-1:
        We return list Rs of length L+1 with:

          - R_env[L] = (1,1)
          - R_env[i] lives on bond right of site i-1 (equivalently bond left of i in reverse view).
        """
        be = self.backend
        Rs: List[Any] = [None] * (self.L + 1)
        Rs[self.L] = be.eye(1, dtype=self.As[0].dtype)
        for i in range(self.L - 1, -1, -1):
            A = self.As[i]
            Ac = be.conj(A)
            # R_i[a, c] = sum_{b,d,p} R_{i+1}[b,d] A[a,p,b] Ac[c,p,d]
            Rs[i] = be.einsum("bd,apb,cpd->ac", Rs[i + 1], A, Ac)
        return Rs

    def get_chi(self):
        r"""get bond dimensions"""
        chis = [len(v) for v in self.Ss]
        return chis
        return [self.As[i].shape[-1] for i in range(self.L)] #TODO: may replace it with num_bonds
    get_bond_dimensions = get_chi

    def copy(self):
        return MPS([A.copy() for A in self.As], boundary=self.bc)

    def get_psi_1site(self, i):
        r"""Calculate effective single-site wave function on sites i in mixed canonical form.
        """
        # Si = self.get_S(i, j)

        return np.tensordot(np.diag(self.Ss[i]), self.As[i], [1, 0])  # (vl, p, vR)

    def get_psi_2site(self, i, j):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``.
        """
        logger.debug(self, f"Ai.shape: {self.As[i].shape}")
        logger.debug(self, f"Aj.shape: {self.As[j].shape}")
        return np.tensordot(self.get_psi_1site(i), self.As[j], [2, 0]) #(vl, p, q, vR)


    # ---------- reduced density matrices ----------
    def make_rdm1(self, i: int, normalize: bool = True) -> Any:
        be = self.backend
        Ls = self.left_envs()
        Rs = self.right_envs()
        A = self.As[i]
        Ac = be.conj(A)
        # rho[p,q] = sum_{a,c,b,d} L[a,c] A[a,p,b] Ac[c,q,d] R[b,d]
        rho = be.einsum("ac,apb,cqd,bd->pq", Ls[i], A, Ac, Rs[i + 1])
        if normalize:
            tr = be.trace(rho)
            rho = rho / tr
        return rho

    def make_rdm2(self, i: int, j: int, normalize: bool = True) -> Any:
        """
        2-site reduced density matrix for sites :math:`i<j`.
        Returns a 4D tensor rho[pi,pj,pi',pj'].
        Computed efficiently by propagating a “carry” object along the chain: :math:`O(|i-j| \chi^3 d)`.
        """
        if i == j:
            raise ValueError("Use rdm1 for single site.")
        if i > j:
            i, j = j, i

        be = self.backend
        Ls = self.left_envs()
        Rs = self.right_envs()

        # Start carry T for site i: T[pi, pi', r, r']
        Ai = self.As[i]
        Aic = be.conj(Ai)
        T = be.einsum("ac,apb,cqd->pqbd", Ls[i], Ai, Aic)  # (p, p', r, r')

        # Propagate through intermediate sites (i+1 ... j-1):
        for k in range(i + 1, j):
            Ak = self.As[k]
            Akc = be.conj(Ak)
            # If k < j, we must contract phys of k (unless k==j which we stop before)
            # T'[pi,pi', r, r'] = sum_{l,l',p} T[pi,pi', l,l'] Ak[l,p,r] Akc[l',p,r']
            T = be.einsum("pqac,apb,cpd->pqbd", T, Ak, Akc)

        # Now T lives on bond left of site j: T[pi,pi', l, l']
        Aj = self.As[j]
        Ajc = be.conj(Aj)
        # Contract with site j and right env:
        # rho[pi,pj,pi',pj'] = sum_{l,l',r,r'} T[pi,pi',l,l'] Aj[l,pj,r] Ajc[l',pj',r'] R[r,r']
        rho = be.einsum("pqac,arb,csd,bd->prqs", T, Aj, Ajc, Rs[j + 1]) # rho[p_i, p_j, p_i', p_j']

        if normalize:
            # trace over pi=pi' and pj=pj'
            d1, d2, d3, d4 = rho.shape
            rho_mat = be.reshape(rho, (d1 * d2, d3 * d4))
            tr = be.trace(rho_mat)
            rho = rho / tr
        return rho


    # ---------- calculate observables ----------

    def expval_1body(self, op: Any, i: int) -> Any:
        r"""Calculate expectation values of a local operator at each site."""
        be = self.backend
        op = be.asarray(op)
        rho = self.make_rdm1(i, normalize=True)
        return be.einsum("pq,qp->", op, rho)
    local_expectation = expval_1body


    def expval_2body(self, op: Any, i: int, j: int) -> Any:
        """
        op can be:
          - shape (d,d,d,d) with indices (pi,pj,pi',pj')
          - or shape (d^2,d^2) acting on vec basis (pi,pj)
        """
        be = self.backend
        op = be.asarray(op)
        rho = self.make_rdm2(i, j, normalize=True)

        if getattr(op, "ndim", None) == 2:
            rho_mat = _as_density_matrix(rho)
            return be.einsum("ab,ba->", op, rho_mat)

        if getattr(op, "ndim", None) == 4:
            return be.einsum("pqrs,rspq->", op, rho)

        raise ValueError("Unsupported op shape for 2-body expectation.")

    def expval_bonds(self, op: List[Any]):
        r"""Calculate the expectation value of local operators along the bond"""

        logger.info(self, "Calculate the expectation value of local operators along the bond")

        assert len(op) == self.num_bonds
        be = self.backend

        values = np.zeros(self.num_bonds)
        for i in range(self.num_bonds):
            j = (i + 1) % self.L
            rhoij = self.make_rdm2(i, j, normalize=False)
            logger.debug(self, f"op[i].shape = {op[i].shape}")
            logger.debug(self, f"rhoij.shape = {rhoij.shape}")

            # oprho= np.tensordot(op[i], rhoij, axes=([2, 3], [1, 2]))
            # tmp = np.tensordot(rhoij.conj(), oprho, [[0, 1, 2], [1, 0, 2]])
            #print("temp.shape = ", tmp.shape)
            #values[i] += np.sum(np.tensordot(rhoij.conj(), oprho, [[0, 1, 2], [1, 0, 2]]))

            tmp = be.einsum("pqrs,pqrs->", op[i], rhoij)
            values[i] += tmp
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(values)

    def expval_bonds_from_psi2(self, op: List[Any]):
        r"""Compute the expectation value of local operators along the bond
        Using effective two-site wavefunction
        """
        # FIXME: It should give the same results as expvan_bonds (using rdm2)
        values = []
        for i in range(self.num_bonds):
            j = (i + 1) % self.L  # FIXME: make it compatible with TTN structure
            psi2 = self.get_psi_2site(i, j)  # vL i j vR
            op_psi = np.tensordot(op[i], psi2, axes=([2, 3], [1, 2]))
            # i j [i*] [j*], vL [i] [j] vR
            values.append(np.tensordot(psi2.conj(), op_psi, [[0, 1, 2, 3], [2, 0, 1, 3]]))
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
        return np.real_if_close(values)


    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy at any bond"""
        bonds = range(1, self.L) if self.bc == 'finite' else range(0, self.L)
        result = []
        for i in bonds:
            S = self.Ss[i]
            S = S[S > self.threshold]
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-13
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)

    # ---------- entanglement ----------
    def entanglement_entropy_cut(self, cut: int, base: float = 2.0, eps: float = 1e-12) -> Any:
        """
        Entanglement across bipartition: [0..cut-1] | [cut..L-1], with 0<cut<L.

        Efficient approach:
          - left-canonize sites < cut (QR sweep) absorbing into site cut
          - right-canonize sites > cut (reverse QR sweep) absorbing into site cut
          - take SVD of site-cut tensor reshaped as (Dl, d*Dr)
          - Schmidt probs = S^2 / sum(S^2)
          - entropy = -sum p log p
        """
        if not (0 < cut < self.L):
            raise ValueError("cut must satisfy 0 < cut < L.")

        be = self.backend
        As = [a for a in self.As]  # shallow copy

        # --- Left-canonize up to cut-1 (absorb into site cut) ---
        for i in range(0, cut):
            A = As[i]
            Dl, d, Dr = A.shape
            M = be.reshape(A, (Dl * d, Dr))
            Q, R = be.qr(M)
            chi = Q.shape[1]
            As[i] = be.reshape(Q, (Dl, d, chi))
            if i + 1 < self.L:
                As[i + 1] = be.einsum("ab,bpc->apc", R, As[i + 1])

        # --- Right-canonize from end down to cut+1 (absorb into site cut) ---
        for i in range(self.L - 1, cut, -1):
            A = As[i]
            Dl, d, Dr = A.shape
            M = be.reshape(A, (Dl, d * Dr))  # want an RQ; do QR on transpose
            Mt = be.transpose(M, (1, 0))     # (d*Dr, Dl)
            Q, R = be.qr(Mt)                 # Mt = Q R
            # Convert back: M = R^T Q^T
            Rt = be.transpose(R, (1, 0))
            Qt = be.transpose(Q, (1, 0))
            As[i] = be.reshape(Qt, (Q.shape[1], d, Dr))  # (Dl_new, d, Dr)
            As[i - 1] = be.einsum("apb,bc->apc", As[i - 1], Rt)

        # --- Schmidt values from center site (site 'cut') ---
        A = As[cut]
        Dl, d, Dr = A.shape
        M = be.reshape(A, (Dl, d * Dr))
        U, S, Vh = be.svd(M, full_matrices=False)
        p = S * S
        p = p / be.sum(p)
        p = be.clip(p, eps, 1.0)
        if base == 2.0:
            return -be.sum(p * (be.log(p) / be.log(be.asarray(2.0))))
        return -be.sum(p * (be.log(p) / be.log(be.asarray(base))))

    def mutual_info_sites(self, i: int, j: int, base: float = 2.0) -> Any:
        rho_i = self.make_rdm1(i, normalize=True)
        rho_j = self.make_rdm1(j, normalize=True)
        rho_ij = self.make_rdm2(i, j, normalize=True)
        # print("rho_ij.shape", rho_ij.shape)

        return mutual_information(rho_i, rho_j, rho_ij, base=base)

    def corr_length(self):
        """Calculate the correlation length."""

        raise NotImplementedError("corr_length is not ready yet")

    def norm(self):
        r"""compute <MPS|MPS>"""

        L = np.ones((1, 1), dtype=complex)
        for A in self.As:
            L = np.einsum('ij,ipk,jpl->kl', L, A, A.conj(), optimize=True)
        assert L.shape == (1, 1)
        assert abs(L[0, 0].imag) < 1.e-10
        overlap = float(L[0, 0].real)
        return float(max(overlap, 0.0))


    def normalize(self) -> None:
        """Normalize in-place using tensor contraction (no dense state formed)."""
        nrm = self.norm()
        if nrm == 0.0:
            raise ValueError("cannot normalize zero state")
        self.As[0] = self.As[0] / np.sqrt(nrm)

    def to_dense(self) -> Array:
        """Debug helper: contract MPS into a dense state vector.

        Note: Each tensor has shape (Dl, d, Dr).
        """
        psi = self.As[0]  # (1,d,D1)
        for A in self.As[1:]:
            # Contract neighboring bond
            psi = np.tensordot(psi, A, axes=([-1], [0]))
        # Remove left/right boundary dimensions (=1)
        psi = np.squeeze(psi)
        # Flatten physical indices into a vector
        return psi.reshape(-1)


# Example usage
# ============================================================

if __name__ == "__main__":
    print(f"{'*' * 50}\n  MPS example\n{'*'*50}")
    # --- MPS example (numpy) ---
    mps = MPS.rand(L=10, d=2, chi=2, seed=0)
    rho3 = mps.make_rdm1(3)
    rho37 = mps.make_rdm2(3, 7)
    S_cut5 = mps.entanglement_entropy_cut(5)
    print("MPS: Tr(rho3) =", float(rho3.trace().real))
    print("MPS: S(cut=5) =", float(S_cut5))

    I_37 = mps.mutual_info_sites(3, 7)
    print("MPS: I(3:7)   =", float(I_37))


    print(f"{'*' * 50}\n  TTN example: tiny 2-leaf tree\n{'*'*50}")
    # --- TTN example: tiny 2-leaf tree ---
    import numpy as np
    from ttqd.linalg import backend as be
    rng = be.random.default_rng(1)
    # leaf tensors: (bond, phys)
    A0 = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    A1 = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    # root: (bond0, bond1)
    R = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))

    nodes = {
        "leaf0": TTNNode(Tensor(be.asarray(A0), ("b0", "k0"), frozenset({"LEAF"})), phys_inds=("k0",)),
        "leaf1": TTNNode(Tensor(be.asarray(A1), ("b1", "k1"), frozenset({"LEAF"})), phys_inds=("k1",)),
        "root":  TTNNode(Tensor(be.asarray(R),  ("b0", "b1"), frozenset({"ROOT"})), phys_inds=()),
    }
    parent = {"root": None, "leaf0": "root", "leaf1": "root"}
    children = {"root": ("leaf0", "leaf1"), "leaf0": (), "leaf1": ()}
    ttn = TreeTensorNetwork(nodes, parent, children, root="root")

    rho0 = ttn.make_rdm1("leaf0")
    rho01 = ttn.rdm2("leaf0", "leaf1")
    I01 = ttn.mutual_information_leaves("leaf0", "leaf1")
    print("TTN: Tr(rho0) =", float(np.trace(rho0).real))
    print("TTN: I(0:1)   =", float(I01))
