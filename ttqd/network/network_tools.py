

r"""
tools for manipualting the network
==================================


 1) graph optimization of network

The cost function can be defined as:

.. math::
    C(I) = \sum_{i<j} f(d_{ij})I_{ij}

where :math:`f(\cdot)` is a monotonously increasing function.
:math:`d_{ij}` is the distance between two sites on a network.
:math:`I_{ij}` is the mutual information between two sites.

This module provides utilities to compute the pairwise mutual information
matrix for a tree tensor network and to produce an ordering of the leaves
that tends to group strongly correlated (high mutual information) sites
close together.  The ordering can then be used as the input for constructing
or re‑wiring a TTN, e.g. when building a binary tree in a density‑matrix
renormalization‑group (DMRG) or TDVP simulation.  A simple greedy heuristic
as well as a hierarchical‑clustering based approach are implemented.
"""

from __future__ import annotations

from typing import List, Tuple, Any, Sequence
from ttqd.network import BaseTreeTensorNetwork

import numpy as np

# additional imports guarded by optional dependencies
try:
    from scipy.cluster import hierarchy
except ImportError:  # scipy is a soft dependency for clustering
    hierarchy = None


def mutual_information_matrix(
    ttn: "BaseTreeTensorNetwork",
    leaves: Sequence[Any] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    r"""Compute the pairwise mutual information matrix for a TTN.

    Parameters
    ----------
    ttn : BaseTreeTensorNetwork
        A tensor network on which the mutual information is defined.  Only
        the leaf nodes (i.e. those whose ``phys_inds`` are non-empty) are
        considered.
    leaves : sequence, optional
        List of leaf keys in the order for which the matrix will be built.  If
        ``None`` the TTN's internal ordering of ``ttn.nodes.keys()`` is used.
    normalize : bool
        If ``True`` each reduced density matrix used in the calculation is
        trace-normalized before computing the entropy.

    Returns
    -------
    mat : ndarray, shape (n, n)
        Symmetric matrix with ``mat[i, j] = I(leaf_i, leaf_j)``.  The
        diagonal is zero.
    """
    # identify leaves
    if leaves is None:
        leaves = [k for k, n in ttn.nodes.items() if n.phys_inds]
    n = len(leaves)
    mat = np.zeros((n, n), dtype=float)

    print("\nOriginal nodes = ", leaves)

    be = ttn.backend
    for ii in range(n):
        for jj in range(ii + 1, n):
            Iij = ttn.mutual_information_leaves(
                leaves[ii], leaves[jj], base=2.0
            )
            # mutual_information returns a scalar; convert to float
            mat[ii, jj] = float(be.real(Iij))
            mat[jj, ii] = mat[ii, jj]
    return mat


def leaf_distance_matrix(
    ttn: "BaseTreeTensorNetwork", leaves: Sequence[Any] | None = None
) -> np.ndarray:
    r"""Pairwise graph distance between leaf nodes in a tree.

    The distance is measured in number of edges along the unique path
    connecting two leaves.  Only nodes with non-empty ``phys_inds`` are
    considered (just like :func:`mutual_information_matrix`).

    Parameters
    ----------
    ttn : BaseTreeTensorNetwork
        Tree network providing ``parent`` and ``children`` mappings.
    leaves : sequence, optional
        List of leaf keys in the order for which the matrix will be built.
        If ``None`` the TTN's internal ordering of ``ttn.nodes.keys()`` is
        used.

    Returns
    -------
    dist : ndarray, shape (n, n)
        Distance matrix where ``dist[i, j]`` is the number of edges between
        ``leaf_i`` and ``leaf_j``.
    """
    if leaves is None:
        leaves = [k for k, n in ttn.nodes.items() if n.phys_inds]
    n = len(leaves)
    dist = np.zeros((n, n), dtype=float)

    # build undirected adjacency for the tree
    adj = {}
    for u, childs in ttn.children.items():
        for v in childs:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

    from collections import deque

    for ii, u in enumerate(leaves):
        # BFS from u
        dq = deque([(u, 0)])
        seen = {u}
        dmap = {u: 0}
        while dq:
            node, d = dq.popleft()
            for nb in adj.get(node, ()):  # type: ignore
                if nb not in seen:
                    seen.add(nb)
                    dmap[nb] = d + 1
                    dq.append((nb, d + 1))
        for jj, v in enumerate(leaves):
            dist[ii, jj] = dmap.get(v, np.inf)
    return dist


def mutual_information_cost(mi: np.ndarray, dist: np.ndarray) -> float:
    r"""Compute cost :math:`\sum_{ij} |I_{ij}| d_{ij}`.

    Parameters
    ----------
    mi : ndarray
        Mutual information matrix (symmetric, zero diagonal).
    dist : ndarray
        Distance matrix of same shape as ``mi``.

    Returns
    -------
    cost : float
        Scalar cost.  Entries where ``dist`` is infinite yield an infinite
        cost (i.e. disconnected leaves).
    """
    if mi.shape != dist.shape:
        raise ValueError("mi and dist must have the same shape")
    return float(np.sum(np.abs(mi) * dist))


def greedy_leaf_ordering(mi: np.ndarray) -> List[int]:
    r"""Simple greedy ordering based on mutual information.

    Start from the site with the largest total mutual information and
    incrementally append the leaf that has the largest *average* mutual
    information with the already ordered set.

    Parameters
    ----------
    mi : ndarray, shape (n, n)
        Pairwise mutual information matrix.

    Returns
    -------
    order : list of int
        Permutation of ``range(n)`` giving the leaf order.
    """
    n = mi.shape[0]
    if n == 0:
        return []
    remaining = set(range(n))
    # seed with the site of largest total correlation
    totals = mi.sum(axis=1)
    current = int(np.argmax(totals))
    order = [current]
    remaining.remove(current)

    while remaining:
        # compute average MI between each candidate and the ordered set
        best = None
        best_val = -np.inf
        for cand in remaining:
            val = mi[cand, order].mean()  # average MI to existing chain
            if val > best_val:
                best_val = val
                best = cand
        order.append(best)  # type: ignore
        remaining.remove(best)
    return order


def hierarchical_leaf_ordering(mi: np.ndarray) -> List[int]:
    r"""Produce a leaf ordering via hierarchical clustering.

    The mutual information is converted to a distance by ``d_ij = 1 -
    I_ij/max(I)`` (so that more strongly correlated sites are closer).  If
    ``scipy`` is not available the greedy heuristic is used as a fallback.

    Parameters
    ----------
    mi : ndarray, shape (n, n)
        Pairwise mutual information matrix.

    Returns
    -------
    order : list of int
        Permutation of ``range(n)`` corresponding to the leaves as they appear
        in the leaves list produced by the clustering dendrogram.
    """
    n = mi.shape[0]
    if n <= 1:
        return list(range(n))
    if hierarchy is None:
        # scipy not installed; fall back
        return greedy_leaf_ordering(mi)

    # convert MI to a condensed distance matrix for linkage
    # we use d = 1 - I/I_max, which is positive and 0 for strongest links
    maxval = mi.max()
    if maxval <= 0:
        # no correlations, return natural order
        return list(range(n))
    dist = 1.0 - mi / maxval
    # condensed form required by scipy (i<j entries)
    triu = dist[np.triu_indices(n, k=1)]
    Z = hierarchy.linkage(triu, method="average")
    leaves = hierarchy.leaves_list(Z)
    return leaves.tolist()


def optimize_leaf_ordering(
    ttn: "BaseTreeTensorNetwork",
    method: str = "hierarchical",
    **kwargs,
) -> Tuple[List[Any], np.ndarray]:
    r"""Calculate a new ordering for the leaves based on mutual information.

    Parameters
    ----------
    ttn : BaseTreeTensorNetwork
        Tensor network with ``mutual_information_leaves`` method.
    method : {'hierarchical', 'greedy'}
        Which ordering algorithm to use.
    **kwargs :
        Passed to :func:`mutual_information_matrix` (e.g. ``leaves`` or
        ``normalize``).

    Returns
    -------
    order : list
        Ordered list of leaf keys extracted from ``ttn.nodes``.
    mi_mat : ndarray
        Pairwise mutual information matrix used in the calculation.
    """
    leaves = kwargs.pop("leaves", None)
    mi_mat = kwargs.pop("mi_mat", None)
    if mi_mat is None:
        mi_mat = mutual_information_matrix(ttn, leaves=leaves, **kwargs)
    if method == "greedy":
        perm = greedy_leaf_ordering(mi_mat)
    else:
        perm = hierarchical_leaf_ordering(mi_mat)
    # map integer positions back to actual leaf keys
    if leaves is None:
        leaves = [k for k, n in ttn.nodes.items() if n.phys_inds]
    ordered_leaves = [leaves[i] for i in perm]
    return ordered_leaves, mi_mat
