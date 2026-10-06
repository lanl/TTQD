

r"""
Tree Tensor network state
=========================

"""

from typing import Any, Sequence, Union
from ttqd.lib import logger
from ttqd.linalg import backend
from ttqd.network import BaseTreeTensorNetwork, TTNNode
from ttqd.tensors import Tensor, add_bond

# ============================================================
# TreeTensorNetwork: General acyclic tree topology
# ============================================================

class TreeTensorNetwork(BaseTreeTensorNetwork):
    """
    General tree tensor network for arbitrary acyclic topologies.

    Inherits message-passing RDM and tree infrastructure from
    :class:`BaseTreeTensorNetwork`.  The user ordinarily supplies three
    pieces of information:

    ``nodes``
        A mapping from node key to :class:`TTNNode` objects.
    ``parent``
        A dictionary giving the parent of each node key (``None`` for the
        root).
    ``children``
        A dictionary giving the ordered tuple of children for each node.
    ``root``
        The key of the network root.

    The class also provides a convenience constructor for *balanced*
    binary trees.  The topology ensures that every internal node has two
    children (except the rightmost branch when there is an odd number of
    leaves); this ordering is useful for DMRG/TDVP calculations where one
    wishes to keep highly correlated sites close in the tree.

    Example of explicit initialization:

    >>> nodes = {
    ...     "leaf0": TTNNode(Tensor(A0, ("b0", "k0")), phys_inds=("k0",)),
    ...     "root": TTNNode(Tensor(R, ("b0", "b1")), phys_inds=()),
    ... }
    >>> parent = {"root": None, "leaf0": "root"}
    >>> children = {"root": ("leaf0",), "leaf0": ()}
    >>> ttn = TreeTensorNetwork(nodes, parent, children, root="root")
    >>> rho = ttn.make_rdm1("leaf0")

    Example of balanced construction:

    >>> leaf_arrays = [np.random.randn(1,2) for _ in range(5)]
    >>> ttn = TreeTensorNetwork.balanced(leaf_arrays)
    >>> # resulting tree has 5 leaves and 4 internal nodes
    >>> print(ttn.children)
    {
        5: (0, 1),
        6: (2, 3),
        7: (5, 6),
        8: (7, 4),
        0: (), 1: (), 2: (), 3: (), 4: ()
    }
    """

    def __init__(
        self,
        nodes: dict,
        parent: dict,
        children: dict,
        root: Any,
        **kwargs,
    ):

        """
        # suppose a N nodes with Z legs, bond dimension chi, physical dimension dims
        self._index = {}
        nodes = {}
        parent = {}
        children = {}
        dim = 2
        Z = nlegs = 3
        N = 1

        if N < 1:
            raise ValueError("N must be >= 1")
        if nlegs < 1:
            raise ValueError("nlegs must be >= 1")

        # create N empty tensors
        # add physical bond on each site
        for i in range(N):
            self.new_index(f"p{i}", size=dim)

        # connect nodes in breadth-first Z-ary fashion
        next_child = 1
        for parent in range(N):
            for _ in range(Z):
                if next_child >= N:
                    break
                # creates a new shared bond index between these two tensors
                ts[parent].new_bond(ts[next_child], size=bond_dim)
                next_child += 1
            if next_child >= N:
                break
        """


        super().__init__(nodes, parent, children, root, **kwargs)


    def new_index(self, label, size):
        self._index[label] = size


    # ------------------------------------------------------------------
    # convenience constructors
    # FIXME: call z_ary using Z=2 and with provided tensors
    # ------------------------------------------------------------------
    @classmethod
    def balanced(
        cls,
        tensors: Sequence[Any],
        tags_prefix: str = "L",
    ) -> "TreeTensorNetwork":
        """Return a balanced binary TTN whose nodes are the supplied arrays.

        Parameters
        ----------
        tensors : sequence
            List of arrays or tensors representing the nodes.  Each
            element should have a leading bond dimension which will become
            the leg connecting the node to its parent.  The remaining
            dimensions are treated as physical legs (contracted when
            computing RDMs).
        tags_prefix : str
            Prefix applied to the ``tags`` set of each tensor; useful
            for later selection.

        Returns
        -------
        ttn : TreeTensorNetwork
            Network with ``len(tensors)`` nodes arranged in a balanced
            binary tree.  Internal nodes carry random data (gaussian,
            unit‑variance) and no physical indices.
        """
        be = backend

        # --- create the nodes ------------------------------------------------
        nodes: dict = {}
        parent: dict = {}
        children: dict = {}
        bond_dims: dict = {}

        # use integer keys for simplicity
        for i, A in enumerate(tensors):
            A_arr = be.asarray(A)
            # assume first axis is bond to parent; if rank==1 default to 1
            chi = int(A_arr.shape[0]) if getattr(A_arr, "ndim", 0) >= 1 else 1
            bond_dims[i] = chi
            kin = f"k{i}"
            biname = f"b{i}"
            nodes[i] = TTNNode(Tensor(A_arr, (biname, kin), frozenset({f"{tags_prefix}{i}"})), phys_inds=(kin,))
            parent[i] = None
            children[i] = ()

        # --- iteratively build internal nodes by pairing -------------------------
        current = list(range(len(tensors)))
        next_id = len(current)
        while len(current) > 1:
            new_level: list = []
            it = iter(current)
            for a in it:
                try:
                    b = next(it)
                except StopIteration:
                    # odd leaf, carry over
                    new_level.append(a)
                    continue
                # bond dimensions of children
                chi_a = bond_dims[a]
                chi_b = bond_dims[b]
                # choose parent bond dimension (simple choice: 1)
                chi_p = 1
                bond_dims[next_id] = chi_p

                # indices connecting to children and parent
                ba = f"b{a}"
                bb = f"b{b}"
                bp = f"b{next_id}"

                # random tensor data for internal node
                T = be.asarray(np.random.randn(chi_a, chi_b, chi_p))
                nodes[next_id] = TTNNode(Tensor(T, (ba, bb, bp)), phys_inds=())
                parent[next_id] = None
                children[next_id] = (a, b)
                parent[a] = next_id
                parent[b] = next_id
                new_level.append(next_id)
                next_id += 1
            current = new_level

        root = current[0]
        return cls(nodes, parent, children, root)

    # ------------------------------------------------------------------
    # additional constructors
    # ------------------------------------------------------------------
    @classmethod
    def z_ary(
        cls,
        N: int,
        Z: int,
        d: int,
        bond_dim: int = 1,
    ) -> "TreeTensorNetwork":
        """Build a tree of ``N`` nodes with maximum degree ``Z``.

        Each tensor is given a single physical index of size ``d``; bonds
        between parent and children are added breadth-first with dimension
        ``bond_dim``.  The root node will be ``0`` and parents are assigned
        in numerical order.
        """
        if N < 1:
            raise ValueError("N must be >= 1")
        if Z < 1:
            raise ValueError("Z must be >= 1")

        be = backend
        # create bare tensors with physical leg only
        ts = []
        for i in range(N):
            arr = be.zeros((d,))
            ts.append(Tensor(arr, (f"p{i}",)))

        # bookkeeping for nodes/parents/children
        nodes: dict = {i: TTNNode(ts[i], phys_inds=(f"p{i}",)) for i in range(N)}
        parent: dict = {i: None for i in range(N)}
        children: dict = {i: () for i in range(N)}

        # connect the tensors in breadth-first Z-ary fashion
        next_child = 1
        for par in range(N):
            for _ in range(Z):
                if next_child >= N:
                    break
                add_bond(ts[par], ts[next_child], size=bond_dim)
                children[par] = children[par] + (next_child,)
                parent[next_child] = par
                next_child += 1
            if next_child >= N:
                break

        root = 0
        io_string = f"A Tree tensornet with {N} nodes and {Z} degree is created!"
        print(io_string)
        return cls(nodes, parent, children, root)

    @classmethod
    def from_networkx(
        cls,
        G,
        d: int,
    ) -> "TreeTensorNetwork":
        """Construct a TTN from a :mod:`networkx` graph.

        Each graph node becomes a tensor with one physical index of size
        ``d``.  Edges create shared bond indices; if the graph has a
        ``weight`` attribute on an edge it is interpreted as the bond
        dimension (must be integer), otherwise a size of ``1`` is used.

        The resulting tree is rooted at an arbitrary node (the first
        element of ``G.nodes()``) and parent/children mappings are
        determined via breadth-first traversal.  The input graph is assumed
        to be acyclic (a tree).
        """
        try:
            import networkx as nx  # noqa: F401
        except ImportError:  # pragma: no cover - optional dependency
            raise ImportError("networkx must be installed to use from_networkx")

        be = backend

        # initial tensors and trivial topology
        ts = {}
        nodes: dict = {}
        parent: dict = {}
        children: dict = {}
        for n in G.nodes():
            arr = be.zeros((d,))
            ts[n] = Tensor(arr, (f"p{n}",))
            nodes[n] = TTNNode(ts[n], phys_inds=(f"p{n}",))
            parent[n] = None
            children[n] = ()

        # add bonds for every edge
        for u, v, data in G.edges(data=True):
            size = int(data.get("weight", 1))
            add_bond(ts[u], ts[v], size=size)

        # orient the tree by choosing a root and doing BFS
        root = next(iter(G.nodes()))
        parent = {root: None}
        children = {n: () for n in G.nodes()}
        from collections import deque

        visited = {root}
        dq = deque([root])
        while dq:
            u = dq.popleft()
            for v in G.neighbors(u):
                if v in visited:
                    continue
                visited.add(v)
                parent[v] = u
                children[u] = children[u] + (v,)
                dq.append(v)

        ttn = cls(nodes, parent, children, root)
        ttn._created_from_nx = True
        return ttn

    def get_psi_1site(self, i, j):
        r"""Calculate effective single-site wave function on along the bond i-j in mixed canonical form.

        """
        import numpy as np
        print(f"Debug: in psi_1site: getting Sij between {i} and {j}")
        Sij = self.get_S(i, j)
        Ai = self.get_As(i, j) # Affective Ai, out-index is i->j bond
        print("Ai.shape =", Ai.shape, self.nodes[i].tensor.data.shape)
        # print("Aj.shape =", Aj.shape, self.nodes[j].tensor.data.shape)
        print("Sij.shape=", Sij.shape)
        return Ai # psi1 = np.tensordot(np.diag(Sij), Ai, [1, 0])
        return psi1

    def get_psi_2site(self, i, j):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``.

        Unlike MPS that each MPS node always has only in/out-coming virtual bonds, general TTN
        nodes can have more than one in/out-coming virtual bonds. Hence, we need to know the in/out-coming
        bonds in the construction of psi_1/2site.

        """
        import numpy as np

        print("Debug: constructing psi2")
        psi1 = self.get_psi_1site(i, j)
        print("psi1.shape=", psi1.shape)

        Aj = self.get_As(j, i, direction="right")
        print("Aj.shape =", Aj.shape)

        psi2 = np.tensordot(psi1, Aj, [2, 0]) #(vl, p, q, vR)
        print("psi2.shape=", psi2.shape)
        # print("psi2 = ", psi2)

        return psi2
