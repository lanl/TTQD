import unittest
import numpy as np

import networkx as nx

from ttqd.network import TreeTensorNetwork, MPS


class TestTreeTensorNetwork(unittest.TestCase):
    def test_z_ary(self):
        ttn = TreeTensorNetwork.z_ary(N=5, Z=2, d=3, bond_dim=4)
        # root should be 0 and it should have children
        self.assertEqual(ttn.root, 0)
        self.assertTrue(len(ttn.children[0]) >= 1)
        # physical indices present and correct size
        for i in range(5):
            node = ttn.nodes[i]
            # exactly one phys index per node
            self.assertEqual(len(node.phys_inds), 1)
            self.assertTrue(node.phys_inds[0].startswith("p"))
            # verify the dimension of the physical axis
            idx = node.phys_inds[0]
            dim = node.tensor.shape[node.tensor.inds.index(idx)]
            self.assertEqual(dim, 3)
        # every bond between parent and child should have bond_dim
        for par, childs in ttn.children.items():
            for ch in childs:
                shared = ttn._shared_bond_inds(par, ch)
                self.assertEqual(len(shared), 1)
                idx = shared[0]
                dim = ttn.nodes[par].tensor.shape[
                    ttn.nodes[par].tensor.inds.index(idx)
                ]
                self.assertEqual(dim, 4)

    def test_edge_singulars_dict(self):
        # after initialization the network should have a Schmidt vector for each
        # bond; there are N-1 edges in a tree with N nodes
        ttn = TreeTensorNetwork.z_ary(N=10, Z=2, d=2, bond_dim=5)
        # compute expected set of edges
        expected_edges = {frozenset({u, v}) for u, ch in ttn.children.items() for v in ch}
        self.assertEqual(set(ttn.Smap.keys()), expected_edges)
        # all entries should have the correct shape
        for S in ttn.Smap.values():
            self.assertEqual(S.shape, (5,))
        # getter helper should agree
        for edge in expected_edges:
            u, v = tuple(edge)
            # print(f"edge = {edge}  u/v = {u} {v}")
            self.assertIs(ttn.get_S(u, v), ttn.Smap[edge])

    def test_from_networkx_and_to_graph(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        G[0][1]["weight"] = 7
        ttn = TreeTensorNetwork.from_networkx(G, d=2)
        # root should be 0 by construction
        self.assertEqual(ttn.root, 0)
        # check parent/child relationships
        self.assertEqual(ttn.parent[0], None)
        self.assertEqual(ttn.parent[1], 0)
        self.assertEqual(ttn.parent[2], 1)
        # converted graph should preserve weights
        G2 = ttn._to_networkx_graph()
        self.assertEqual(G2[0][1]["weight"], 7)
        self.assertEqual(G2[1][2]["weight"], 1)

    def test_visualize(self):
        # Basic test that visualize doesn't crash and layout is hierarchical
        ttn = TreeTensorNetwork.z_ary(N=15, Z=2, d=2, bond_dim=1)
        # test with show=False to avoid display
        try:
            # fig = ttn.visualize(show=True)
            fig = ttn.visualize(show=False)
            self.assertIsNotNone(fig)
            # compute expected hierarchy positions ourselves and check root highest
            G = ttn._to_networkx_graph()
            def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
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
            pos = hierarchy_pos(G, ttn.root)
            # ensure root has largest y coordinate
            root_y = pos[ttn.root][1]
            for node, (x,y) in pos.items():
                self.assertGreaterEqual(root_y, y)
            # cleanup figure
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            # matplotlib not available, skip test
            pass

    # FIXME:
    # def test_get_As_set_As_roundtrip(self):
    #     # test that get_As and set_As are proper inverses, including cases where
    #     # the bond dimension changes (e.g. after truncation)
    #     ttn = TreeTensorNetwork.z_ary(N=5, Z=2, d=2, bond_dim=4)

    #     # pick an arbitrary edge in the tree
    #     i, j = 0, ttn.children[0][0]  # parent and first child

    #     # save original tensor
    #     A_orig = ttn.nodes[i].tensor.data.copy()

    #     # get reshaped 3-index tensor
    #     Ai_reshaped = ttn.get_As(i, j)
    #     self.assertEqual(len(Ai_reshaped.shape), 3)

    #     # simulate a truncation: reduce bond dimension to 2
    #     R, d, D_ij = Ai_reshaped.shape
    #     Ai_truncated = Ai_reshaped[:, :, :2]

    #     # set it back
    #     ttn.set_As(i, j, Ai_truncated)

    #     # retrieve it again
    #     Ai_roundtrip = ttn.get_As(i, j)

    #     # should match the truncated version
    #     np.testing.assert_allclose(Ai_roundtrip, Ai_truncated)

    #     # the original should have been modified
    #     A_new = ttn.nodes[i].tensor.data
    #     self.assertFalse(np.allclose(A_orig, A_new))

    def test_mps_visualization(self):
        # build a small random MPS and check structure and visualization
        mps = MPS.rand(L=6, d=2, chi=3)
        # topology should be chain 0-1-2-3
        G = mps._to_networkx_graph()
        edges = set(tuple(sorted(e)) for e in G.edges())
        expected = {(0,1),(1,2),(2,3),(3,4),(4,5)}
        self.assertTrue(expected.issubset(edges))
        # visualization
        try:
            # fig = mps.visualize(show=True)
            fig = mps.visualize(show=False)
            self.assertIsNotNone(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
