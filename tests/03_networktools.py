import unittest
import numpy as np

from ttqd.network import TreeTensorNetwork, TTNNode
from ttqd.tensors import Tensor
from ttqd.network import network_tools


class TestNetworkTools(unittest.TestCase):
    def setUp(self):
        # build simple chain tree 0-1-2 with physical legs at each node
        nodes = {
            0: TTNNode(
                Tensor(np.zeros((1, 2, 1)), ("b0", "p0", "b1")), phys_inds=("p0",)
            ),
            1: TTNNode(
                Tensor(np.zeros((1, 2, 2)), ("b1", "p1", "b2")), phys_inds=("p1",)
            ),
            2: TTNNode(
                Tensor(np.zeros((2, 2)), ("b2", "p2")), phys_inds=("p2",)
            ),
        }
        parent = {0: None, 1: 0, 2: 1}
        children = {0: (1,), 1: (2,), 2: ()}
        self.ttn = TreeTensorNetwork(nodes, parent, children, root=0)

    def test_leaf_distance_matrix(self):
        dist = network_tools.leaf_distance_matrix(self.ttn)
        expected = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        self.assertTrue(np.allclose(dist, expected))

    def test_mutual_information_cost(self):
        # MI values on chain
        mi = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        dist = network_tools.leaf_distance_matrix(self.ttn)
        cost = network_tools.mutual_information_cost(mi, dist)
        # manually compute: (1*1 + 1*1 + 1*2) * 2 since matrix counted twice
        # but our cost uses full matrix so result should be 4
        self.assertEqual(cost, 4.0)

    # FIXME:
    # def test_optimize_leaf_ordering_returns_valid(self):
    #     # running optimization should produce permutation of leaves
    #     ordered, mi_mat = network_tools.optimize_leaf_ordering(self.ttn)
    #     leaves = [k for k, n in self.ttn.nodes.items() if n.phys_inds]
    #     self.assertCountEqual(ordered, leaves)
    #     # cost can be computed from mi and distance
    #     dist = network_tools.leaf_distance_matrix(self.ttn)
    #     cost = network_tools.mutual_information_cost(mi_mat, dist)
    #     self.assertIsInstance(cost, float)

    def test_optimize_on_binary_tree(self):
        # build a small binary tree of 8 nodes
        ttn = TreeTensorNetwork.z_ary(N=8, Z=2, d=2, bond_dim=1)

        def fake_mi(u, v, base=2.0):
            return 0.1 * (u - v) ** 2.0
            par_u = ttn.parent.get(u)
            par_v = ttn.parent.get(v)
            if par_u is not None and par_u == par_v:
                return 1.0
            return 0.1
        ttn.mutual_information_leaves = fake_mi  # type: ignore

        ordered, mi_mat = network_tools.optimize_leaf_ordering(
            ttn, method="greedy"
        )
        print("Ordered nodes = ", ordered)
        print("mi_mat  = ", mi_mat)

        # expect siblings to be adjacent at least once
        sibling_adj = 0
        for a, b in zip(ordered, ordered[1:]):
            if ttn.parent.get(a) == ttn.parent.get(b):
                sibling_adj += 1
        self.assertGreaterEqual(sibling_adj, 1)


if __name__ == "__main__":
    unittest.main()
