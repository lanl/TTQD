import unittest
import numpy as np
from ttqd.linalg import set_backend
from ttqd.linalg import backend as np
set_backend("numpy")
from ttqd.tensors import Tensor, add_bond, contract_two


class TestTensor(unittest.TestCase):
    def setUp(self):
        self.A = Tensor(np.random.rand(20).reshape(4, 5), ("i", "j"))

    def test_repr_and_properties(self):
        self.assertIn("shape=(4, 5)", repr(self.A))
        self.assertEqual(self.A.shape, (4, 5))
        self.assertEqual(self.A.ndim, 2)
        self.assertEqual(self.A.size, 20)
        self.assertEqual(self.A.dtype, self.A.data.dtype)

    def test_add_index(self):
        B = self.A.add_index("k", 6, inplace=False)
        self.assertEqual(B.inds, ("i", "j", "k"))
        self.assertEqual(B.shape, (4, 5, 6))
        self.assertTrue(np.allclose(B.data[..., 0], self.A.data))

        B2 = self.A.add_index("z", 3, pos=0, inplace=False)
        self.assertEqual(B2.inds, ("z", "i", "j"))
        self.assertEqual(B2.shape, (3, 4, 5))
        self.assertTrue(np.allclose(B2.data[0, ...], self.A.data))

        B3 = self.A.add_index("m", 2, inplace=False)
        self.assertEqual(B3.inds[-1], "m")

        # adding existing index returns same object
        C = B.add_index("j", 5, inplace=False)
        self.assertIs(C, B)

        # in-place operation should modify tensor and return itself
        D = self.A.add_index("p", 2, inplace=True)
        self.assertIs(D, self.A)
        self.assertEqual(self.A.inds[-1], "p")
        self.assertEqual(self.A.shape, (4, 5, 2))

    def test_equality(self):
        A_copy = Tensor(self.A.data.copy(), self.A.inds, self.A.tags)
        self.assertEqual(self.A, A_copy)
        A_alt = Tensor(self.A.data, ("i", "k"))
        self.assertNotEqual(self.A, A_alt)

    def test_other_ops(self):
        D = self.A.reindex({"i": "x", "j": "y"})
        self.assertEqual(D.inds, ("x", "y"))
        T = self.A.transpose_to(("j", "i"))
        self.assertEqual(T.inds, ("j", "i"))
        self.assertTrue(np.allclose(T.data, self.A.data.T))

        # contract_two
        A1 = Tensor(np.ones((5, 5)), ("a", "b"))
        A2 = Tensor(np.random.rand(30).reshape(5, 6), ("b", "c"))
        C1 = contract_two(A1, A2)
        self.assertEqual(C1.inds, ("a", "c"))
        self.assertEqual(C1.shape, (5, 6))

    def test_make_bond(self):
        # print("test make bond")
        A1 = Tensor(np.ones((5, 5)), ("a", "b"))
        A2 = Tensor(np.random.rand(30).reshape(5, 6), ("b", "c"))
        add_bond(A1, A2, size=4)
        self.assertEqual(A1.shape, (4, 5, 5))
        self.assertEqual(A2.shape, (4, 5, 6))

    def test_uid_container(self):
        # ensure existing indices are recorded so get_uid doesn't reuse them
        from ttqd.tensors import IndexContainer, get_uid
        cont = IndexContainer()
        cont.add_many(["a", "b", "c"])
        # first available single char after a,b,c should be d
        uid = cont.get_uid()
        self.assertTrue(uid in ["d", "D", "α"])  # may pick lowercase first
        # using add_bond should register the partners automatically
        A = Tensor(np.zeros((2)), ("a",))
        B = Tensor(np.zeros((3)), ("b",))
        add_bond(A, B, size=1, container=cont)
        # next uid cannot be a or b
        uid2 = cont.get_uid()
        self.assertNotIn(uid2, ["a","b"])

        # check repeated generated bond names without growing one tensor past
        # NumPy's maximum supported rank.
        generated = set()
        for i in range(40):
            A_loop = Tensor(np.zeros((2,)), (f"a{i}",))
            B_loop = Tensor(np.zeros((3,)), (f"b{i}",))
            add_bond(A_loop, B_loop, size=1)
            shared = set(A_loop.inds).intersection(B_loop.inds)
            self.assertEqual(len(shared), 1)
            bond = shared.pop()
            self.assertNotIn(bond, generated)
            generated.add(bond)


if __name__ == "__main__":
    unittest.main()
