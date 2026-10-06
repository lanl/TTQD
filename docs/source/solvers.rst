Tensor network (eigen) solvers
==============================

.. contents::
   :local:
   :depth: 2


.. automodule:: ttqd.solvers
   :members:
   :undoc-members:
   :show-inheritance:


DMRG Sweep Algorithm (Variational Ground State)
-----------------------------------------------

The density matrix renormalization group (DMRG) :cite:`White:1992dm` is an
adaptive variational algorithm for optimizing a matrix product state (MPS), also
known as a tensor train.  In :mod:`ttqd.solvers.dmrg`, the state is represented
by local tensors

.. math::

   A^{[i]}_{\alpha_i\,p_i\,\alpha_{i+1}},

with left virtual index :math:`\alpha_i`, physical index :math:`p_i`, and right
virtual index :math:`\alpha_{i+1}`.  The corresponding wavefunction is

.. math::

   \psi_{p_1\cdots p_L}
   =
   \sum_{\alpha_2,\ldots,\alpha_L}
   A^{[1]}_{\alpha_1 p_1 \alpha_2}
   A^{[2]}_{\alpha_2 p_2 \alpha_3}
   \cdots
   A^{[L]}_{\alpha_L p_L \alpha_{L+1}},

where the boundary virtual dimensions are usually one for an open chain.  The
Hamiltonian is represented as a matrix product operator (MPO),

.. math::

   W^{[i]}_{\mu_i\,\mu_{i+1}\,p'_i\,p_i},

with operator virtual indices :math:`\mu_i,\mu_{i+1}` and physical output/input
indices :math:`p'_i,p_i`.  The MPO tensors in the code therefore have shape
``(wL, wR, d_out, d_in)``.  The variational objective is

.. figure:: /images/H_tensor.png
   :scale: 50%
   :align: center
   :alt: A figure TBA

.. math::

   E[\psi]
   =
   \frac{\langle \psi | \hat H | \psi \rangle}
        {\langle \psi | \psi \rangle},
   \qquad
   |\psi\rangle \in \mathcal{M}_{\mathrm{MPS}}(\chi).

The solver uses sparse local eigensolves with ``which='SA'``; it therefore
targets the lowest-energy vector in the local variational subspace.

MPO environments
~~~~~~~~~~~~~~~~

The tensors ``Lenv`` and ``Renv`` store contractions of the MPS, MPO, and
conjugate MPS outside the active site or active bond.  They each have three
indices: two state virtual indices and one MPO virtual index.  The left boundary
is initialized as

.. math::

   L^{[1]}_{\alpha_1\,\mu_1\,\beta_1}
   =
   \delta_{\alpha_1\beta_1}\,\delta_{\mu_1,0},

and the right boundary as

.. math::

   R^{[L+1]}_{\alpha_{L+1}\,\mu_{L+1}\,\beta_{L+1}}
   =
   \delta_{\alpha_{L+1}\beta_{L+1}}\,
   \delta_{\mu_{L+1},w_R-1}.

These are the ``L0[:, 0, :] = I`` and ``R0[:, wD - 1, :] = I`` boundary
conditions in ``DMRG.init_envs``.

The left environment update implemented by ``update_left`` is

.. math::

   L^{[i+1]}_{\alpha_{i+1}\,\mu_{i+1}\,\beta_{i+1}}
   =
   \sum_{\alpha_i,\beta_i,\mu_i,p,q}
   L^{[i]}_{\alpha_i\,\mu_i\,\beta_i}\,
   A^{[i]}_{\alpha_i\,p\,\alpha_{i+1}}\,
   W^{[i]}_{\mu_i\,\mu_{i+1}\,p\,q}\,
   \overline{A^{[i]}_{\beta_i\,q\,\beta_{i+1}}}.

The right environment update implemented by ``update_right`` contracts one site
from the right:

.. math::

   R^{[i]}_{\alpha_i\,\mu_i\,\beta_i}
   =
   \sum_{\alpha_{i+1},\beta_{i+1},\mu_{i+1},p,q}
   A^{[i]}_{\alpha_i\,q\,\alpha_{i+1}}\,
   W^{[i]}_{\mu_i\,\mu_{i+1}\,p\,q}\,
   R^{[i+1]}_{\alpha_{i+1}\,\mu_{i+1}\,\beta_{i+1}}\,
   \overline{A^{[i]}_{\beta_i\,p\,\beta_{i+1}}}.

For an open chain, these recursions are the usual left-to-right and
right-to-left environment contractions.  The current implementation stores the
same objects with directed edge keys in ``psi.Lenv`` and ``psi.Renv`` so that the
MPS container can share traversal utilities with tree tensor networks.

One-site effective Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although the current ``DMRG.update_local`` routine performs two-site updates,
``Heff_1site`` defines the matrix-free one-site effective Hamiltonian.  For a
trial center tensor :math:`X_{\alpha_i p_i \alpha_{i+1}}`, its action is

.. math::

   \left(H_{\mathrm{eff}}^{[i]}X\right)_{
      \alpha_i\,p_i\,\alpha_{i+1}}
   =
   \sum_{\substack{\beta_i,\beta_{i+1}\\
                   \mu_i,\mu_{i+1},q_i}}
   L^{[i]}_{\alpha_i\,\mu_i\,\beta_i}\,
   W^{[i]}_{\mu_i\,\mu_{i+1}\,p_i\,q_i}\,
   X_{\beta_i\,q_i\,\beta_{i+1}}\,
   R^{[i+1]}_{\beta_{i+1}\,\mu_{i+1}\,\alpha_{i+1}}.

In a perfectly canonical MPS gauge, the overlap metric for this local problem is
the identity.  In a noncanonical representation the variational equation is the
generalized problem

.. math::

   H_{\mathrm{eff}}^{[i]} x
   =
   E\,N_{\mathrm{eff}}^{[i]} x,

where :math:`x=\mathrm{vec}(X)`.  In the canonical case used by the local
``LinearOperator`` this reduces to an ordinary eigenvalue problem.

Two-site update used by ``DMRG``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The active two-site tensor on bond :math:`(i,i+1)` is

.. math::

   \Theta_{\alpha_i\,p_i\,p_{i+1}\,\alpha_{i+2}}
   =
   \sum_{\alpha_{i+1}}
   A^{[i]}_{\alpha_i\,p_i\,\alpha_{i+1}}\,
   A^{[i+1]}_{\alpha_{i+1}\,p_{i+1}\,\alpha_{i+2}}.

``Heff_2site`` applies the effective Hamiltonian without explicitly forming the
large matrix:

.. math::

   \left(H_{\mathrm{eff}}^{[i,i+1]}\Theta\right)_{
      \alpha_i\,p_i\,p_{i+1}\,\alpha_{i+2}}
   =
   \sum_{\substack{\beta_i,\beta_{i+2}\\
                   \mu_i,\mu_{i+1},\mu_{i+2}\\
                   q_i,q_{i+1}}}
   L^{[i]}_{\alpha_i\,\mu_i\,\beta_i}\,
   W^{[i]}_{\mu_i\,\mu_{i+1}\,p_i\,q_i}\,
   W^{[i+1]}_{\mu_{i+1}\,\mu_{i+2}\,p_{i+1}\,q_{i+1}}\,
   \Theta_{\beta_i\,q_i\,q_{i+1}\,\beta_{i+2}}\,
   R^{[i+2]}_{\beta_{i+2}\,\mu_{i+2}\,\alpha_{i+2}}.

The local optimization solved in ``DMRG.update_local`` is

.. math::

   H_{\mathrm{eff}}^{[i,i+1]}\theta
   =
   E\,\theta,

where :math:`\theta=\mathrm{vec}(\Theta)`.  The initial Lanczos vector is the
current two-site wavefunction returned by ``psi.get_psi_2site(i, j)``, and the
lowest local eigenvector is computed by ``scipy.sparse.linalg.eigsh``.

After diagonalization, ``split_truncate_theta`` reshapes the optimized
:math:`\Theta` into a matrix

.. math::

   M_{(\alpha_i p_i),(p_{i+1}\alpha_{i+2})}
   =
   \Theta_{\alpha_i\,p_i\,p_{i+1}\,\alpha_{i+2}},

then computes a truncated SVD

.. math::

   M \approx U S V^\dagger.

The retained bond dimension is

.. math::

   \chi_{\mathrm{new}}
   =
   \min\left(\texttt{maxchi},\,\#\{s_k:s_k>\epsilon\}\right),

with :math:`\epsilon=10^{-8}` in ``DMRG.update_local``.  The code normalizes the
retained singular values by :math:`S\leftarrow S/\|S\|_2`, reshapes
:math:`U` into ``Ai`` with legs ``(vL, i, vC)``, reshapes :math:`V^\dagger` into
``Bj`` with legs ``(vC, j, vR)``, and writes ``(Ai, S, Bj)`` back through
``psi.update_bond(i, j, Ai, S, Bj)``.

Sweep and convergence
~~~~~~~~~~~~~~~~~~~~~

``DMRG.kernel`` initializes boundary and directed-edge environments with
``init_envs``.  Each call to ``sweep`` follows the MPS downward traversal, which
for a chain is the left-to-right order, and then sweeps back in the opposite
direction.  For every active bond, ``update_local``:

* chooses the current left and right environments ``prev_ledge`` and
  ``prev_redge``;
* constructs ``Heff_2site`` from those environments and the two local MPO
  tensors;
* solves the local lowest-eigenvalue problem;
* splits and truncates the optimized two-site tensor;
* updates the MPS tensors and refreshes the adjacent left/right environments.

Sweeps repeat until the absolute energy change between consecutive sweeps is
smaller than ``tol`` or until ``maxiter`` sweeps have been reached.

.. automodule:: ttqd.solvers.basesolver
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttqd.solvers.dmrg
   :members:
   :undoc-members:
   :show-inheritance:


Tree DMRG solvers
-----------------

Let :math:`T=(V,E)` be the acyclic graph shared by the tree tensor
network state (TTNS) and the tree tensor network operator (TTNO).  For a
node :math:`i`, denote its ordered neighbor set by
:math:`\partial i=(n_1,\ldots,n_{z_i})`; in the implementation this order is
``(parent, children...)`` and is returned by ``_edge_order(i)``.  The state
tensor stored at node :math:`i` is packed as

.. math::

    A^{[i]}_{\alpha_{i n_1}\cdots \alpha_{i n_{z_i}},\,p_i},

where :math:`\alpha_{ij}` is the virtual state bond on edge :math:`(i,j)` and
:math:`p_i` is the fused physical index of node :math:`i`.  With one shared
virtual index per edge, the TTNS coefficients are

.. math::

    \psi_{p_1\cdots p_{|V|}} = \sum_{\{\alpha_e\}_{e\in E}}
    \prod_{i\in V} A^{[i]}_{\{\alpha_{ij}:j\in\partial i\},\,p_i}.

The TTNO has the same tree topology.  Its local tensor on node :math:`i` is

.. math::

    W^{[i]}_{\omega_{i n_1}\cdots\omega_{i n_{z_i}},\,p'_i p_i},

where :math:`\omega_{ij}` is the operator auxiliary bond on edge
:math:`(i,j)`.  The global operator represented by the TTNO is

.. math::

    \hat H_{p'_1\cdots p'_{|V|},\,p_1\cdots p_{|V|}}
    = \sum_{\{\omega_e\}_{e\in E}} \prod_{i\in V}
    W^{[i]}_{\{\omega_{ij}:j\in\partial i\},\,p'_i p_i}.

In code this means
``W_i.shape == (*operator_edge_dims, d_out, d_in)``, with one operator edge
dimension for each neighbor in ``_edge_order(i)``.  ``_pack_state_node`` puts
the state tensor into the matching ``(*state_edge_dims, d)`` layout, fusing any
local physical legs into the trailing physical dimension :math:`d`.

Directed subtree messages
~~~~~~~~~~~~~~~~~~~~~~~~~

Cut an edge :math:`(u,v)`.  Removing this edge splits the tree into two
components; let :math:`C_{u|v}` be the component containing :math:`u`.  Since
the graph has no loops, every tensor in :math:`C_{u|v}` can be contracted
exactly into a directed message from :math:`u` to :math:`v`.

The norm message leaves open only the ket and bra state indices on the cut
bond:

.. math::

    N^{(u\to v)}_{\alpha\beta}
    =
    \sum_{\substack{\text{all physical indices in }C_{u|v}\\
                    \text{all internal virtual indices}}}
    \prod_{x\in C_{u|v}}
    A^{[x]}_{\mathrm{ket}}\,
    \overline{A^{[x]}_{\mathrm{bra}}}.

Equivalently, writing the neighbors of :math:`u` other than :math:`v` as
:math:`\partial u\setminus v`, the recursion used by ``_msg_norm`` is

.. math::

    N^{(u\to v)}_{\alpha\beta}
    =
    \sum_{p_u,\{\gamma_n,\delta_n\}}
    A^{[u]}_{\{\gamma_n\}_{n\ne v},\,\alpha,\,p_u}\,
    \overline{
        A^{[u]}_{\{\delta_n\}_{n\ne v},\,\beta,\,p_u}
    }
    \prod_{n\in\partial u\setminus v}
    N^{(n\to u)}_{\gamma_n\delta_n}.

The Hamiltonian message keeps the same two state-bond indices and also the
TTNO auxiliary index on the cut:

.. math::

    H^{(u\to v)}_{\alpha\,\omega\,\beta}
    =
    \sum_{\substack{p_u,q_u,\{\gamma_n,\delta_n\}\\
                    \{\omega_n\}_{n\ne v}}}
    A^{[u]}_{\{\gamma_n\}_{n\ne v},\,\alpha,\,p_u}\,
    W^{[u]}_{\{\omega_n\}_{n\ne v},\,\omega,\,p_u q_u}\,
    \overline{
        A^{[u]}_{\{\delta_n\}_{n\ne v},\,\beta,\,q_u}
    }
    \prod_{n\in\partial u\setminus v}
    H^{(n\to u)}_{\gamma_n\,\omega_n\,\delta_n}.

This is implemented by ``_msg_ham``.  For a leaf, the products over incoming
branch messages are empty products, equal to one.

One-site tree DMRG
~~~~~~~~~~~~~~~~~~

For an active node :math:`i`, let
:math:`\partial i=(n_1,\ldots,n_k)` and flatten

.. math::

    x \equiv \mathrm{vec}
    \left(A^{[i]}_{\alpha_1\cdots\alpha_k,\,p}\right).

All branches adjacent to :math:`i` are replaced by incoming messages
:math:`N^{(n_r\to i)}` and :math:`H^{(n_r\to i)}`.  The effective norm action is

.. math::

    \left(N_{\mathrm{eff}}^{(i)}x\right)_{\alpha_1\cdots\alpha_k,\,p}
    =
    \sum_{\beta_1,\ldots,\beta_k}
    \left[
        \prod_{r=1}^k
        N^{(n_r\to i)}_{\alpha_r\beta_r}
    \right]
    x_{\beta_1\cdots\beta_k,\,p}.

The effective Hamiltonian action is

.. math::

    \left(H_{\mathrm{eff}}^{(i)}x\right)_{\alpha_1\cdots\alpha_k,\,p}
    =
    \sum_{\substack{\beta_1,\ldots,\beta_k\\
                    \omega_1,\ldots,\omega_k\\ q}}
    W^{[i]}_{\omega_1\cdots\omega_k,\,p q}
    \left[
        \prod_{r=1}^k
        H^{(n_r\to i)}_{\alpha_r\,\omega_r\,\beta_r}
    \right]
    x_{\beta_1\cdots\beta_k,\,q}.

Thus the local Rayleigh quotient is

.. math::

    E[x]
    =
    \frac{x^\dagger H_{\mathrm{eff}}^{(i)} x}
         {x^\dagger N_{\mathrm{eff}}^{(i)} x},

and stationarity gives the generalized eigenvalue equation

.. math::

    H_{\mathrm{eff}}^{(i)}x
    =
    \lambda\,N_{\mathrm{eff}}^{(i)}x.

``_update_one_site`` builds these two matrix-free actions with ``np.einsum``,
solves for the lowest eigenvector, reshapes it back to
``(*edge_dims, d)``, and unpacks it into node :math:`i`.

Two-site tree DMRG
~~~~~~~~~~~~~~~~~~

For an active edge :math:`(i,j)`, define the external neighbor sets

.. math::

    I=\partial i\setminus j,\qquad J=\partial j\setminus i.

The current two-site center tensor is formed by contracting the state bond
:math:`\chi` on the active edge:

.. math::

    \Theta_{\boldsymbol{\alpha}\,p_i p_j\,\boldsymbol{\eta}}
    =
    \sum_\chi
    A^{[i]}_{\boldsymbol{\alpha},\,\chi,\,p_i}\,
    A^{[j]}_{\chi,\,\boldsymbol{\eta},\,p_j},

where :math:`\boldsymbol{\alpha}` collects the state indices on branches in
:math:`I` and :math:`\boldsymbol{\eta}` collects those in :math:`J`.  After
flattening :math:`x=\mathrm{vec}(\Theta)`, the effective norm is

.. math::

    \left(N_{\mathrm{eff}}^{(ij)}x\right)_{
        \boldsymbol{\alpha}\,p_i p_j\,\boldsymbol{\eta}}
    =
    \sum_{\boldsymbol{\beta},\boldsymbol{\zeta}}
    \left[
        \prod_{n\in I} N^{(n\to i)}_{\alpha_n\beta_n}
    \right]
    \left[
        \prod_{m\in J} N^{(m\to j)}_{\eta_m\zeta_m}
    \right]
    x_{\boldsymbol{\beta}\,p_i p_j\,\boldsymbol{\zeta}}.

The active TTNO bond :math:`\nu` between :math:`i` and :math:`j` is contracted
between the two local operator tensors, while all spectator branches enter as
Hamiltonian messages:

.. math::

    \left(H_{\mathrm{eff}}^{(ij)}x\right)_{
        \boldsymbol{\alpha}\,p_i p_j\,\boldsymbol{\eta}}
    =
    \sum_{\substack{\boldsymbol{\beta},\boldsymbol{\zeta}\\
                    \boldsymbol{\mu},\boldsymbol{\rho}\\
                    q_i,q_j,\nu}}
    W^{[i]}_{\boldsymbol{\mu},\,\nu,\,p_i q_i}\,
    W^{[j]}_{\nu,\,\boldsymbol{\rho},\,p_j q_j}
    \left[
        \prod_{n\in I}
        H^{(n\to i)}_{\alpha_n\,\mu_n\,\beta_n}
    \right]
    \left[
        \prod_{m\in J}
        H^{(m\to j)}_{\eta_m\,\rho_m\,\zeta_m}
    \right]
    x_{\boldsymbol{\beta}\,q_i q_j\,\boldsymbol{\zeta}}.

The same generalized problem is solved,

.. math::

    H_{\mathrm{eff}}^{(ij)}x
    =
    \lambda\,N_{\mathrm{eff}}^{(ij)}x.

The optimized tensor is then reshaped into a matrix with row index
:math:`(\boldsymbol{\alpha},p_i)` and column index
:math:`(p_j,\boldsymbol{\eta})`, followed by a truncated singular value
decomposition

.. math::

    \Theta_{\mathrm{opt}}
    \approx
    U\,S\,V^\dagger.

``split_truncate_theta`` keeps at most ``maxchi`` singular values and discards
values below ``cutoff``.  The implementation absorbs :math:`S` into the left
tensor, stores the retained singular values on the edge when the state object
supports it, and writes the resulting tensors back to nodes :math:`i` and
:math:`j`.

Numerical solve
~~~~~~~~~~~~~~~

Both one-site and two-site updates minimize the same local Rayleigh quotient.
``_solve_local_generalized`` treats small local spaces with a dense generalized
Hermitian eigensolve and larger spaces with ``scipy.sparse.linalg.eigsh`` using
matrix-free actions.  The optional ``norm_reg`` parameter defaults to zero.  A
positive value replaces
:math:`N_{\mathrm{eff}}` by
:math:`N_{\mathrm{eff}}+\epsilon I` in the solver to regularize nearly singular
overlap metrics, but it also perturbs the local variational problem.  The
sweep driver applies these local updates along a downward traversal and then an
upward traversal of the tree.


.. automodule:: ttqd.solvers.dmrg_tree
   :members:
   :undoc-members:
   :private-members:
   :show-inheritance:
