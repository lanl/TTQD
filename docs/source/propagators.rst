Tensor network propagators
==========================

.. contents::
   :local:
   :depth: 2

This module collects time-evolution methods for tensor-network quantum states
:cite:`Paeckel:2019aa`.  The common target is the time-dependent Schrodinger
equation

.. math::

   i\frac{d}{dt}|\psi(t)\rangle = \hat H |\psi(t)\rangle,
   \qquad
   |\psi(t+\Delta t)\rangle = e^{-i\Delta t \hat H}|\psi(t)\rangle,

or, for imaginary-time propagation,

.. math::

   \frac{d}{d\tau}|\psi(\tau)\rangle = -\hat H|\psi(\tau)\rangle,
   \qquad
   |\psi(\tau+\Delta\tau)\rangle = e^{-\Delta\tau \hat H}|\psi(\tau)\rangle.

The exact exponential generally leaves the fixed-bond-dimension tensor-network
manifold.  TEBD approximates the exponential by local two-site gates and
truncates after each gate.  TDVP instead projects the dynamics onto the tangent
space of the MPS/TTN manifold and integrates the projected equations.  Belief
propagation (BP) is not a time integrator, but is included here as the
message-passing language used for approximate tensor-network contractions.

Time propagator interface
-------------------------

``PropagatorBase`` stores the common simulation data:

* ``psi``: the tensor-network state to be evolved;
* ``mpo`` or local Hamiltonian data: the generator of time evolution;
* ``dt``, ``t0`` and ``tmax``: the time step and time interval;
* ``imaginary``: whether to use real-time factors :math:`-i\Delta t` or
  imaginary-time factors :math:`-\Delta\tau`;
* ``maxchi``: the largest bond dimension retained by methods that split
  two-site tensors.

Subclasses implement ``step(state)``.  The base ``run`` method repeatedly calls
``step`` and optionally invokes a user callback after each step:

.. math::

   |\psi_{n+1}\rangle = \mathcal{U}_{\mathrm{TN}}(\Delta t)|\psi_n\rangle,
   \qquad
   t_n = t_0 + n\Delta t.

The operator :math:`\mathcal{U}_{\mathrm{TN}}` is method dependent: for TEBD it
is a Trotterized product of gates; for TDVP it is a sequence of local projected
exponentials and gauge moves.

Local exponential actions
-------------------------

The TDVP routines repeatedly need the action of a local exponential on a vector
or tensor, rather than a dense global propagator.  If a local effective
Hamiltonian is represented as a matrix-free operator :math:`H_{\mathrm{eff}}`,
the basic real-time update is

.. math::

   x(t+\Delta t)
   =
   \exp\left(-i\Delta t\,H_{\mathrm{eff}}\right)x(t).

``TDVP1.expm_multiply`` uses ``scipy.sparse.linalg.expm_multiply`` on the
``Heff_0site``, ``Heff_1site`` and ``Heff_2site`` linear operators defined in
:mod:`ttqd.solvers.dmrg`.  ``lanczos_expm_multiply`` provides the same idea with
an explicit Lanczos basis.  Starting from :math:`q_1=x/\|x\|`, the Lanczos
recursion builds an orthonormal basis :math:`Q_m` and a small tridiagonal matrix
:math:`T_m=Q_m^\dagger H_{\mathrm{eff}}Q_m`; then

.. math::

   e^{zH_{\mathrm{eff}}}x
   \approx
   \|x\|\,Q_m e^{zT_m}e_1,
   \qquad
   z=-i\Delta t
   \quad\text{or}\quad
   z=-\Delta\tau.

For tree TDVP with a nontrivial local overlap metric :math:`N_{\mathrm{eff}}`,
the local equation is generalized:

.. math::

   i\,N_{\mathrm{eff}}\dot{x}=H_{\mathrm{eff}}x,
   \qquad
   x(t+\Delta t)
   =
   \exp\left(-i\Delta t\,N_{\mathrm{eff}}^{-1}H_{\mathrm{eff}}\right)x(t).

``TreeTTNOTDVP`` applies this either by dense metric whitening for moderate
local dimensions or by an iterative nested Krylov solve for larger local
spaces.

Time-Evolving Block Decimation (TEBD)
-------------------------------------

TEBD :cite:`Vidal:2003aa` is designed for Hamiltonians represented as a sum of
nearest-neighbor terms,

.. math::

   H=\sum_{i=1}^{L-1}h_{i,i+1}.

The code accepts ``h2_bonds`` as this list of two-site Hamiltonians.  Each term
may be shaped as a matrix :math:`(d^2,d^2)` or as a rank-four tensor.  For a
time interval :math:`\tau`, ``TEBD._build_gate`` constructs

.. math::

   G_{i,i+1}(\tau)
   =
   \begin{cases}
   \exp\left(-i\tau h_{i,i+1}\right), & \text{real time},\\
   \exp\left(-\tau h_{i,i+1}\right),  & \text{imaginary time},
   \end{cases}

by diagonalizing the Hermitian part of :math:`h_{i,i+1}`.  The gate is reshaped
as

.. math::

   G^{p'_i p'_{i+1}}_{p_i p_{i+1}},

which matches the ``(d, d, d, d)`` gate layout consumed by
``MPS.apply_two_site_gate``.

Even-odd decomposition
~~~~~~~~~~~~~~~~~~~~~~

For a chain, the Hamiltonian is split into commuting layers

.. math::

   H_{\mathrm{even}}=\sum_{i\ \mathrm{even}}h_{i,i+1},
   \qquad
   H_{\mathrm{odd}}=\sum_{i\ \mathrm{odd}}h_{i,i+1},
   \qquad
   H=H_{\mathrm{even}}+H_{\mathrm{odd}}.

The first-order update implemented by ``order=1`` is

.. math::

   U_1(\Delta t)
   =
   e^{-i\Delta t H_{\mathrm{even}}}
   e^{-i\Delta t H_{\mathrm{odd}}}
   +
   \mathcal{O}(\Delta t^2),

where each exponential is a product of non-overlapping two-site gates.  The
second-order Strang update implemented by ``order=2`` is

.. math::

   U_2(\Delta t)
   =
   e^{-i\frac{\Delta t}{2}H_{\mathrm{even}}}
   e^{-i\Delta t H_{\mathrm{odd}}}
   e^{-i\frac{\Delta t}{2}H_{\mathrm{even}}}
   +
   \mathcal{O}(\Delta t^3).

For imaginary time, replace each factor :math:`-i\Delta t` by
:math:`-\Delta\tau`.

Applying and truncating a gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an active bond :math:`(i,i+1)`, TEBD first forms the two-site MPS block

.. math::

   \Theta_{\alpha_i\,p_i\,p_{i+1}\,\alpha_{i+2}}
   =
   \sum_{\alpha_{i+1}}
   A^{[i]}_{\alpha_i p_i\alpha_{i+1}}
   A^{[i+1]}_{\alpha_{i+1}p_{i+1}\alpha_{i+2}}.

The gate acts only on the physical legs:

.. math::

   \widetilde{\Theta}_{\alpha_i\,p'_i\,p'_{i+1}\,\alpha_{i+2}}
   =
   \sum_{p_i,p_{i+1}}
   G^{p'_i p'_{i+1}}_{p_i p_{i+1}}\,
   \Theta_{\alpha_i\,p_i\,p_{i+1}\,\alpha_{i+2}}.

The evolved block is then reshaped into a matrix and split by SVD,

.. math::

   \widetilde{\Theta}_{(\alpha_i p'_i),(p'_{i+1}\alpha_{i+2})}
   \approx
   U S V^\dagger.

``MPS.apply_two_site_gate`` writes the two new tensors back and keeps at most
``maxchi`` singular values above ``cutoff``.  This truncation is the main source
of variational compression error, in addition to the Trotter error.

TDVP (Time-Dependent Variational Principle) on MPS
--------------------------------------------------

TDVP :cite:`Haegeman:2011aa` evolves within the variational manifold
:math:`\mathcal{M}_{\mathrm{MPS}}(\chi)` by projecting the Schrodinger equation
onto the tangent space:

.. math::

   i\,\partial_t|\psi\rangle
   =
   P_{T_\psi\mathcal{M}}\,\hat H|\psi\rangle.

The implementation uses the same MPO environments and effective Hamiltonian
objects as DMRG.  A left environment :math:`L^{[i]}` and a right environment
:math:`R^{[i]}` summarize the contraction outside the active site or bond.  The
one-site, zero-site and two-site effective Hamiltonian actions are represented
by ``Heff_1site``, ``Heff_0site`` and ``Heff_2site``.

MPS TDVP1
~~~~~~~~~

In mixed-canonical form, a one-site center tensor
:math:`A_C^{[i]}` is evolved forward under its local effective Hamiltonian:

.. math::

   A_C^{[i]}(t+\delta)
   =
   \exp\left(-i\delta H_{\mathrm{eff}}^{[i]}\right)
   A_C^{[i]}(t).

``TDVP1.evolve_one_site`` performs this update with ``Heff_1site``.  After the
site evolution, ``split_one_site_theta`` factorizes the center tensor by SVD.
When moving the orthogonality center from site :math:`i` to :math:`i+1`,

.. math::

   A_C^{[i]}{}_{\alpha p\beta}
   \rightarrow
   A_L^{[i]}{}_{\alpha p\gamma}\,C_{\gamma\beta}.

The bond-center matrix :math:`C` is then evolved backward:

.. math::

   C(t+\delta)
   =
   \exp\left(+i\delta H_{\mathrm{eff}}^{[i,i+1]}{}_{\mathrm{bond}}\right)
   C(t),

which appears in the code as ``evolve_zero_site`` with a negative local time
step in the projector-splitting sequence.  This backward bond evolution removes
the over-counting introduced by splitting the tangent-space projector into
site-local pieces.

The active ``TDVP1._sweep`` uses half steps on internal sites, a full step at
the boundary, then a return sweep:

.. math::

   \text{site forward} \;\rightarrow\;
   \text{SVD gauge move} \;\rightarrow\;
   \text{bond backward} \;\rightarrow\;
   \text{next site}.

Since no singular values are discarded in ``TDVP1.split_one_site_theta``, the
one-site method keeps the MPS bond dimensions fixed.

MPS TDVP2
~~~~~~~~~

Two-site TDVP evolves a block

.. math::

   \Theta_{\alpha_i\,p_i\,p_{i+1}\,\alpha_{i+2}}
   =
   \sum_{\alpha_{i+1}}
   A^{[i]}_{\alpha_i p_i\alpha_{i+1}}
   A^{[i+1]}_{\alpha_{i+1}p_{i+1}\alpha_{i+2}}

with ``Heff_2site``:

.. math::

   \Theta(t+\delta)
   =
   \exp\left(-i\delta H_{\mathrm{eff}}^{[i,i+1]}\right)\Theta(t).

``TDVP2.evolve_split_two_site`` then applies ``split_truncate_theta``:

.. math::

   \Theta_{(\alpha_i p_i),(p_{i+1}\alpha_{i+2})}
   \approx
   U S V^\dagger,
   \qquad
   \chi_{\mathrm{new}}\le \texttt{maxchi}.

This allows the bond dimension to grow or shrink according to the retained
singular values.  In the active ``TDVP2._sweep``, the carry tensor on the next
site is propagated backward by a one-site effective Hamiltonian before being
contracted with the following MPS tensor.  The helper ``_sweep_lr`` contains the
more explicit bond-center variant, where :math:`C=\operatorname{diag}(S)` is
evolved backward and then absorbed into the right tensor.

Tree TDVP with TTNS and TTNO
----------------------------

``TreeTTNOTDVP`` extends the TTNO message-passing machinery used by
``TreeTTNODMRG`` from variational eigensolves to time propagation.  For a node
or edge center, all external branches are contracted into incoming norm messages
:math:`N` and Hamiltonian messages :math:`H`, producing a local generalized
equation

.. math::

   i\,N_{\mathrm{eff}}\dot{x}
   =
   H_{\mathrm{eff}}x.

For real time,

.. math::

   x(t+\Delta t)
   =
   \exp\left(-i\Delta t\,N_{\mathrm{eff}}^{-1}H_{\mathrm{eff}}\right)x(t),

and for imaginary time,

.. math::

   x(\tau+\Delta\tau)
   =
   \exp\left(-\Delta\tau\,N_{\mathrm{eff}}^{-1}H_{\mathrm{eff}}\right)x(\tau).

One-site tree TDVP
~~~~~~~~~~~~~~~~~~

For an active node :math:`i` with neighbors
:math:`\partial i=(n_1,\ldots,n_k)`, the packed node tensor is

.. math::

   x=\mathrm{vec}\left(A^{[i]}_{\alpha_1\cdots\alpha_k,p}\right).

``_evolve_one_site`` builds the same effective Hamiltonian and metric actions
as the one-site tree DMRG local problem, but sends them to
``_apply_local_propagator`` instead of a generalized eigensolver.  Moving the
TDVP center across an edge :math:`(i,j)` is done by SVD:

.. math::

   A^{[i]} \rightarrow A_{\mathrm{iso}}^{[i]} C_{ij},

followed by a backward edge-center propagation of :math:`C_{ij}` and absorption
into the neighboring tensor.  This is implemented by
``_move_center_one_site`` through ``_propagate_edge_center``.

Two-site tree TDVP
~~~~~~~~~~~~~~~~~~

For an active edge :math:`(i,j)`, the two-site center tensor is formed by
contracting the TTNS bond between the two nodes and packing all other incident
branches as external indices.  The local propagation is

.. math::

   \Theta(t+\Delta t)
   =
   \exp\left(-i\Delta t\,N_{\mathrm{eff}}^{-1}H_{\mathrm{eff}}\right)
   \Theta(t),

with the imaginary-time replacement when ``imaginary=True``.  The result is
split with ``split_truncate_theta`` and bounded by ``maxchi`` and ``cutoff``.
The projector-splitting correction evolves the bond-center matrix backward with
effective actions constructed by ``_bond_center_matvecs``.

The tree sweep can be symmetric.  In symmetric mode, the code applies two
half-step projector sweeps; otherwise it applies one full sweep.  For small
trees, ``exact_small_system=True`` can contract the TTN/TTNO into a dense state
and Hamiltonian, evolve exactly, and refactor the result back onto the tree.

Time-Step Targeting Method (Local Krylov)
-----------------------------------------

The local Krylov idea is to approximate the action of an exponential on the
current tensor without constructing the full exponential.  For a local vector
:math:`x`, build the Krylov subspace

.. math::

   \mathcal{K}_m(H,x)
   =
   \operatorname{span}\{x,Hx,H^2x,\ldots,H^{m-1}x\}.

Lanczos or Arnoldi projection gives a small matrix :math:`T_m`; the update is

.. math::

   e^{zH}x
   \approx
   \|x\|Q_m e^{zT_m}e_1.

This is the numerical core behind ``lanczos_expm_multiply`` and behind SciPy's
``expm_multiply`` calls used in ``TDVP1`` and ``TDVP2``.  It is especially useful
for tensor networks because the effective Hamiltonians are naturally available
as contractions, i.e. as matrix-vector products, rather than as dense matrices.

Belief Propagation (BP) for Tensor Networks
-------------------------------------------

The ``Belief`` class is currently a placeholder, but the equations below are the
message-passing framework it is intended to implement.  BP rewrites a
contraction problem as inference on a factor graph.  Variables are indices
:math:`x_i`; factors are tensors or local weights :math:`f_a(x_{\partial a})`.
The unnormalized weight is

.. math::

   p(x) \propto \prod_a f_a(x_{\partial a}).

Sum-product BP
~~~~~~~~~~~~~~

Messages are functions on one variable.  The variable-to-factor update is

.. math::

   m_{i\to a}(x_i)
   =
   \frac{1}{Z_{i\to a}}
   \prod_{b\in\partial i\setminus a}
   m_{b\to i}(x_i),

and the factor-to-variable update is

.. math::

   m_{a\to i}(x_i)
   =
   \frac{1}{Z_{a\to i}}
   \sum_{x_{\partial a\setminus i}}
   f_a(x_{\partial a})
   \prod_{j\in\partial a\setminus i}
   m_{j\to a}(x_j).

The constants :math:`Z_{i\to a}` and :math:`Z_{a\to i}` normalize messages and
help avoid overflow or underflow.  Damped BP updates often use

.. math::

   m^{\mathrm{new}}
   \leftarrow
   (1-\eta)m^{\mathrm{old}}+\eta\,\widehat{m}^{\mathrm{new}},
   \qquad
   0<\eta\le 1,

where :math:`\widehat{m}^{\mathrm{new}}` is the raw update.

The approximate marginal, or belief, at a variable is

.. math::

   b_i(x_i)
   =
   \frac{1}{Z_i}
   \prod_{a\in\partial i}m_{a\to i}(x_i),

and the belief at a factor is

.. math::

   b_a(x_{\partial a})
   =
   \frac{1}{Z_a}
   f_a(x_{\partial a})
   \prod_{i\in\partial a}m_{i\to a}(x_i).

On a tree factor graph these beliefs are exact after messages have propagated
from the leaves inward and back outward.  On loopy graphs they define fixed
point equations and provide the usual loopy-BP approximation.

Tensor-network BP
~~~~~~~~~~~~~~~~~

A tensor network contraction is a factor graph in which each tensor is a factor
node and each bond index is a variable node.  For a tensor
:math:`T^{[a]}_{x_1\cdots x_k}`, the outgoing tensor-to-index message along
index :math:`x_r` is

.. math::

   m_{a\to r}(x_r)
   =
   \frac{1}{Z_{a\to r}}
   \sum_{\{x_s:s\ne r\}}
   T^{[a]}_{x_1\cdots x_k}
   \prod_{s\ne r} m_{s\to a}(x_s).

If an index connects exactly two tensors, the index-to-tensor message is just
the incoming message from the other tensor, up to normalization:

.. math::

   m_{r\to a}(x_r)
   \propto
   m_{b\to r}(x_r),
   \qquad
   \partial r=\{a,b\}.

For hyperedges or factor graphs where more than two factors share a variable,
the index-to-tensor message is the product of all other incoming messages.

Bethe partition function
~~~~~~~~~~~~~~~~~~~~~~~~

The full contraction is

.. math::

   Z = \sum_x \prod_a f_a(x_{\partial a}).

At a BP fixed point, the Bethe approximation is

.. math::

   \log Z_{\mathrm{Bethe}}
   =
   \sum_a \log Z_a
   -
   \sum_i (d_i-1)\log Z_i,

where :math:`d_i=|\partial i|` and

.. math::

   Z_a
   =
   \sum_{x_{\partial a}}
   f_a(x_{\partial a})
   \prod_{i\in\partial a}m_{i\to a}(x_i),
   \qquad
   Z_i
   =
   \sum_{x_i}\prod_{a\in\partial i}m_{a\to i}(x_i).

Max-product BP
~~~~~~~~~~~~~~

For maximum-a-posteriori or zero-temperature problems, replace sums by maxima:

.. math::

   m_{a\to i}(x_i)
   \propto
   \max_{x_{\partial a\setminus i}}
   f_a(x_{\partial a})
   \prod_{j\in\partial a\setminus i}
   m_{j\to a}(x_j).

In log space, products become sums:

.. math::

   \mu_{a\to i}(x_i)
   =
   \max_{x_{\partial a\setminus i}}
   \left[
      \log f_a(x_{\partial a})
      +
      \sum_{j\in\partial a\setminus i}\mu_{j\to a}(x_j)
   \right]
   + \mathrm{const}.


Functions
---------

.. automodule:: ttqd.solvers.propagators
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ttqd.solvers.tdvp_tree
   :members:
   :undoc-members:
   :private-members:
   :member-order: bysource
   :show-inheritance:
