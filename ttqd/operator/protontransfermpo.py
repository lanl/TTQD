import numpy as np
from typing import List, Tuple
from ttqd.lib import deprecated

r"""

"""
def boson_ops(d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bosonic operators in truncated Fock basis |0>,...,|d-1>.
    Returns: I, a, adag, n, x=(a+adag)
    """
    I = np.eye(d, dtype=np.complex128)
    a = np.zeros((d, d), dtype=np.complex128)
    for n in range(1, d):
        a[n-1, n] = np.sqrt(n)
    adag = a.conj().T
    n_op = adag @ a
    x_op = a + adag
    return I, a, adag, n_op, x_op

def two_level_ops() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-level system in basis |e>, |k>.
    Returns: I, Pee=|e><e|, Pkk=|k><k|, Sx = |e><k|+|k><e|
    """
    I = np.eye(2, dtype=np.complex128)
    Pee = np.array([[1, 0],[0, 0]], dtype=np.complex128)
    Pkk = np.array([[0, 0],[0, 1]], dtype=np.complex128)
    Sx  = np.array([[0, 1],[1, 0]], dtype=np.complex128)
    return I, Pee, Pkk, Sx

def protontransfer_mpo(
    w0e: float, w0k: float, x0e: float, x0k: float, Delta: float,
    dRC: int, d: int, N: int,
    bathparams,      # [[eps_0..],[t_0..], c0]
    RCparams,         # [wRC, g_scale]   (see note below)
    lambda_reorg: float,
    *,
    include_zero_point: bool = True,
    sign_g: float = -1.0,
) -> List[np.ndarray]:
    """Return the boundary-shaped proton-transfer MPO.

    MPO tensors W[i] with shapes (Dl, Dr, d_i, d_i).

    This wrapper delegates to :func:`protontransfer_mpo_fullD` and applies the
    boundary vectors afterwards so the boundary-shaped and full-bond versions
    stay exactly consistent.
    """
    return apply_boundaries(
        protontransfer_mpo_fullD(
            w0e,
            w0k,
            x0e,
            x0k,
            Delta,
            dRC,
            d,
            N,
            bathparams,
            RCparams,
            lambda_reorg,
            include_zero_point=include_zero_point,
            sign_g=sign_g,
        )
    )


@deprecated
def protontransfer_mpo(
    w0e: float, w0k: float, x0e: float, x0k: float, Delta: float,
    dRC: int, d: int, N: int,
    bathparams,      # [[eps_0..],[t_0..], c0]
    RCparams,         # [wRC, g_scale]   (see note below)
    lambda_reorg: float,
    *,
    include_zero_point: bool = True,
    sign_g: float = -1.0,
) -> List[np.ndarray]:
    """Return the boundary-shaped proton-transfer MPO.

    MPO tensors W[i] with shapes (Dl, Dr, d_i, d_i).

    Hamiltonian structure matches the docs:
      H_S + H_RC + H_{S-RC}
      + H_bath_chain + H_{RC-bath} + lambda_reorg * (x_RC)^2
    where x_RC = (d + d^^\dag), and bath coupling is via x operators.
    :contentReference[oaicite:2]{index=2}

    Notes on parameters:
      - bathparams = [eps_list, t_list, c0] as in the docs for chain mapping. :contentReference[oaicite:3]{index=3}
      - RCparams in the docs are described as [ω_RC, -g/x]. :contentReference[oaicite:4]{index=4}
        Here we use RCparams = [wRC, g_scale] and set:
            g_e = sign_g * g_scale * x0e
            g_k = sign_g * g_scale * x0k
        The sign convention varies between derivations; `sign_g` flips it.
    """
    if N < 1:
        raise ValueError("N must be >= 1 (number of bath chain sites).")

    eps_list, t_list, c0 = bathparams
    eps_list = list(eps_list)
    t_list = list(t_list)

    if len(eps_list) != N:
        raise ValueError(f"Expected len(eps_list)==N ({N}), got {len(eps_list)}")
    if len(t_list) != N - 1:
        raise ValueError(f"Expected len(t_list)==N-1 ({N-1}), got {len(t_list)}")

    wRC, g_scale = RCparams

    # --- local operators ---
    Is, Pee, Pkk, Sx = two_level_ops()
    IRC, aRC, adagRC, nRC, xRC = boson_ops(dRC)
    Ib, ab, adagb, nb, xb = boson_ops(d)

    # System Hamiltonian (2-level)
    Hs = w0e * Pee + w0k * Pkk + Delta * Sx

    # System-RC coupling operator on system site:
    #   (g_e |e><e| + g_k |k><k|) \otimes x_RC
    g_e = sign_g * g_scale * x0e
    g_k = sign_g * g_scale * x0k
    As_x = g_e * Pee + g_k * Pkk    # lives on system site
    BRC_x = xRC                     # lives on RC site

    # RC onsite Hamiltonian:
    HRC = wRC * nRC
    if include_zero_point:
        HRC = HRC + 0.5 * wRC * IRC
    HRC = HRC + (lambda_reorg * (xRC @ xRC))  # λ_reorg (d+d^\dag)^2 term :contentReference[oaicite:5]{index=5}

    # Bath onsite h_i = eps_i n_i
    # Bath hopping t_i (b_i^\dag b_{i+1} + b_i b_{i+1}^\dag)

    # RC-bath coupling on bond (RC, bath0): c0 * x_RC \otimes x_bath0
    # (Sign conventions differ; adjust c0 sign outside if needed)
    ARC_x_to_bath = c0 * xRC
    Bb0_x = xb

    # --- MPO channel design (finite-state automaton) ---
    # Channels k=1..K represent two-site terms A_i(k) \otimes B_{i+1}(k)
    # We'll use:
    #   k=1 : "x-type" channel (system-RC and RC-bath)
    #   k=2 : hopping channel for (b^\dag)_i \otimes (b)_{i+1}
    #   k=3 : hopping channel for (b)_i \otimes (b^\dag)_{i+1}
    K = 3
    D = K + 2           # bond dimension
    END = D - 1

    dims = [2, dRC] + [d] * N
    L = len(dims)       # total sites = 2 + N

    def make_W(di: int, Dl: int, Dr: int) -> np.ndarray:
        return np.zeros((Dl, Dr, di, di), dtype=np.complex128)

    W = []

    # Helper to place operator blocks
    def put(Wi, l, r, op):
        Wi[l, r, :, :] += op

    # Site 0: system
    Dl = 1
    Dr = D
    W0 = make_W(2, Dl, Dr)
    put(W0, 0, 0, Is)
    put(W0, 0, END, Hs)

    # incoming-from-left operators (B-channels) are on row 0, cols 1..K
    # For site 0 there is no incoming bond, but we still set row0/colk = Bk for uniformity if desired.
    # outgoing-to-right operators (A-channels) are on rows 1..K, col END.
    # Here we start the system-RC coupling on channel k=1 by placing A_x on (row=1, col=END),
    # and on site 1 (RC) we will place B_x on (row=0, col=1).
    put(W0, 1, END, As_x)   # channel 1: A on system
    # channels 2,3 unused on system
    put(W0, END, END, Is)
    W.append(W0)

    # Site 1: RC
    Dl = D
    Dr = D
    W1 = make_W(dRC, Dl, Dr)
    put(W1, 0, 0, IRC)
    put(W1, 0, END, HRC)

    # B-channel for system-RC coupling: B_x on (0,1)
    put(W1, 0, 1, BRC_x)

    # Start RC-bath coupling on same x-channel (k=1): A_x_to_bath on (1,END)
    put(W1, 1, END, ARC_x_to_bath)

    # No hopping channels from RC
    put(W1, END, END, IRC)
    W.append(W1)

    # Bath sites 0..N-1 map to MPO sites 2..L-1
    for j in range(N):
        di = d
        Dl = D
        Dr = D if j < N - 1 else 1   # last site closes MPO to bond dim 1
        Wj = make_W(di, Dl, Dr)

        put(Wj, 0, 0, Ib)
        put(Wj, 0, (END if Dr > 1 else 0), eps_list[j] * nb)  # onsite term goes to END (or only col if Dr=1)

        # Incoming B-operators on row 0, cols 1..K (if Dr>1)
        if Dr > 1:
            if j == 0:
                # RC-bath x coupling arrives here on channel 1
                put(Wj, 0, 1, Bb0_x)
            else:
                put(Wj, 0, 1, np.zeros_like(Ib))  # no x-coupling for later bath sites

            # hopping B-operators: b and b^\dag
            put(Wj, 0, 2, ab)     # channel 2 expects A = t * b^\dag on left
            put(Wj, 0, 3, adagb)  # channel 3 expects A = t * b on left

            # Outgoing A-operators on (row k, col END)
            if j < N - 1:
                t = t_list[j]
                put(Wj, 2, END, t * adagb)  # channel 2: A = t b^\dag
                put(Wj, 3, END, t * ab)     # channel 3: A = t b
            # else: last site has no outgoing hopping

            put(Wj, END, END, Ib)
        else:
            # Last site: Dr=1. Need to terminate channels by putting END->END identity into the only column.
            # Also we must accept incoming channels on left bond; those are already in Dl dimension.
            put(Wj, 0, 0, eps_list[j] * nb + Ib * 0.0)  # onsite already placed above
            put(Wj, END, 0, Ib)

        W.append(Wj)

    return W


def protontransfer_mpo_fullD(
    w0e: float, w0k: float, x0e: float, x0k: float, Delta: float,
    dRC: int, d: int, N: int,
    bathparams,      # [eps_list, t_list, c0]
    RCparams,         # [wRC, g_scale]
    lambda_reorg: float,
    *,
    include_zero_point: bool = True,
    sign_g: float = -1.0,
) -> List[np.ndarray]:
    if N < 1:
        raise ValueError("N must be >= 1.")

    eps_list, t_list, c0 = bathparams
    eps_list = list(eps_list)
    t_list   = list(t_list)

    if len(eps_list) != N:
        raise ValueError(f"Expected len(eps_list)==N ({N}), got {len(eps_list)}")
    if len(t_list) != N - 1:
        raise ValueError(f"Expected len(t_list)==N-1 ({N-1}), got {len(t_list)}")

    wRC, g_scale = RCparams

    # Local ops
    Is, Pee, Pkk, Ssys_x = two_level_ops()
    IRC, aRC, adagRC, nRC, xRC = boson_ops(dRC)
    Ib,  ab,  adagb,  nb,  xb  = boson_ops(d)

    # System H (2-level: |e>,|k>)
    Hs = w0e * Pee + w0k * Pkk + Delta * Ssys_x

    # Couplings
    g_e = sign_g * g_scale * x0e
    g_k = sign_g * g_scale * x0k
    As_x = g_e * Pee + g_k * Pkk  # operator on system site (to multiply xRC on RC site)

    # RC onsite
    HRC = wRC * nRC
    if include_zero_point:
        HRC = HRC + 0.5 * wRC * IRC
    HRC = HRC + lambda_reorg * (xRC @ xRC)

    # H_RC-bath = -c0 * x_RC \otimes x_bath0 + ...
    # ARC_x_to_bath = c0 * xRC
    ARC_x_to_bath = -c0 * xRC

    # MPO automaton channels
    # state 0 = "idle/start", state END = "done/end"
    # channel 1 = x-type (sys-RC completion and RC-bath start)
    # channel 2 = hop type (b^\dag on left, b on right)
    # channel 3 = hop type (b on left, b^\dag on right)
    K = 3
    D = K + 2
    END = D - 1

    def make_W(di: int) -> np.ndarray:
        return np.zeros((D, D, di, di), dtype=np.complex128)

    def put(Wi, l, r, op):
        Wi[l, r, :, :] += op

    W = []

    # --- Site 0: system (2-level) ---
    W0 = make_W(2)
    put(W0, 0, 0, Is)         # propagate idle
    put(W0, 0, END, Hs)       # local terms (can "finish" here)
    put(W0, 0, 1, As_x)       # START sys-RC term on channel 1
    put(W0, END, END, Is)     # propagate end
    W.append(W0)

    # --- Site 1: RC (boson) ---
    W1 = make_W(dRC)
    put(W1, 0, 0, IRC)
    put(W1, 0, END, HRC)

    # COMPLETE sys-RC: channel 1 -> END with xRC
    put(W1, 1, END, xRC)

    # START RC-bath on same channel 1: idle -> channel 1 with c0*xRC
    put(W1, 0, 1, ARC_x_to_bath)

    put(W1, END, END, IRC)
    W.append(W1)

    # --- Bath sites: j = 0..N-1 ---
    for j in range(N):
        Wj = make_W(d)
        put(Wj, 0, 0, Ib)
        put(Wj, 0, END, eps_list[j] * nb)
        put(Wj, END, END, Ib)

        # COMPLETE RC-bath on the first bath site (j==0): channel 1 -> END with xb
        if j == 0:
            put(Wj, 1, END, xb)

        # COMPLETE hopping from previous bath site (j>0):
        # previous site started: (t_{j-1} b^\dag) \otimes (b) and (t_{j-1} b) \otimes (b^\dag)
        if j > 0:
            put(Wj, 2, END, ab)     # channel 2 completion: ... b on right
            put(Wj, 3, END, adagb)  # channel 3 completion: ... b^\dag on right

        # START hopping to next bath site (j < N-1)
        if j < N - 1:
            t = t_list[j]
            put(Wj, 0, 2, t * adagb)  # start channel 2 with t*b^\dag
            put(Wj, 0, 3, t * ab)     # start channel 3 with t*b

        W.append(Wj)

    return W

def apply_boundaries(mpo_full: List[np.ndarray]) -> List[np.ndarray]:
    """
    Turn full-D MPO into boundary-shaped MPO:
      first tensor: (1, D, d, d)  by left vector selecting state 0
      last tensor:  (D, 1, d, d)  by right vector selecting state END
    """
    D = mpo_full[0].shape[0]
    END = D - 1
    vL = np.zeros(D, dtype=np.complex128); vL[0] = 1.0
    vR = np.zeros(D, dtype=np.complex128); vR[END] = 1.0

    W0 = np.tensordot(vL, mpo_full[0], axes=(0, 0))     # (D, d, d)
    W0 = W0[None, :, :, :]                              # (1, D, d, d)

    WN = np.tensordot(mpo_full[-1], vR, axes=(1, 0))     # (D, d, d)
    WN = WN[:, None, :, :]                              # (D, 1, d, d)

    return [W0] + mpo_full[1:-1] + [WN]
