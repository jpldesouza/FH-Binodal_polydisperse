import numpy as np
from scipy.special import genlaguerre, gammaln
from scipy.interpolate import interp1d


# ============================================================
# Distributions and utilities
# ============================================================
def evaluate_mod_poisson(lmbda, x):
    """
    Modified Poisson-like weight:

        w_x = (λ / (λ + 1)) * e^{-λ} * λ^{x-2} / Γ(x)

    Implemented in log-space for numerical stability.
    """
    log_term1 = np.log(lmbda) - np.log(lmbda + 1)
    log_term2 = -lmbda
    log_term3 = (x - 2) * np.log(lmbda)
    log_term4 = gammaln(x)

    log_w_x = log_term1 + log_term2 + log_term3 - log_term4
    return np.exp(log_w_x)


def mu(phi, param):
    """
    Chemical potential-like quantity:
        μ = ln(phi)/N - ln(1 - sum(phi)) - 2 χ sum(phi)
    """
    N_vec = param["Nmulti"]
    chimat = param["chimat"]

    sum_phi = np.sum(phi)
    return np.log(phi) / N_vec - np.log(1 - sum_phi) - 2 * chimat * sum_phi


def Pi(phi, param):
    """
    Osmotic pressure-like quantity:
        Π = sum((1/N - 1) phi) - ln(1 - sum(phi)) - phiᵀ χ phi
    """
    N_vec = param["Nmulti"]
    chimat = param["chimat"]

    sum_phi = np.sum(phi)
    term1 = np.sum((1.0 / N_vec - 1.0) * phi)
    term2 = -np.log(1 - sum_phi)
    term3 = -np.dot(phi, np.dot(chimat, phi.T))

    return term1 + term2 + term3


def generate_laguerre_polynomials(num_terms):
    """
    Generate generalized Laguerre polynomials L_{n-1}^{(1)} for n=1..num_terms.
    Returns a list of callable polynomials.
    """
    return [genlaguerre(n - 1, 1) for n in range(1, num_terms + 1)]


def h(x):
    """
    Simple h(x) = arctanh(x)/x used elsewhere.
    NOTE: This matches the *final* definition in your original code.
    """
    return np.arctanh(x) / x


# ============================================================
# Inverse h via Laguerre expansion
# ============================================================
def hinv(hx, laguerre_polynomials, num_terms=100):
    """
    Approximate inverse of h(x) returning log(eps) = log(1 - z)
    via different asymptotic regimes and a Laguerre expansion.
    """
    logeps = np.zeros_like(hx)

    # Regions in hx
    cond1 = hx < 1.076
    cond2 = hx > 5.0
    cond3 = ~cond1 & ~cond2  # 1.076 <= hx <= 5

    # Small-hx asymptotic expression
    hx1 = hx[cond1]
    if hx1.size > 0:
        num = (
            637875 * hx1
            + np.sqrt((637875 * hx1 - 557172) ** 2 + 5833096416)
            - 557172
        ) ** (1.0 / 3.0)
        term1 = num / (45 * 2 ** (1.0 / 3.0))
        term2 = (126 * 2 ** (1.0 / 3.0)) / (5 * num)
        logeps[cond1] = np.log(
            1
            - np.sqrt(
                term1
                - term2
                - 7.0 / 15.0
            )
        )

    # Large-hx asymptotic expression
    hx2 = hx[cond2]
    if hx2.size > 0:
        logeps[cond2] = np.log(2.0) - 2.0 * hx2

    # Intermediate regime via Laguerre expansion
    hx3 = hx[cond3]
    if hx3.size > 0:
        logeps_cond3 = np.zeros_like(hx3)
        for n in range(num_terms):
            # laguerre_polynomials[n] corresponds to L_{n}^{(1)} (since list is n=0..num_terms-1)
            L_n_minus_1_1 = laguerre_polynomials[n]
            k = n + 1
            # Same expression as original code
            term = (
                -(-1) ** (n + 1)
                * (2.0 * np.exp(-2.0 * hx3 * k) / k)
                * L_n_minus_1_1(4.0 * hx3 * k)
            )
            logeps_cond3 += term

        logeps_cond3 = np.log(logeps_cond3)
        logeps[cond3] = logeps_cond3

    return logeps


# ============================================================
# Full binodal computation
# ============================================================
def compute_poly_binodal(logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials):
    """
    Full polymer binodal computation.

    Inputs:
        logeps1vec : log(ε₁) grid
        Nmulti     : array of chain lengths
        umulti     : parent length distribution
        nu         : kinetic/regularization parameter
        N1         : reference chain length
        laguerre_polynomials : precomputed list from generate_laguerre_polynomials

    Returns:
        chi_vec, phiA, phiB, phiAmulti, phiBmulti
    """
    # Normalize so that u(N=N1) = 1
    index = np.argmin(np.abs(Nmulti - N1))
    umulti = umulti / umulti[index]

    # Scalar grid quantities
    eps1vec = np.exp(logeps1vec)
    y1vec = 1.0 - eps1vec

    # Broadcasted matrices
    num_y = len(y1vec)
    num_N = len(Nmulti)

    Nmultimat = np.tile(Nmulti, (num_y, 1)).T
    umultimat = np.tile(umulti, (num_y, 1)).T

    logeps1mat = np.tile(logeps1vec, (num_N, 1))
    eps1mat = np.exp(logeps1mat)
    y1mat = 1.0 - eps1mat

    # atanh(y1) in vector/matrix form
    atanhy1vec = 0.5 * np.log(2.0 - eps1vec) - 0.5 * logeps1vec
    atanhy1mat = 0.5 * np.log(2.0 - eps1mat) - 0.5 * logeps1mat

    # y for each N: ymulti = tanh(N/N1 * atanh(y1))
    atanhymulti = (Nmultimat / N1) * atanhy1mat
    ymulti = np.tanh(atanhymulti)

    # Small-ymulti region (|y| < 0.1)
    cond1 = np.abs(ymulti) < 0.1
    cond2 = ~cond1

    # Weight matrix w(N,eps)
    wmulti = np.zeros_like(ymulti)

    # General expression
    numerator = nu + 0.5 * (eps1mat[cond2] / y1mat[cond2])
    denom = nu + 1.0 / (np.exp(np.clip(2.0 * atanhymulti[cond2], -700, 700)) - 1.0)
    wmulti[cond2] = umultimat[cond2] * numerator / denom

    # Small-y expansion (same algebra as original)
    num_small = (
        nu * ymulti[cond1] * y1mat[cond1] * umultimat[cond1]
        + 0.5 * ymulti[cond1] * umultimat[cond1]
        - 0.5 * ymulti[cond1] * y1mat[cond1] * umultimat[cond1]
    )
    den_small = (
        nu * ymulti[cond1] * y1mat[cond1]
        + 0.5 * y1mat[cond1]
        - 0.5 * ymulti[cond1] * y1mat[cond1]
    )
    wmulti[cond1] = num_small / den_small

    # Integrals I1, I2, I3 (sum over sorted N axis)
    I1 = np.sum(np.sort(wmulti, axis=0), axis=0)
    I2 = np.sum(
        np.sort(wmulti * (atanhymulti / ymulti - 1.0) / Nmultimat, axis=0),
        axis=0,
    )
    I3 = np.sum(np.sort(wmulti / ymulti, axis=0), axis=0)

    hz = 1.0 + I2 / I1
    print(I1)
    print(hz)

    logepsZvec = hinv(hz, laguerre_polynomials)
    epsZvec = np.exp(logepsZvec)
    zvec = 1.0 - epsZvec
    atanhzvec = zvec * hz

    beta_multi_times_ymulti = 2.0 * wmulti / (I1 / zvec + I3)

    phiAmulti = np.zeros_like(ymulti)
    phiBmulti = np.zeros_like(ymulti)

    # General ymulti expression
    exp_pos = np.exp(np.clip(2.0 * atanhymulti[cond2], -700, 700))
    exp_neg = np.exp(np.clip(-2.0 * atanhymulti[cond2], -700, 700))

    phiAmulti[cond2] = beta_multi_times_ymulti[cond2] / (1.0 - exp_neg)
    phiBmulti[cond2] = beta_multi_times_ymulti[cond2] / (-1.0 + exp_pos)

    # Small-ymulti expansion
    phiAmulti[cond1] = 0.5 * beta_multi_times_ymulti[cond1] / ymulti[cond1] * (
        1.0 + ymulti[cond1]
    )
    phiBmulti[cond1] = 0.5 * beta_multi_times_ymulti[cond1] / ymulti[cond1] * (
        1.0 - ymulti[cond1]
    )

    # Total phase volume fractions
    phiA = np.sum(np.sort(phiAmulti, axis=0), axis=0)
    phiB = np.sum(np.sort(phiBmulti, axis=0), axis=0)

    # χ(k) or χ(eps) expression
    chi_vec = (
        2.0 * atanhzvec
        + np.sum(
            np.sort((1.0 / Nmultimat - 1.0) * (phiAmulti - phiBmulti), axis=0),
            axis=0,
        )
    ) / (phiA**2 - phiB**2)

    return chi_vec, phiA, phiB, phiAmulti, phiBmulti


# ============================================================
# Helper: find logeps1 such that χ = χ₀
# ============================================================
def find_y1(chi_vec, logeps1vec, chi0):
    """
    Interpolate logeps1 at given chi0.
    """
    # chi_vec need not be sorted for interp1d with 'extrapolate'
    interpolation_function = interp1d(
        chi_vec,
        logeps1vec,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    logeps1vec_at_chi0 = interpolation_function(chi0)
    return np.array(logeps1vec_at_chi0)


# ============================================================
# Approximate binodal (0th-order)
# ============================================================
def approx_poly_binodal0(logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials):
    """
    Approximate polymer binodal (0th-order) using analytic simplifications.
    Returns the same tuple as compute_poly_binodal.
    """
    index = np.argmin(np.abs(Nmulti - N1))

    eps1vec = np.exp(logeps1vec)
    y1vec = 1.0 - eps1vec

    num_y = len(y1vec)
    num_N = len(Nmulti)

    Nmultimat = np.tile(Nmulti, (num_y, 1)).T
    umultimat = np.tile(umulti, (num_y, 1)).T

    logeps1mat = np.tile(logeps1vec, (num_N, 1))
    eps1mat = np.exp(logeps1mat)
    y1mat = 1.0 - eps1mat

    atanhy1vec = 0.5 * np.log(2.0 - eps1vec) - 0.5 * logeps1vec
    atanhy1mat = 0.5 * np.log(2.0 - eps1mat) - 0.5 * logeps1mat

    atanhymulti = (Nmultimat / N1) * atanhy1mat
    ymulti = np.tanh(atanhymulti)

    cond1 = np.abs(y1mat * Nmultimat) < 0.1
    cond2 = ~cond1

    wmulti = np.zeros_like(ymulti)

    # General expression
    numerator = nu + 0.5 * (eps1mat[cond2] / y1mat[cond2])
    denom = nu + 1.0 / (np.exp(np.clip(2.0 * atanhymulti[cond2], -700, 700)) - 1.0)
    wmulti[cond2] = umultimat[cond2] * numerator / denom

    # Small-y expansion (second expression in your original code)
    num_small = (
        nu * ymulti[cond1] * y1mat[cond1] * umultimat[cond1]
        + 0.5 * ymulti[cond1] * umultimat[cond1]
        - 0.5 * ymulti[cond1] * y1mat[cond1] * umultimat[cond1]
    )
    den_small = (
        nu * ymulti[cond1] * y1mat[cond1]
        + 0.5 * y1mat[cond1]
        - 0.5 * ymulti[cond1] * y1mat[cond1]
    )
    wmulti[cond1] = num_small / den_small

    # Approximate integrals
    I1 = 1.0 / umulti[index]
    I2 = (atanhy1vec / y1vec - 1.0) / (N1 * umulti[index])
    I3 = 1.0 / (umulti[index] * y1vec)

    hz = 1.0 + I2 / I1
    logepsZvec = hinv(hz, laguerre_polynomials)
    epsZvec = np.exp(logepsZvec)
    zvec = 1.0 - epsZvec

    beta_multi_times_ymulti = 2.0 * wmulti / (I1 / zvec + I3)

    phiAmulti = np.zeros_like(ymulti)
    phiBmulti = np.zeros_like(ymulti)

    # General expression
    exp_pos = np.exp(np.clip(2.0 * atanhymulti[cond2], -700, 700))
    phiAmulti[cond2] = beta_multi_times_ymulti[cond2] / (1.0 - np.exp(-2.0 * atanhymulti[cond2]))
    phiBmulti[cond2] = beta_multi_times_ymulti[cond2] / (-1.0 + exp_pos)

    # Small-y expansion
    phiAmulti[cond1] = 0.5 * beta_multi_times_ymulti[cond1] / ymulti[cond1] * (
        1.0 + ymulti[cond1]
    )
    phiBmulti[cond1] = 0.5 * beta_multi_times_ymulti[cond1] / ymulti[cond1] * (
        1.0 - ymulti[cond1]
    )

    # Phase volume fractions (analytic)
    phiA = zvec * (I3 + I1) / (I1 + zvec * I3)
    phiB = zvec * (I3 - I1) / (I1 + zvec * I3)

    chi_vec = ((I1 + zvec * I3) / (2.0 * zvec * I1)) * (
        (1.0 / N1) * atanhy1vec + np.arctanh(zvec)
    )

    return chi_vec, phiA, phiB, phiAmulti, phiBmulti
