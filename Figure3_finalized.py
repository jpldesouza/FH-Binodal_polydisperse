from SharedFunctions_finalized import *


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def dash_pattern(k):
    """Return the dash pattern for curve index k."""
    if k == 0:
        return (2, 0)
    elif k == 1:
        return (2, 0.25)
    else:
        return (2, 0.1, 1, 0.1)


# ------------------------------------------------------------
# Global / shared settings
# ------------------------------------------------------------
nu_values = [0]
chi0 = np.array([0.8])

# Colors (reused)
colorsP = ["#a1dab4", "#41b6c4", "#225ea8"]

# Figure and axes
plt.figure(figsize=(6, 5))
axNorm = plt.subplot(2, 2, 1)
axMod = plt.subplot(2, 2, 2)
axBim = plt.subplot(2, 2, 3)
axFS = plt.subplot(2, 2, 4)

plt.suptitle(r"$\nu=0$", fontsize=16)


# ============================================================
# (a) Normal distribution
# ============================================================
N1 = 200
std_devs = [10, 20, 40]

# x-axis for chain lengths
Nmin, Nmax = 1, 450
Nmulti = np.arange(Nmin, Nmax + 1)

# Normal (Gaussian) curves
gaussian_curves = [
    1 / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N1) ** 2 / (2 * std_dev**2))
    for std_dev in std_devs
]

# Parameters for binodal
param = {"Nmulti": Nmulti}
alphamat = np.ones((len(Nmulti), len(Nmulti)))  # currently unused

logeps1vec = np.linspace(-50, -1e-2, 100)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

# Result matrices
phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

# Laguerre polynomials
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)

for k in range(3):
    umulti = gaussian_curves[k]
    nu = 0
    i = 0
    param["nu"] = nu

    chi_vec, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
        logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
    )
    chi_vec_app0, phiA_app0, phiB_app0, phiAmulti_app0, phiBmulti_app0 = approx_poly_binodal0(
        logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
    )

    phiAmat[i, :] = phiA
    phiBmat[i, :] = phiB
    chi_mat[i, :] = chi_vec

    phiAmat_app0[i, :] = phiA_app0
    phiBmat_app0[i, :] = phiB_app0
    chi_mat_app0[i, :] = chi_vec_app0

    dashset = dash_pattern(k)

    axNorm.plot(
        phiA,
        chi_vec,
        label=rf"$N_\sigma$ ={std_devs[k]}",
        linewidth=4,
        color=colorsP[k],
        linestyle="dashed",
        dashes=dashset,
    )
    axNorm.plot(
        phiB,
        chi_vec,
        linewidth=4,
        color=colorsP[k],
        linestyle="dashed",
        dashes=dashset,
    )

    # Black approximate curves
    if k > -2:
        axNorm.plot(phiA_app0, chi_vec_app0, linewidth=0.75, color="black", linestyle="-")
        axNorm.plot(phiB_app0, chi_vec_app0, linewidth=0.75, color="black", linestyle="-")

    axNorm.set_xscale("log")
    axNorm.set_xticks([1e-10, 1e-5, 1.0])
    axNorm.set_yticks([0.5, 0.7, 0.9])
    axNorm.set_ylim(0.5, 0.9)
    axNorm.set_xlim(1e-10, 1)
    axNorm.legend(fontsize=8, frameon=False, loc="lower left")

    if i == 0:
        labelstring = "(a)"
        axNorm.set_title("Normal", fontsize=12)
        axNorm.set_xlabel(r"$\phi_t$", fontsize=12)
        axNorm.set_ylabel(r"$\chi$", fontsize=12)
        nustring = "$\\nu=$" + str(nu)  # currently unused

    axNorm.annotate(
        labelstring,
        xy=(-0.35, 1.1),
        xycoords="axes fraction",
        fontsize=12,
        ha="left",
        va="top",
    )


# ============================================================
# (d) Flory–Schulz distribution
# ============================================================
N1 = 200
b_values = [0.025, 0.02, 0.01]

Nmin, Nmax = 1, 800
Nmulti = np.arange(Nmin, Nmax + 1)

curves = [bval**2 * Nmulti * (1 - bval) ** (Nmulti - 1) for bval in b_values]

param = {"Nmulti": Nmulti}

# Overwrite logeps1vec as in original code
logeps1vec = -np.logspace(-2, np.log10(50), 500)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

laguerre_polynomials = generate_laguerre_polynomials(num_terms)

for k in range(3):
    umulti = curves[k]
    N1 = 2 / b_values[k] - 1
    print(N1)

    for i, nu in enumerate(nu_values):
        param["nu"] = nu

        chi_vec, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )
        (
            chi_vec_app0,
            phiA_app0,
            phiB_app0,
            phiAmulti_app0,
            phiBmulti_app0,
        ) = approx_poly_binodal0(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmat[i, :] = phiA
        phiBmat[i, :] = phiB
        chi_mat[i, :] = chi_vec

        phiAmat_app0[i, :] = phiA_app0
        phiBmat_app0[i, :] = phiB_app0
        chi_mat_app0[i, :] = chi_vec_app0

        dashset = dash_pattern(k)

        axFS.plot(
            phiA,
            chi_vec,
            label=rf"$b$ ={1 - b_values[k]}",
            linewidth=3,
            color=colorsP[k],
            linestyle="dashed",
            dashes=dashset,
        )
        axFS.plot(
            phiB,
            chi_vec,
            linewidth=3,
            color=colorsP[k],
            linestyle="dashed",
            dashes=dashset,
        )

        if k > -2:
            axFS.plot(phiA_app0, chi_vec_app0, linewidth=0.75, color="black", linestyle="-")
            axFS.plot(phiB_app0, chi_vec_app0, linewidth=0.75, color="black", linestyle="-")

        axFS.set_xscale("log")
        axFS.set_xticks([1e-10, 1e-5, 1.0])
        axFS.set_yticks([0.5, 0.7, 0.9])
        axFS.set_ylim(0.5, 0.9)
        axFS.set_xlim(1e-10, 1)
        axFS.legend(fontsize=8, frameon=True, loc="lower left")

        if i == 0:
            labelstring = "(d)"
            axFS.set_title("Flory-Schulz", fontsize=12)
            axFS.set_xlabel(r"$\phi_t$", fontsize=12)
            axFS.set_ylabel(r"$\chi$", fontsize=12)
            nustring = "$\\nu=$" + str(nu)

        axFS.annotate(
            labelstring,
            xy=(-0.35, 1.1),
            xycoords="axes fraction",
            fontsize=12,
            ha="left",
            va="top",
        )

        # Locate eps for chi0 and recompute as in original
        logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)
        logeps1vec_at_chi0_app0 = find_y1(chi_vec_app0, logeps1vec, chi0)

        chival, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )
        print(chival)

        (
            chi_vec_app0,
            phiA_app0,
            phiB_app0,
            phiAmulti_app0,
            phiBmulti_app0,
        ) = approx_poly_binodal0(
            logeps1vec_at_chi0_app0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )


# ============================================================
# (b) Modified Poisson distribution
# ============================================================
N1 = 200
b_values = [180, 200, 220]

Nmin, Nmax = 100, 300
Nmulti = np.arange(Nmin, Nmax + 1)

curves = [evaluate_mod_poisson(bval, Nmulti) for bval in b_values]

Nshift = 0  # unused but kept for parity

param = {"Nmulti": Nmulti}

# reuse logeps1vec from Flory block (as in original)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

laguerre_polynomials = generate_laguerre_polynomials(num_terms)

for k in range(3):
    umulti = curves[k]
    N1 = b_values[k] + 2
    print(N1)

    for i, nu in enumerate(nu_values):
        param["nu"] = nu

        chi_vec, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )
        (
            chi_vec_app0,
            phiA_app0,
            phiB_app0,
            phiAmulti_app0,
            phiBmulti_app0,
        ) = approx_poly_binodal0(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmat[i, :] = phiA
        phiBmat[i, :] = phiB
        chi_mat[i, :] = chi_vec

        phiAmat_app0[i, :] = phiA_app0
        phiBmat_app0[i, :] = phiB_app0
        chi_mat_app0[i, :] = chi_vec_app0

        dashset = dash_pattern(k)

        axMod.plot(
            phiA,
            chi_vec,
            label=rf"$\lambda$ ={b_values[k]}",
            linewidth=3,
            color=colorsP[k],
            linestyle="dashed",
            dashes=dashset,
        )
        axMod.plot(
            phiB,
            chi_vec,
            linewidth=3,
            color=colorsP[k],
            linestyle="dashed",
            dashes=dashset,
        )

        if k > -2:
            axMod.plot(phiA_app0, chi_vec_app0, linewidth=0.75, color="black", linestyle="-")
            axMod.plot(phiB_app0, chi_vec_app0, linewidth=0.75, color="black", linestyle="-")

        axMod.set_xscale("log")
        axMod.set_xticks([1e-10, 1e-5, 1.0])
        axMod.set_yticks([0.5, 0.7, 0.9])
        axMod.set_ylim(0.5, 0.9)
        axMod.set_xlim(1e-10, 1)
        axMod.legend(fontsize=8, frameon=False, loc="lower left")

        if i == 0:
            labelstring = "(b)"
            axMod.set_title("Modified-Poisson", fontsize=12)
            axMod.set_xlabel(r"$\phi_t$", fontsize=12)
            axMod.set_ylabel(r"$\chi$", fontsize=12)
            nustring = "$\\nu=$" + str(nu)

        axMod.annotate(
            labelstring,
            xy=(-0.35, 1.1),
            xycoords="axes fraction",
            fontsize=12,
            ha="left",
            va="top",
        )

        logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)
        logeps1vec_at_chi0_app0 = find_y1(chi_vec_app0, logeps1vec, chi0)

        chival, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )
        (
            chi_vec_app0,
            phiA_app0,
            phiB_app0,
            phiAmulti_app0,
            phiBmulti_app0,
        ) = approx_poly_binodal0(
            logeps1vec_at_chi0_app0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )


# ============================================================
# (c) Bimodal distribution
# ============================================================
N1 = 150
N2 = 250
std_dev = 20.0
frac_vals = [0.25, 0.5, 0.75]

Nmin, Nmax = 1, 450
Nmulti = np.arange(Nmin, Nmax + 1)

gaussian_curves = [
    frac
    / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N1) ** 2 / (2 * std_dev**2))
    + (1 - frac)
    / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N2) ** 2 / (2 * std_dev**2))
    for frac in frac_vals
]

# colorsP already defined, reused

param = {"Nmulti": Nmulti}
alphamat = np.ones((len(Nmulti), len(Nmulti)))  # unused

eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

laguerre_polynomials = generate_laguerre_polynomials(num_terms)

for k in range(3):
    umulti = gaussian_curves[k]
    i = 0

    # nu was last set in the previous loops; original code reused it.
    # For clarity and identical behavior, we keep nu unchanged here.
    param["nu"] = nu

    chi_vec, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
        logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
    )
    (
        chi_vec_app0,
        phiA_app0,
        phiB_app0,
        phiAmulti_app0,
        phiBmulti_app0,
    ) = approx_poly_binodal0(
        logeps1vec,
        Nmulti,
        umulti,
        nu,
        N1 * frac_vals[k] + (1 - frac_vals[k]) * N2,
        laguerre_polynomials,
    )

    phiAmat[i, :] = phiA
    phiBmat[i, :] = phiB
    chi_mat[i, :] = chi_vec

    phiAmat_app0[i, :] = phiA_app0
    phiBmat_app0[i, :] = phiB_app0
    chi_mat_app0[i, :] = chi_vec_app0

    dashset = dash_pattern(k)

    axBim.plot(
        phiA,
        chi_vec,
        label=rf"$\eta$ ={frac_vals[k]}",
        linewidth=3,
        color=colorsP[k],
        linestyle="dashed",
        dashes=dashset,
    )
    axBim.plot(
        phiB,
        chi_vec,
        linewidth=3,
        color=colorsP[k],
        linestyle="dashed",
        dashes=dashset,
    )

    if k > -2:
        axBim.plot(phiA_app0, chi_vec_app0, linewidth=0.75, color="black", linestyle="-")
        axBim.plot(phiB_app0, chi_vec_app0, linewidth=0.75, color="black", linestyle="-")

    axBim.set_xscale("log")
    axBim.set_xticks([1e-10, 1e-5, 1.0])
    axBim.set_yticks([0.5, 0.7, 0.9])
    axBim.set_ylim(0.5, 0.9)
    axBim.set_xlim(1e-10, 1)
    axBim.legend(fontsize=8, frameon=False, loc="lower left")

    if i == 0:
        labelstring = "(c)"
        axBim.set_title("Bimodal", fontsize=12)
        axBim.set_xlabel(r"$\phi_t$", fontsize=12)
        axBim.set_ylabel(r"$\chi$", fontsize=12)
        nustring = "$\\nu=$" + str(nu)

    axBim.annotate(
        labelstring,
        xy=(-0.35, 1.1),
        xycoords="axes fraction",
        fontsize=12,
        ha="left",
        va="top",
    )


# ------------------------------------------------------------
# Final layout and save
# ------------------------------------------------------------
plt.tight_layout()
plt.savefig("figure3ad_final.pdf", format="pdf")
plt.show()
