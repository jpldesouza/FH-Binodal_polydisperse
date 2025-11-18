import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # noqa: F401

from SharedFunctions_finalized import *


# ============================================================
# Figure and global style
# ============================================================
fig = plt.figure(figsize=(6, 5))
ax4 = plt.subplot(2, 2, 1)  # Normal
ax6 = plt.subplot(2, 2, 2)  # Modified Poisson
ax7 = plt.subplot(2, 2, 3)  # Bimodal
ax5 = plt.subplot(2, 2, 4)  # Flory–Schulz

fig.suptitle(r"$\chi=0.65$", fontsize=16)

# Colors
colorsB = ["#bdd7e7", "#6baed6", "#2171b5"]  # Different shades of blue (unused here)
colorsP = ["#cbc9e2", "#9e9ac8", "#6a51a3"]  # Different shades of purple
colorsR = ["#fcae91", "#fb6a4a", "#cb181d"]  # Different shades of red

# ============================================================
# Shared thermodynamic / numerical parameters
# ============================================================
nu_values = [0]
chi0 = np.array([0.65])

logeps1vec = np.linspace(-100, -1e-6, 2000)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

# Laguerre polynomials (shared by all panels)
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)


# ============================================================
# Helper to run binodal and return phase A distribution
# ============================================================
def compute_phaseA_distribution(Nmulti, umulti, N1):
    """
    Given a length distribution umulti on Nmulti and reference N1,
    compute the coexisting Phase A distribution at chi0.
    """
    nu = nu_values[0]  # only one value in all cases

    chi_vec, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
        logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
    )

    # Find logeps1vec corresponding to chi0 and recompute
    logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)

    chival, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
        logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
    )

    return phiAmulti


# ============================================================
# Panel (e): Normal Distribution
# ============================================================
N1_normal = 200
std_devs = [10, 20, 40]

Nmin = 1
Nmax = 450
Nmulti_normal = np.arange(Nmin, Nmax + 1)

gaussian_curves = [
    1
    / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti_normal - N1_normal) ** 2 / (2 * std_dev**2))
    for std_dev in std_devs
]

for k in range(3):
    base_dist = gaussian_curves[k]
    phiAmulti = compute_phaseA_distribution(Nmulti_normal, base_dist, N1_normal)

    # Parent distribution
    ax4.plot(
        Nmulti_normal,
        base_dist / np.sum(base_dist),
        label=rf"$N_\sigma$ = {std_devs[k]}",
        linewidth=3,
        color=colorsP[k],
        alpha=0.8,
    )

    # Phase A distribution
    if k == 2:
        ax4.plot(
            Nmulti_normal,
            phiAmulti / np.sum(phiAmulti),
            linewidth=3,
            color=colorsR[k],
            alpha=0.8,
            linestyle="dashed",
            dashes=(2, 0.25),
            label="Phase A",
        )
    else:
        ax4.plot(
            Nmulti_normal,
            phiAmulti / np.sum(phiAmulti),
            linewidth=3,
            color=colorsR[k],
            alpha=0.8,
            linestyle="dashed",
            dashes=(2, 0.25),
        )

ax4.set_xlim(100, Nmax)
ax4.set_xticks([1, 200, 400])
ax4.set_yticks([0.0, 0.02, 0.04])
ax4.set_title("Normal", fontsize=12)
ax4.set_xlabel("N", fontsize=12)
ax4.set_ylabel("p(N)", fontsize=12)
ax4.set_ylim(bottom=0)
ax4.legend(fontsize=8, frameon=False, loc="upper right")


# ============================================================
# Panel (h): Flory–Schulz Distribution
# ============================================================
N1_flory = 200
b_values_flory = [0.025, 0.02, 0.01]

Nmin = 1
Nmax_flory = 800
Nmulti_flory = np.arange(Nmin, Nmax_flory + 1)

curves_flory = [
    bval**2 * Nmulti_flory * (1 - bval) ** (Nmulti_flory - 1)
    for bval in b_values_flory
]

for k in range(3):
    base_dist = curves_flory[k]
    phiAmulti = compute_phaseA_distribution(Nmulti_flory, base_dist, N1_flory)

    # Parent distribution
    ax5.plot(
        Nmulti_flory,
        base_dist / np.sum(base_dist),
        label=rf"$b$ = {1 - b_values_flory[k]}",
        linewidth=3,
        color=colorsP[k],
        alpha=0.8,
    )

    # Phase A distribution
    if k == 2:
        ax5.plot(
            Nmulti_flory,
            phiAmulti / np.sum(phiAmulti),
            linewidth=3,
            color=colorsR[k],
            linestyle="dashed",
            dashes=(2, 0.25),
            alpha=0.8,
            label="Phase A",
        )
    else:
        ax5.plot(
            Nmulti_flory,
            phiAmulti / np.sum(phiAmulti),
            linewidth=3,
            color=colorsR[k],
            linestyle="dashed",
            dashes=(2, 0.25),
            alpha=0.8,
        )

ax5.set_xlim(100, Nmax_flory)
ax5.set_xticks([1, 200, 400, 600, 800])
ax5.set_yticks([0.0, 0.01])
ax5.set_ylim([0, 0.01])
ax5.set_title("Flory-Schulz", fontsize=12)
ax5.legend(fontsize=8, frameon=False, loc="upper center")


# ============================================================
# Panel (f): Modified Poisson Distribution
# ============================================================
N1_poisson = 200
lambda_values = [160, 200, 240]

Nmin = 1
Nmax_poisson = 450
Nmulti_poisson = np.arange(Nmin, Nmax_poisson + 1)

curves_poisson = [
    evaluate_mod_poisson(lam, Nmulti_poisson) for lam in lambda_values
]

for k in range(3):
    base_dist = curves_poisson[k]
    phiAmulti = compute_phaseA_distribution(Nmulti_poisson, base_dist, N1_poisson)

    # Parent distribution
    ax6.plot(
        Nmulti_poisson,
        base_dist / np.sum(base_dist),
        label=rf"$\lambda$ = {lambda_values[k]}",
        linewidth=3,
        color=colorsP[k],
        alpha=0.8,
    )

    # Phase A distribution
    if k == 2:
        ax6.plot(
            Nmulti_poisson,
            phiAmulti / np.sum(phiAmulti),
            linewidth=3,
            color=colorsR[k],
            linestyle="dashed",
            dashes=(2, 0.25),
            alpha=0.8,
            label="Phase A",
        )
    else:
        ax6.plot(
            Nmulti_poisson,
            phiAmulti / np.sum(phiAmulti),
            linewidth=3,
            color=colorsR[k],
            linestyle="dashed",
            dashes=(2, 0.25),
            alpha=0.8,
        )

ax6.set_xlim(1, 450)
ax6.set_xticks([1, 200, 400])
ax6.set_yticks([0.0, 0.02, 0.04])
ax6.set_ylim([0, 0.045])
ax6.set_title("Modified-Poisson", fontsize=12)
ax6.legend(fontsize=8, frameon=False, loc="upper right")


# ============================================================
# Panel (g): Bimodal Distribution
# ============================================================
N1_bimodal = 150
N2_bimodal = 250
std_dev_bimodal = 20
frac_vals = [0.25, 0.5, 0.75]

Nmin = 1
Nmax_bimodal = 450
Nmulti_bimodal = np.arange(Nmin, Nmax_bimodal + 1)

gaussian_curves_bimodal = [
    frac
    / (std_dev_bimodal * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti_bimodal - N1_bimodal) ** 2 / (2 * std_dev_bimodal**2))
    + (1 - frac)
    / (std_dev_bimodal * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti_bimodal - N2_bimodal) ** 2 / (2 * std_dev_bimodal**2))
    for frac in frac_vals
]

for k in range(3):
    base_dist = gaussian_curves_bimodal[k]
    phiAmulti = compute_phaseA_distribution(Nmulti_bimodal, base_dist, N1_bimodal)

    # Parent distribution
    ax7.plot(
        Nmulti_bimodal,
        base_dist / np.sum(base_dist),
        label=rf"$\eta$ = {frac_vals[k]}",
        linewidth=3,
        color=colorsP[k],
        alpha=0.8,
    )

    # Phase A distribution
    if k == 2:
        ax7.plot(
            Nmulti_bimodal,
            phiAmulti / np.sum(phiAmulti),
            linewidth=3,
            color=colorsR[k],
            linestyle="dashed",
            dashes=(2, 0.25),
            alpha=0.8,
            label="Phase A",
        )
    else:
        ax7.plot(
            Nmulti_bimodal,
            phiAmulti / np.sum(phiAmulti),
            linewidth=3,
            color=colorsR[k],
            linestyle="dashed",
            dashes=(2, 0.25),
            alpha=0.8,
        )

ax7.set_xlim(100, 450)
ax7.set_xticks([1, 200, 400])
ax7.set_yticks([0.0, 0.015, 0.03])
ax7.set_title("Bimodal", fontsize=12)
ax7.set_ylim(bottom=0)
ax7.legend(fontsize=8, frameon=False, loc="upper left")


# ============================================================
# Panel labels, layout, save
# ============================================================
ax4.annotate(
    "(a)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)
ax5.annotate(
    "(d)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)
ax6.annotate(
    "(b)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)
ax7.annotate(
    "(c)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)

plt.tight_layout()
plt.savefig("figure2ad_final.pdf", format="pdf")
plt.show()
