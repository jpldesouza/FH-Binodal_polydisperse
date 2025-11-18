import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

from SharedFunctions_finalized import (
    generate_laguerre_polynomials,
    compute_poly_binodal,
    find_y1,
    evaluate_mod_poisson,
)

# ============================================================
# Colormaps
# ============================================================

# Purple/teal-like colormap for χ
colors_p = [
    "#ffffd9",
    "#edf8b1",
    "#c7e9b4",
    "#7fcdbb",
    "#41b6c4",
    "#1d91c0",
    "#225ea8",
    "#253494",
    "#081d58",
]
custom_cmapP = LinearSegmentedColormap.from_list("custom_p", colors_p, N=256)

# Optional blue and red maps (currently only used in commented sections)
colors_b = ["#08519c", "#f7fbff"]
custom_cmapB = LinearSegmentedColormap.from_list("custom_b", colors_b, N=256)

colors_r = ["#a50f15", "#fff5f0"]
custom_cmapR = LinearSegmentedColormap.from_list("custom_r", colors_r, N=256)

# ============================================================
# Figure setup
# ============================================================

fig = plt.figure(figsize=(6, 5))

# Shared χ normalization (we saturate χ at vmax in each block)
vmin = 0.5
vmax = 1.5
saturate_value = vmax
norm = BoundaryNorm(boundaries=np.linspace(vmin, vmax, 257), ncolors=256, clip=True)

# Common number of Laguerre terms
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)


# ============================================================
# Helper: compute phibar from φ_A, φ_B, ν
# ============================================================

def compute_phibar(phiA_mat, phiB_mat, nu_vec, y1_vec):
    """Return phibar grid with same shape as phiA_mat/phiB_mat."""
    _, nu_grid = np.meshgrid(y1_vec, nu_vec)
    return nu_grid * phiA_mat + (1 - nu_grid) * phiB_mat


# ============================================================
# (a) Normal distribution
# ============================================================

nu_values = np.logspace(-20, 0, 100)
N1 = 200
std_devs = [20]
chi0 = np.array([0.8])

Nmin, Nmax = 1, 450
Nmulti = np.arange(Nmin, Nmax + 1)

gaussian_curves = [
    1.0 / (sd * np.sqrt(2 * np.pi)) * np.exp(-(Nmulti - N1) ** 2 / (2 * sd**2))
    for sd in std_devs
]

param = {"Nmulti": Nmulti}
logeps1vec = np.concatenate(
    [np.linspace(-2000, -110, 40), np.linspace(-100, -0.3, 100)]
)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros_like(phiAmat)
chi_mat = np.zeros_like(phiAmat)
phiAmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiBmultimat = np.zeros_like(phiAmultimat)

phiAat_chi0 = np.zeros_like(nu_values)
phiBat_chi0 = np.zeros_like(nu_values)

for k in range(1):  # only one std_dev
    umulti = gaussian_curves[k]

    for i, nu in enumerate(nu_values):
        param["nu"] = nu
        print("Normal, i =", i)

        chi_vec, phiA_all, phiB_all, phiAmulti_all, phiBmulti_all = compute_poly_binodal(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmat[i, :] = phiA_all
        phiBmat[i, :] = phiB_all
        chi_mat[i, :] = chi_vec

        # Evaluate at χ = χ0
        logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)
        _, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmultimat[i, :] = phiAmulti[:, 0]
        phiBmultimat[i, :] = phiBmulti[:, 0]
        phiAat_chi0[i] = phiA[0]
        phiBat_chi0[i] = phiB[0]

# Build phibar grid
phibar = compute_phibar(phiAmat, phiBmat, nu_values, y1vec)

# Saturate χ
chi_mat[chi_mat > saturate_value] = saturate_value

# Plot
ax = fig.add_subplot(2, 2, 1)
surfA = ax.pcolormesh(
    phiAmat, phibar, chi_mat, cmap=custom_cmapP, linewidth=1, edgecolors="face"
)
ax.pcolormesh(
    phiBmat, phibar, chi_mat, cmap=custom_cmapP, linewidth=1, edgecolors="face"
)
ax.contour(
    phiAmat,
    phibar,
    chi_mat,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=vmin,
    vmax=vmax,
)
ax.contour(
    phiBmat,
    phibar,
    chi_mat,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=vmin,
    vmax=vmax,
)

ax.set_title("Normal", fontsize=12)
ax.set_xlabel(r"$\phi_t$", fontsize=12)
ax.set_ylabel(r"$\bar{\phi}$", fontsize=12)
ax.set_facecolor("black")
ax.grid(False)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1e-10, 1])
ax.set_ylim([1e-10, 1])

cbar = fig.colorbar(surfA, ax=ax)
cbar.ax.set_title(r"$\chi$", fontsize=12)
ax.annotate("(a)", xy=(-0.35, 1.1), xycoords="axes fraction", fontsize=12,
            ha="left", va="top")


# ============================================================
# (d) Flory–Schulz distribution
# ============================================================

nu_values = np.concatenate([np.logspace(-20, -2, 100), np.logspace(-1.99, 0, 100)])
N1 = 200
b_values = [0.01]
chi0 = np.array([0.8])

Nmin, Nmax = 1, 800
Nmulti = np.arange(Nmin, Nmax + 1)

curves = [b**2 * Nmulti * (1 - b) ** (Nmulti - 1) for b in b_values]

param = {"Nmulti": Nmulti}
logeps1vec = np.concatenate(
    [
        np.linspace(-4000, -110, 40),
        np.linspace(-100, -1, 50),
        np.linspace(-0.99, -0.001, 100),
    ]
)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros_like(phiAmat)
chi_mat = np.zeros_like(phiAmat)
phiAmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiBmultimat = np.zeros_like(phiAmultimat)

phiAat_chi0 = np.zeros_like(nu_values)
phiBat_chi0 = np.zeros_like(nu_values)

for k in range(1):
    umulti = curves[k]
    N1 = 2 / b_values[k] - 1

    for i, nu in enumerate(nu_values):
        param["nu"] = nu
        print("Flory–Schulz, i =", i)

        chi_vec, phiA, phiB, _, _ = compute_poly_binodal(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmat[i, :] = phiA
        phiBmat[i, :] = phiB
        chi_mat[i, :] = chi_vec

        logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)
        _, phiA_chi0, phiB_chi0, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmultimat[i, :] = phiAmulti[:, 0]
        phiBmultimat[i, :] = phiBmulti[:, 0]
        phiAat_chi0[i] = phiA_chi0[0]
        phiBat_chi0[i] = phiB_chi0[0]

phibar = compute_phibar(phiAmat, phiBmat, nu_values, y1vec)
phibar_at_chi0 = phiAat_chi0 * nu_values + (1 - nu_values) * phiBat_chi0
Nmultimat, phibar_grid_at_chi0 = np.meshgrid(Nmulti, phibar_at_chi0)

chi_mat[chi_mat > saturate_value] = saturate_value

ax = fig.add_subplot(2, 2, 4)
surfA = ax.pcolormesh(
    phiAmat, phibar, chi_mat, cmap=custom_cmapP, linewidth=1, edgecolors="face"
)
ax.pcolormesh(
    phiBmat, phibar, chi_mat, cmap=custom_cmapP, linewidth=1, edgecolors="face"
)
ax.contour(
    phiAmat,
    phibar,
    chi_mat,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=vmin,
    vmax=vmax,
)
ax.contour(
    phiBmat,
    phibar,
    chi_mat,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=vmin,
    vmax=vmax,
)

ax.set_title("Flory-Schulz", fontsize=12)
ax.set_xlabel(r"$\phi_t$", fontsize=12)
ax.set_ylabel(r"$\bar{\phi}$", fontsize=12)
ax.set_facecolor("black")
ax.grid(False)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1e-10, 1])
ax.set_ylim([1e-10, 1])

cbar = fig.colorbar(surfA, ax=ax)
cbar.ax.set_title(r"$\chi$", fontsize=12)
ax.annotate("(d)", xy=(-0.35, 1.1), xycoords="axes fraction", fontsize=12,
            ha="left", va="top")


# ============================================================
# (b) Bimodal distribution
# ============================================================

nu_values = np.logspace(-20, 0, 100)
N1 = 150
N2 = 250
frac = 0.5
std_dev = 20
chi0 = np.array([0.8])

Nmin, Nmax = 1, 450
Nmulti = np.arange(Nmin, Nmax + 1)

gaussian_curves = [
    frac / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(Nmulti - N1) ** 2 / (2 * std_dev**2))
    + (1 - frac) / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N2) ** 2 / (2 * std_dev**2))
]

param = {"Nmulti": Nmulti}
logeps1vec = np.concatenate(
    [np.linspace(-4000, -110, 40), np.linspace(-100, -0.3, 100)]
)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros_like(phiAmat)
chi_mat = np.zeros_like(phiAmat)
phiAmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiBmultimat = np.zeros_like(phiAmultimat)

phiAat_chi0 = np.zeros_like(nu_values)
phiBat_chi0 = np.zeros_like(nu_values)

for k in range(1):
    umulti = gaussian_curves[k]

    for i, nu in enumerate(nu_values):
        param["nu"] = nu
        print("Bimodal, i =", i)

        chi_vec, phiA, phiB, _, _ = compute_poly_binodal(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmat[i, :] = phiA
        phiBmat[i, :] = phiB
        chi_mat[i, :] = chi_vec

        logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)
        _, phiA_chi0, phiB_chi0, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmultimat[i, :] = phiAmulti[:, 0]
        phiBmultimat[i, :] = phiBmulti[:, 0]
        phiAat_chi0[i] = phiA_chi0[0]
        phiBat_chi0[i] = phiB_chi0[0]

phibar = compute_phibar(phiAmat, phiBmat, nu_values, y1vec)
phibar_at_chi0 = phiAat_chi0 * nu_values + (1 - nu_values) * phiBat_chi0
Nmultimat, phibar_grid_at_chi0 = np.meshgrid(Nmulti, phibar_at_chi0)

chi_mat[chi_mat > saturate_value] = saturate_value

ax = fig.add_subplot(2, 2, 3)
surfA = ax.pcolormesh(
    phiAmat, phibar, chi_mat, cmap=custom_cmapP, linewidth=1, edgecolors="face"
)
ax.pcolormesh(
    phiBmat, phibar, chi_mat, cmap=custom_cmapP, linewidth=1, edgecolors="face"
)
ax.contour(
    phiAmat,
    phibar,
    chi_mat,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=vmin,
    vmax=vmax,
)
ax.contour(
    phiBmat,
    phibar,
    chi_mat,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=vmin,
    vmax=vmax,
)

ax.set_title("Bimodal", fontsize=12)
ax.set_xlabel(r"$\phi_t$", fontsize=12)
ax.set_ylabel(r"$\bar{\phi}$", fontsize=12)
ax.set_facecolor("black")
ax.grid(False)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1e-10, 1])
ax.set_ylim([1e-10, 1])

cbar = fig.colorbar(surfA, ax=ax)
cbar.ax.set_title(r"$\chi$", fontsize=12)
ax.annotate("(c)", xy=(-0.35, 1.1), xycoords="axes fraction", fontsize=12,
            ha="left", va="top")


# ============================================================
# (c) Modified-Poisson distribution
# ============================================================

nu_values = np.logspace(-20, 0, 100)
N1 = 200
b_values = [200]
chi0 = np.array([0.8])

Nmin, Nmax = 100, 300
Nmulti = np.arange(Nmin, Nmax + 1)

curves = [evaluate_mod_poisson(b, Nmulti) for b in b_values]

param = {"Nmulti": Nmulti}
logeps1vec = np.concatenate(
    [np.linspace(-4000, -110, 40), np.linspace(-100, -0.3, 100)]
)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros_like(phiAmat)
chi_mat = np.zeros_like(phiAmat)
phiAmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiBmultimat = np.zeros_like(phiAmultimat)

phiAat_chi0 = np.zeros_like(nu_values)
phiBat_chi0 = np.zeros_like(nu_values)

for k in range(1):
    umulti = curves[k]
    N1 = b_values[k]

    for i, nu in enumerate(nu_values):
        param["nu"] = nu
        print("Modified-Poisson, i =", i)

        chi_vec, phiA, phiB, _, _ = compute_poly_binodal(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmat[i, :] = phiA
        phiBmat[i, :] = phiB
        chi_mat[i, :] = chi_vec

        logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)
        _, phiA_chi0, phiB_chi0, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmultimat[i, :] = phiAmulti[:, 0]
        phiBmultimat[i, :] = phiBmulti[:, 0]
        phiAat_chi0[i] = phiA_chi0[0]
        phiBat_chi0[i] = phiB_chi0[0]

phibar = compute_phibar(phiAmat, phiBmat, nu_values, y1vec)
phibar_at_chi0 = phiAat_chi0 * nu_values + (1 - nu_values) * phiBat_chi0
Nmultimat, phibar_grid_at_chi0 = np.meshgrid(Nmulti, phibar_at_chi0)

chi_mat[chi_mat > saturate_value] = saturate_value

ax = fig.add_subplot(2, 2, 2)
surfA = ax.pcolormesh(
    phiAmat, phibar, chi_mat, cmap=custom_cmapP, linewidth=1, edgecolors="face"
)
ax.pcolormesh(
    phiBmat, phibar, chi_mat, cmap=custom_cmapP, linewidth=1, edgecolors="face"
)
ax.contour(
    phiAmat,
    phibar,
    chi_mat,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=vmin,
    vmax=vmax,
)
ax.contour(
    phiBmat,
    phibar,
    chi_mat,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=vmin,
    vmax=vmax,
)

ax.set_title("Modified-Poisson", fontsize=12)
ax.set_xlabel(r"$\phi_t$", fontsize=12)
ax.set_ylabel(r"$\bar{\phi}$", fontsize=12)
ax.set_facecolor("black")
ax.grid(False)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1e-10, 1])
ax.set_ylim([1e-10, 1])

cbar = fig.colorbar(surfA, ax=ax)
cbar.ax.set_title(r"$\chi$", fontsize=12)
ax.annotate("(b)", xy=(-0.35, 1.1), xycoords="axes fraction", fontsize=12,
            ha="left", va="top")


# ============================================================
# Finalize
# ============================================================

plt.tight_layout()
plt.savefig("figure4_final.pdf", format="pdf")
plt.show()
