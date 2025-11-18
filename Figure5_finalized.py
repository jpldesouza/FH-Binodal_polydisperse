from SharedFunctions_finalized import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# ============================================================
# Global style / constants
# ============================================================

fontsizeval = 12

# ------------------------------------------------------------
# Colormaps
# ------------------------------------------------------------
colors = [
    "#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb",
    "#41b6c4", "#1d91c0", "#225ea8", "#253494", "#081d58"
]
custom_cmapP = LinearSegmentedColormap.from_list("custom_purple", colors, N=256)

colors = ["#08519c", "#f7fbff"]
custom_cmapB = LinearSegmentedColormap.from_list("custom_blue", colors, N=256)

colors = ["#a50f15", "#fff5f0"]
custom_cmapR = LinearSegmentedColormap.from_list("custom_red", colors, N=256)

# ------------------------------------------------------------
# Helper: common chi saturation
# ------------------------------------------------------------
def saturate_chi(chi_mat, vmin=0.5, vmax=1.5):
    """
    Clip chi_mat from above at vmax and return (chi_mat, norm, vmin, vmax).
    """
    saturate_value = vmax
    chi_mat[chi_mat > saturate_value] = saturate_value
    norm = BoundaryNorm(
        boundaries=np.linspace(vmin, vmax, 257),
        ncolors=256,
        clip=True
    )
    return chi_mat, norm, vmin, vmax


# ============================================================
# Figure container
# ============================================================

fig = plt.figure(figsize=(12, 5))


# ============================================================
# 1. NORMAL DISTRIBUTION
# ============================================================

nu_values = np.logspace(-20, 0, 100)
N1 = 200
std_devs = [20]
chi0 = np.array([0.8])

# Chain-length axis
Nmin, Nmax = 1, 450
Nmulti = np.arange(Nmin, Nmax + 1)

# Gaussian distribution
gaussian_curves = [
    1.0 / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N1) ** 2 / (2 * std_dev**2))
    for std_dev in std_devs
]

# Parameters / y-grid
param = {"Nmulti": Nmulti}
logeps1vec = np.concatenate([
    np.linspace(-2000, -110, 40),
    np.linspace(-100, -0.3, 100),
])
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

# Storage
phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))
phiAmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiBmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiAat_chi0 = np.zeros_like(nu_values)
phiBat_chi0 = np.zeros_like(nu_values)

# Laguerre polynomials
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)

# Binodal computation over nu
for k in range(1):
    umulti = gaussian_curves[k]

    for i, nu in enumerate(nu_values):
        param["nu"] = nu
        print(i)

        # Full chi(y) curve at this nu
        chi_vec, phiA, phiB, _, _ = compute_poly_binodal(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )
        phiAmat[i, :] = phiA
        phiBmat[i, :] = phiB
        chi_mat[i, :] = chi_vec

        # Evaluate at chi0
        logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)
        _, phiA0, phiB0, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmultimat[i, :] = phiAmulti[:, 0]
        phiBmultimat[i, :] = phiBmulti[:, 0]
        phiAat_chi0[i] = phiA0[0]
        phiBat_chi0[i] = phiB0[0]

# Convert to phibar
y1mat, nu_values_2d = np.meshgrid(y1vec, nu_values)
phibar = nu_values_2d * phiAmat + (1 - nu_values_2d) * phiBmat

phibar_at_chi0 = phiAat_chi0 * nu_values + (1 - nu_values) * phiBat_chi0
Nmultimat, phibar_grid_at_chi0 = np.meshgrid(Nmulti, phibar_at_chi0)

# Saturate chi
chi_mat, norm, vmin, vmax = saturate_chi(chi_mat)

# Plot Normal p_B(N)
ax = fig.add_subplot(2, 4, 1)
norm_phiB = (phiBmultimat.T / np.sum(phiBmultimat, axis=1)).T
surf = ax.pcolormesh(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiB,
    cmap=custom_cmapB,
    linewidth=1,
    edgecolors="face",
)
ax.contour(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiB,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=0,
    vmax=0.05,
)
ax.set_title("Normal", fontsize=fontsizeval)
ax.set_xlabel("$N$", fontsize=fontsizeval)
ax.set_ylabel("$\\bar{\\phi}$", fontsize=fontsizeval)
ax.set_facecolor("black")
ax.grid(False)
ax.set_yscale("log")
ax.set_xlim([1, 450])
ax.set_ylim([1e-10, 1])
fig.colorbar(surf, ax=ax)
ax.annotate(
    "(a)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)

# Plot Normal p_A(N)
ax = fig.add_subplot(2, 4, 5)
norm_phiA = (phiAmultimat.T / np.sum(phiAmultimat, axis=1)).T
surf = ax.pcolormesh(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiA,
    cmap=custom_cmapR,
    linewidth=1,
    edgecolors="face",
)
ax.contour(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiA,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=0,
    vmax=0.05,
)
ax.set_xlabel("$N$", fontsize=fontsizeval)
ax.set_ylabel("$\\bar{\\phi}$", fontsize=fontsizeval)
ax.set_facecolor("black")
ax.grid(False)
ax.set_yscale("log")
ax.set_xlim([1, 450])
ax.set_ylim([1e-10, 1])
fig.colorbar(surf, ax=ax)
ax.annotate(
    "(e)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)


# ============================================================
# 2. FLORY–SCHULZ DISTRIBUTION
# ============================================================

nu_values = np.concatenate([
    np.logspace(-20, -2, 100),
    np.logspace(-1.99, 0, 100),
])
N1 = 200
b_values = [0.01]
chi0 = np.array([0.8])

Nmin, Nmax = 1, 800
Nmulti = np.arange(Nmin, Nmax + 1)

curves = [bval**2 * Nmulti * (1 - bval) ** (Nmulti - 1) for bval in b_values]

param = {"Nmulti": Nmulti}
logeps1vec = np.concatenate([
    np.linspace(-4000, -110, 40),
    np.linspace(-100, -1, 50),
    np.linspace(-0.99, -0.001, 100),
])
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))
phiAmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiBmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiAat_chi0 = np.zeros_like(nu_values)
phiBat_chi0 = np.zeros_like(nu_values)

laguerre_polynomials = generate_laguerre_polynomials(num_terms)

for k in range(1):
    umulti = curves[k]
    N1 = 2 / b_values[k] - 1

    for i, nu in enumerate(nu_values):
        param["nu"] = nu
        print(i)

        chi_vec, phiA, phiB, _, _ = compute_poly_binodal(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )
        phiAmat[i, :] = phiA
        phiBmat[i, :] = phiB
        chi_mat[i, :] = chi_vec

        logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)
        _, phiA0, phiB0, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmultimat[i, :] = phiAmulti[:, 0]
        phiBmultimat[i, :] = phiBmulti[:, 0]
        phiAat_chi0[i] = phiA0[0]
        phiBat_chi0[i] = phiB0[0]

y1mat, nu_values_2d = np.meshgrid(y1vec, nu_values)
phibar = nu_values_2d * phiAmat + (1 - nu_values_2d) * phiBmat
phibar_at_chi0 = phiAat_chi0 * nu_values + (1 - nu_values) * phiBat_chi0
Nmultimat, phibar_grid_at_chi0 = np.meshgrid(Nmulti, phibar_at_chi0)

chi_mat, norm, vmin, vmax = saturate_chi(chi_mat)

# Flory–Schulz p_B(N)
ax = fig.add_subplot(2, 4, 4)
norm_phiB = (phiBmultimat.T / np.sum(phiBmultimat, axis=1)).T
surf = ax.pcolormesh(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiB,
    cmap=custom_cmapB,
    linewidth=1,
    edgecolors="face",
)
ax.contour(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiB,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=0,
    vmax=0.05,
)
ax.set_title("Flory-Schulz", fontsize=fontsizeval)
ax.set_xlabel("$N_i$", fontsize=fontsizeval)
ax.set_ylabel("$\\bar{\\phi}$", fontsize=fontsizeval)
ax.set_facecolor("black")
ax.grid(False)
ax.set_yscale("log")
ax.set_xlim([1, 800])
ax.set_ylim([1e-10, 1])
fig.colorbar(surf, ax=ax)
ax.annotate(
    "(d)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)

# Flory–Schulz p_A(N)
ax = fig.add_subplot(2, 4, 8)
norm_phiA = (phiAmultimat.T / np.sum(phiAmultimat, axis=1)).T
surf = ax.pcolormesh(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiA,
    cmap=custom_cmapR,
    linewidth=1,
    edgecolors="face",
)
ax.contour(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiA,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=0,
    vmax=0.05,
)
ax.set_xlabel("$N_i$", fontsize=fontsizeval)
ax.set_ylabel("$\\bar{\\phi}$", fontsize=fontsizeval)
ax.set_facecolor("black")
ax.grid(False)
ax.set_yscale("log")
ax.set_xlim([1, 800])
ax.set_ylim([1e-10, 1])
fig.colorbar(surf, ax=ax)
ax.annotate(
    "(h)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)


# ============================================================
# 3. MODIFIED POISSON DISTRIBUTION
# ============================================================

nu_values = np.logspace(-20, 0, 100)
N1 = 200
b_values = [200]
chi0 = np.array([0.8])

Nmin, Nmax = 100, 300
Nmulti = np.arange(Nmin, Nmax + 1)

curves = [evaluate_mod_poisson(bval, Nmulti) for bval in b_values]

param = {"Nmulti": Nmulti}
logeps1vec = np.concatenate([
    np.linspace(-4000, -110, 40),
    np.linspace(-100, -0.3, 100),
])
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))
phiAmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiBmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiAat_chi0 = np.zeros_like(nu_values)
phiBat_chi0 = np.zeros_like(nu_values)

laguerre_polynomials = generate_laguerre_polynomials(num_terms)

for k in range(1):
    umulti = curves[k]
    N1 = b_values[k]

    for i, nu in enumerate(nu_values):
        param["nu"] = nu
        print(i)

        chi_vec, phiA, phiB, _, _ = compute_poly_binodal(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )
        phiAmat[i, :] = phiA
        phiBmat[i, :] = phiB
        chi_mat[i, :] = chi_vec

        logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)
        _, phiA0, phiB0, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmultimat[i, :] = phiAmulti[:, 0]
        phiBmultimat[i, :] = phiBmulti[:, 0]
        phiAat_chi0[i] = phiA0[0]
        phiBat_chi0[i] = phiB0[0]

y1mat, nu_values_2d = np.meshgrid(y1vec, nu_values)
phibar = nu_values_2d * phiAmat + (1 - nu_values_2d) * phiBmat
phibar_at_chi0 = phiAat_chi0 * nu_values + (1 - nu_values) * phiBat_chi0
Nmultimat, phibar_grid_at_chi0 = np.meshgrid(Nmulti, phibar_at_chi0)

chi_mat, norm, vmin, vmax = saturate_chi(chi_mat)

# Modified-Poisson p_B(N)
ax = fig.add_subplot(2, 4, 2)
norm_phiB = (phiBmultimat.T / np.sum(phiBmultimat, axis=1)).T
surf = ax.pcolormesh(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiB,
    cmap=custom_cmapB,
    linewidth=1,
    edgecolors="face",
)
ax.contour(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiB,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=0,
    vmax=0.05,
)
ax.set_title("Modified-Poisson", fontsize=fontsizeval)
ax.set_xlabel("$N_i$", fontsize=fontsizeval)
ax.set_ylabel("$\\bar{\\phi}$", fontsize=fontsizeval)
ax.set_facecolor("black")
ax.grid(False)
ax.set_yscale("log")
ax.set_xlim([100, 300])
ax.set_ylim([1e-10, 1])
fig.colorbar(surf, ax=ax)
ax.annotate(
    "(b)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)

# Modified-Poisson p_A(N)
ax = fig.add_subplot(2, 4, 6)
norm_phiA = (phiAmultimat.T / np.sum(phiAmultimat, axis=1)).T
surf = ax.pcolormesh(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiA,
    cmap=custom_cmapR,
    linewidth=1,
    edgecolors="face",
)
ax.contour(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiA,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=0,
    vmax=0.05,
)
ax.set_xlabel("$N_i$", fontsize=fontsizeval)
ax.set_ylabel("$\\bar{\\phi}$", fontsize=fontsizeval)
ax.set_facecolor("black")
ax.grid(False)
ax.set_yscale("log")
ax.set_xlim([100, 300])
ax.set_ylim([1e-10, 1])
fig.colorbar(surf, ax=ax)
ax.annotate(
    "(f)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)


# ============================================================
# 4. BIMODAL DISTRIBUTION
# ============================================================

nu_values = np.logspace(-20, 0, 100)
N1 = 150
N2 = 250
frac = 0.5
std_dev = 20.0
chi0 = np.array([0.8])

Nmin, Nmax = 1, 450
Nmulti = np.arange(Nmin, Nmax + 1)

gaussian_curves = [
    frac / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N1) ** 2 / (2 * std_dev**2))
    + (1 - frac) / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N2) ** 2 / (2 * std_dev**2))
]

param = {"Nmulti": Nmulti}
logeps1vec = np.concatenate([
    np.linspace(-4000, -110, 40),
    np.linspace(-100, -0.3, 100),
])
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))
phiAmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiBmultimat = np.zeros((len(nu_values), len(Nmulti)))
phiAat_chi0 = np.zeros_like(nu_values)
phiBat_chi0 = np.zeros_like(nu_values)

laguerre_polynomials = generate_laguerre_polynomials(num_terms)

for k in range(1):
    umulti = gaussian_curves[k]

    for i, nu in enumerate(nu_values):
        param["nu"] = nu
        print(i)

        chi_vec, phiA, phiB, _, _ = compute_poly_binodal(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )
        phiAmat[i, :] = phiA
        phiBmat[i, :] = phiB
        chi_mat[i, :] = chi_vec

        logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)
        _, phiA0, phiB0, phiAmulti, phiBmulti = compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )

        phiAmultimat[i, :] = phiAmulti[:, 0]
        phiBmultimat[i, :] = phiBmulti[:, 0]
        phiAat_chi0[i] = phiA0[0]
        phiBat_chi0[i] = phiB0[0]

y1mat, nu_values_2d = np.meshgrid(y1vec, nu_values)
phibar = nu_values_2d * phiAmat + (1 - nu_values_2d) * phiBmat
phibar_at_chi0 = phiAat_chi0 * nu_values + (1 - nu_values) * phiBat_chi0
Nmultimat, phibar_grid_at_chi0 = np.meshgrid(Nmulti, phibar_at_chi0)

chi_mat, norm, vmin, vmax = saturate_chi(chi_mat)

# Bimodal p_B(N)
ax = fig.add_subplot(2, 4, 3)
norm_phiB = (phiBmultimat.T / np.sum(phiBmultimat, axis=1)).T
surf = ax.pcolormesh(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiB,
    cmap=custom_cmapB,
    linewidth=1,
    edgecolors="face",
)
ax.contour(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiB,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=0,
    vmax=0.05,
)
ax.set_title("Bimodal", fontsize=fontsizeval)
ax.set_xlabel("$N_i$", fontsize=fontsizeval)
ax.set_ylabel("$\\bar{\\phi}$", fontsize=fontsizeval)
ax.set_facecolor("black")
ax.grid(False)
ax.set_yscale("log")
ax.set_xlim([1, 450])
ax.set_ylim([1e-10, 1])
fig.colorbar(surf, ax=ax)
ax.annotate(
    "(c)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)

# Bimodal p_A(N)
ax = fig.add_subplot(2, 4, 7)
norm_phiA = (phiAmultimat.T / np.sum(phiAmultimat, axis=1)).T
surf = ax.pcolormesh(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiA,
    cmap=custom_cmapR,
    linewidth=1,
    edgecolors="face",
)
ax.contour(
    Nmultimat,
    phibar_grid_at_chi0,
    norm_phiA,
    levels=10,
    colors="black",
    linewidths=0.5,
    vmin=0,
    vmax=0.05,
)
ax.set_xlabel("$N_i$", fontsize=fontsizeval)
ax.set_ylabel("$\\bar{\\phi}$", fontsize=fontsizeval)
ax.set_facecolor("black")
ax.grid(False)
ax.set_yscale("log")
ax.set_xlim([1, 450])
ax.set_ylim([1e-10, 1])
fig.colorbar(surf, ax=ax)
ax.annotate(
    "(g)", xy=(-0.35, 1.1), xycoords="axes fraction",
    fontsize=12, ha="left", va="top"
)


# ============================================================
# Finalize
# ============================================================

plt.tight_layout()
plt.savefig("figure5_final.pdf", format="pdf")
plt.show()
