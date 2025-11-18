import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager

from SharedFunctions_finalized import *

# ============================================================
# Font setup (robust to missing file)
# ============================================================

# Adjust this path if you know where cmunrm.ttf actually lives
font_path = "../../Downloads/computer-modern/cmunrm.ttf"

if os.path.isfile(font_path):
    font_manager.fontManager.addfont(font_path)
    custom_font = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = custom_font.get_name()
else:
    print(
        f"Warning: font file not found at '{font_path}'. "
        "Using default Matplotlib font instead."
    )

# ============================================================
# Laguerre polynomials + h^{-1} test
# ============================================================

num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)

hz = np.array([1.01, 1.02, 1.07, 1.075, 1.076, 1.077, 5, 10])
logepsZvec = hinv(hz, laguerre_polynomials)
epsZvec = np.exp(logepsZvec)
zvec = 1 - epsZvec

hzcalc = np.arctanh(zvec) / zvec
print("Relative error (%) in h^{-1} inversion:", (hzcalc / hz - 1) * 100)

# ============================================================
# Random-walk polymer utilities
# ============================================================

def create_limited_angle_random_walk_polymers(
    num_polymers, num_steps, step_size, box_size, max_angle_degrees
):
    """Generate 2D polymers via a limited-angle random walk."""
    polymers = []
    max_angle = np.radians(max_angle_degrees)

    for _ in range(num_polymers):
        x = [np.random.uniform(-box_size / 2, box_size / 2)]
        y = [np.random.uniform(-box_size / 2, box_size / 2)]
        angle = np.random.uniform(0, 2 * np.pi)

        for _ in range(num_steps - 1):
            angle += np.random.uniform(-max_angle, max_angle)
            x.append(x[-1] + step_size * np.cos(angle))
            y.append(y[-1] + step_size * np.sin(angle))

        polymers.append((x, y))

    return polymers


def add_polymers(polymers, color, ax, line_width):
    """Plot polymers with a black outline and colored interior."""
    for x, y in polymers:
        ax.plot(
            x,
            y,
            color="black",
            solid_capstyle="round",
            lw=line_width + 2,  # outline
        )
        ax.plot(
            x,
            y,
            color=color,
            solid_capstyle="round",
            lw=line_width,      # colored line
        )

# ============================================================
# Parameters for polymer cartoon
# ============================================================

num_large_polymers = 50
num_small_polymers = 100
large_length = 50
small_length = 10
box_size = 150
line_width = 3
step_size = 1.0
max_angle_degrees = 30  # max change in angle per step (deg)

# ============================================================
# Figure + layout
# ============================================================

fig = plt.figure(figsize=(7, 3))
spec = fig.add_gridspec(3, 7)

ax2 = fig.add_subplot(spec[0:3, 0:3])  # polymer cartoon
ax3 = fig.add_subplot(spec[0:3, 3:5])  # MW distributions
ax1 = fig.add_subplot(spec[0:3, 5:7])  # phase diagram

# ============================================================
# Polymer fractionation panel (ax2)
# ============================================================

# Large polymers more compact, small polymers more extended
large_polymers = create_limited_angle_random_walk_polymers(
    num_large_polymers, large_length, step_size, box_size / 3, max_angle_degrees
)
small_polymers = create_limited_angle_random_walk_polymers(
    num_small_polymers, small_length, step_size, box_size, max_angle_degrees
)

for i in range(num_large_polymers):
    # “Background” small polymers
    add_polymers(
        small_polymers[i + num_large_polymers : num_large_polymers + i + 1],
        "#2c7fb8",
        ax2,
        line_width,
    )
    # Large polymer
    add_polymers(large_polymers[i : i + 1], "#a1dab4", ax2, line_width)
    # “Foreground” small polymer
    add_polymers(small_polymers[i : i + 1], "#2c7fb8", ax2, line_width)

ax2.set_xlim(-box_size / 2, box_size / 2)
ax2.set_ylim(-box_size / 2, box_size / 2)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_title("Polymer Fractionation")

# ============================================================
# Setup for thermodynamics panels
# ============================================================

# Binodal / MW-distribution parameters
nu_values = [0]               # only nu=0 here
N1 = 200
std_devs = [10, 40, 40]
chi0 = np.array([0.8])

# Chain-length axis
Nmin = 1
Nmax = 800
Nmulti = np.arange(Nmin, Nmax + 1)

# Gaussian MW distributions
gaussian_curves = [
    1 / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N1) ** 2 / (2 * std_dev**2))
    for std_dev in std_devs
]

# Colors
colorsB = ["#bdd7e7", "#6baed6", "#2171b5"]  # dilute
colorsP = ["#cbc9e2", "#9e9ac8", "#6a51a3"]  # overall
colorsR = ["#fcae91", "#fb6a4a", "#cb181d"]  # condensed

Nshift = 20

# Parameters for compute_poly_binodal
param = {"Nmulti": Nmulti}
alphamat = np.ones((len(Nmulti), len(Nmulti)))

logeps1vec = np.linspace(-100, -1e-2, 2000)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

# Storage (only one nu, but keep structure)
phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros_like(phiAmat)
phiBmat_app0 = np.zeros_like(phiBmat)
chi_mat_app0 = np.zeros_like(chi_mat)

# ============================================================
# Compute binodal + plot phase diagram and MW distributions
# ============================================================

k = 1  # use std_devs[1] = 40
umulti = gaussian_curves[k]

for i, nu in enumerate(nu_values):
    param["nu"] = nu

    # Full polydisperse binodal
    chi_vec, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
        logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
    )

    # Approximate monodisperse binodal
    chi_vec_app0, phiA_app0, phiB_app0, phiAmulti_app0, phiBmulti_app0 = (
        approx_poly_binodal0(
            logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
        )
    )

    phiAmat[i, :] = phiA
    phiBmat[i, :] = phiB
    chi_mat[i, :] = chi_vec

    phiAmat_app0[i, :] = phiA_app0
    phiBmat_app0[i, :] = phiB_app0
    chi_mat_app0[i, :] = chi_vec_app0

    # Find state at chi = chi0
    logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)

    chival, phiAval, phiBval, phiAmulti_chi0, phiBmulti_chi0 = (
        compute_poly_binodal(
            logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
        )
    )
    print("chi at chosen point:", chival)

    # ------------------------------------
    # MW distributions panel (ax3)
    # ------------------------------------
    ax3.plot(
        Nmulti,
        phiBmulti_chi0 / np.sum(phiBmulti_chi0),
        label="Dilute Phase",
        linewidth=5,
        color=colorsB[1],
        linestyle="dashed",
        dashes=(2, 0.1, 1, 0.1),
    )
    ax3.set_xticks([Nmin + Nshift, N1, Nmax - Nshift])
    ax3.set_yticks([0.0, 0.02, 0.04])

    ax3.plot(
        Nmulti,
        phiAmulti_chi0 / np.sum(phiAmulti_chi0),
        label="Concentrated Phase",
        linewidth=5,
        color=colorsR[1],
        linestyle="dashed",
        dashes=(2, 0.25),
    )
    ax3.set_xticks([Nmin + Nshift, N1, Nmax - Nshift])

    # Overall MW distribution
    ax3.plot(
        Nmulti,
        gaussian_curves[1] / np.sum(gaussian_curves[1]),
        label="Overall",
        linewidth=5,
        color=colorsP[1],
    )

    ax3.set_xlim([50, 500])
    ax3.set_xticks([50, 250, 450])
    ax3.set_yticks([])
    ax3.set_title("MW Distributions", fontsize=12)
    ax3.set_xlabel(r"$N_i$", fontsize=12)
    ax3.set_ylabel(r"$p(N_i)$", fontsize=12)
    ax3.set_ylim(bottom=0)

    # Text labels along curves
    ax3.annotate(
        "Dilute Phase",
        xy=(0.05, 0.2),
        xycoords="axes fraction",
        fontsize=12,
        rotation=82.5,
        color=colorsP[2],
    )
    ax3.annotate(
        "Condensed Phase",
        xy=(0.75, 0.2),
        xycoords="axes fraction",
        fontsize=12,
        rotation=-82.5,
        color=colorsR[2],
    )

    ax3.tick_params(axis="both", which="both", length=0)
    ax3.legend(fontsize=10, frameon=False, loc="upper left")
    ax3.legend().set_visible(False)

    # ------------------------------------
    # Phase diagram panel (ax1)
    # ------------------------------------
    # Monodisperse approximation
    mono_color = "#225ea8"
    poly_color = "#41b6c4"

    ax1.plot(
        phiA_app0,
        chi_vec_app0,
        linewidth=5,
        color=mono_color,
        linestyle="-",
    )
    ax1.plot(
        phiB_app0,
        chi_vec_app0,
        linewidth=5,
        color=mono_color,
        linestyle="-",
    )

    dashset = (2, 0.25)
    ax1.plot(
        phiA,
        chi_vec,
        label=rf"$N_\sigma$ = {std_devs[k]}",
        linewidth=4,
        color=poly_color,
        linestyle="dashed",
        dashes=dashset,
    )
    ax1.plot(
        phiB,
        chi_vec,
        linewidth=4,
        color=poly_color,
        linestyle="dashed",
        dashes=dashset,
    )

    ax1.set_xscale("log")
    ax1.set_xticks([1e-10, 1e-5, 1.0])
    ax1.set_yticks([0.6, 0.7, 0.8])
    ax1.set_ylim(0.55, 0.8)
    ax1.set_xlim(1e-10, 1)
    ax1.set_title("Phase diagram", fontsize=12)
    ax1.set_xlabel(r"$\phi_t$", fontsize=12)
    ax1.set_ylabel(r"$\chi$", fontsize=12)
    ax1.grid(False)

    ax1.annotate(
        "Monodisperse",
        xy=(0.3, 0.50),
        xycoords="axes fraction",
        fontsize=12,
        rotation=-60,
        color=mono_color,
    )
    ax1.annotate(
        "Polydisperse",
        xy=(0.2, 0.20),
        xycoords="axes fraction",
        fontsize=12,
        rotation=-60,
        color=poly_color,
    )

    plt.tick_params(axis="both", which="both", length=0)
    ax1.legend(fontsize=8, frameon=False, loc="lower left")
    ax1.legend().set_visible(False)

# Optionally, panel labels:
# ax1.annotate('(a)', xy=(-0.35, 1.1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
# ax2.annotate('(b)', xy=(-0.35, 1.1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
# ax3.annotate('(c)', xy=(-0.35, 1.1), xycoords='axes fraction', fontsize=12, ha='left', va='top')

plt.tight_layout()
plt.savefig("TOC.pdf", format="pdf")
plt.show()
