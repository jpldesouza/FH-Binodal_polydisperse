import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # noqa: F401

from SharedFunctions_finalized import *


# ============================================================
# Random-walk polymer generation
# ============================================================
def create_limited_angle_random_walk_polymers(
    num_polymers, num_steps, step_size, box_size, max_angle_degrees
):
    """Create polymers via a random walk with a maximum turning angle."""
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


def add_polymers(polymers, color, ax):
    """Add polymers to an axis with a black outline and colored center."""
    for x, y in polymers:
        # Black outline
        ax.plot(
            x,
            y,
            color="black",
            solid_capstyle="round",
            lw=line_width + 2,
        )
        # Colored line
        ax.plot(
            x,
            y,
            color=color,
            solid_capstyle="round",
            lw=line_width,
        )


# ============================================================
# Parameters
# ============================================================
num_large_polymers = 50
num_small_polymers = 100
large_length = 50
small_length = 10
box_size = 150
line_width = 3
step_size = 1.0          # Step size for the random walk
max_angle_degrees = 30   # Maximum change in angle per step (degrees)

# ============================================================
# First set of polymers (polydisperse mixture)
# ============================================================
large_polymers = create_limited_angle_random_walk_polymers(
    num_large_polymers, large_length, step_size, box_size, max_angle_degrees
)
small_polymers = create_limited_angle_random_walk_polymers(
    num_small_polymers, small_length, step_size, box_size, max_angle_degrees
)

# ============================================================
# Figure / subplots
# ============================================================
fig = plt.figure(figsize=(9, 3))
spec = fig.add_gridspec(1, 3)

ax1 = fig.add_subplot(spec[0:1, 0:1])
ax2 = fig.add_subplot(spec[0:1, 1:2])
ax3 = fig.add_subplot(spec[0:1, 2:3])

# ============================================================
# Panel (a): Polydisperse mixture
# ============================================================
for i in range(num_large_polymers):
    add_polymers(
        small_polymers[i + num_large_polymers : num_large_polymers + i + 1],
        "#2c7fb8",
        ax1,
    )
    add_polymers(large_polymers[i : i + 1], "#a1dab4", ax1)
    add_polymers(small_polymers[i : i + 1], "#2c7fb8", ax1)

ax1.set_xlim(-box_size / 2, box_size / 2)
ax1.set_ylim(-box_size / 2, box_size / 2)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_title("Polydisperse mixture")

# ============================================================
# Panel (b): Polymer fractionation
# ============================================================
large_polymers = create_limited_angle_random_walk_polymers(
    num_large_polymers, large_length, step_size, box_size / 3, max_angle_degrees
)
small_polymers = create_limited_angle_random_walk_polymers(
    num_small_polymers, small_length, step_size, box_size, max_angle_degrees
)

for i in range(num_large_polymers):
    add_polymers(
        small_polymers[i + num_large_polymers : num_large_polymers + i + 1],
        "#2c7fb8",
        ax2,
    )
    add_polymers(large_polymers[i : i + 1], "#a1dab4", ax2)
    add_polymers(small_polymers[i : i + 1], "#2c7fb8", ax2)

ax2.set_xlim(-box_size / 2, box_size / 2)
ax2.set_ylim(-box_size / 2, box_size / 2)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_title("Polymer Fractionation")

# ============================================================
# Panel (c): Molecular-weight distributions
# ============================================================
nu_values = [5e-9]
N1 = 200
std_devs = [10, 20, 40]
chi0 = np.array([0.8])

Nmin = 100
Nmax = 300
Nmulti = np.arange(Nmin, Nmax + 1)

# Gaussian curves (different std devs)
gaussian_curves = [
    1
    / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N1) ** 2 / (2 * std_dev**2))
    for std_dev in std_devs
]

# Colors
colorsB = ["#bdd7e7", "#6baed6", "#2171b5"]
colorsP = ["#cbc9e2", "#9e9ac8", "#6a51a3"]
colorsR = ["#fcae91", "#fb6a4a", "#cb181d"]

Nshift = 20

# Parameters for binodal computation
param = {"Nmulti": Nmulti}
alphamat = np.ones((len(Nmulti), len(Nmulti)))  # currently unused

logeps1vec = np.linspace(-100, -1e-6, 2000)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

# Initialize result matrices
phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

# Laguerre polynomials
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)

k = 1
umulti = gaussian_curves[k]

for i, nu in enumerate(nu_values):
    param["nu"] = nu

    chi_vec, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
        logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
    )
    phiAmat[i, :] = phiA
    phiBmat[i, :] = phiB
    chi_mat[i, :] = chi_vec

    logeps1vec_at_chi0 = find_y1(chi_vec, logeps1vec, chi0)

    chival, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
        logeps1vec_at_chi0, Nmulti, umulti, nu, N1, laguerre_polynomials
    )
    print(chival)

    # Dilute phase
    ax3.plot(
        Nmulti,
        phiBmulti / np.sum(phiBmulti),
        label="Dilute Phase",
        linewidth=5,
        color=colorsB[1],
        linestyle="dashed",
        dashes=(2, 0.1, 1, 0.1),
    )
    ax3.set_xticks([Nmin + Nshift, N1, Nmax - Nshift])
    ax3.set_yticks([0.0, 0.02, 0.04])

    # Concentrated phase
    ax3.plot(
        Nmulti,
        phiAmulti / np.sum(phiAmulti),
        label="Concentrated Phase",
        linewidth=5,
        color=colorsR[1],
        linestyle="dashed",
        dashes=(2, 0.25),
    )
    ax3.set_xticks([Nmin + Nshift, N1, Nmax - Nshift])

    # Overall distribution
    ax3.plot(
        Nmulti,
        gaussian_curves[1] / np.sum(gaussian_curves[1]),
        label="Overall",
        linewidth=5,
        color=colorsP[1],
    )

    ax3.set_xlim([Nmin + Nshift, Nmax - Nshift])
    ax3.set_xticks([Nmin + Nshift, N1, Nmax - Nshift])
    ax3.set_yticks([0.0, 0.02, 0.04])
    ax3.set_title("MW Distributions", fontsize=12)
    ax3.set_xlabel(r"$N$", fontsize=12)
    ax3.set_ylabel(r"$p(N)$", fontsize=12)
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=10, frameon=False, loc="upper left")

# ============================================================
# Panel labels and save
# ============================================================
ax1.annotate(
    "(a)",
    xy=(-0.35, 1.1),
    xycoords="axes fraction",
    fontsize=12,
    ha="left",
    va="top",
)
ax2.annotate(
    "(b)",
    xy=(-0.35, 1.1),
    xycoords="axes fraction",
    fontsize=12,
    ha="left",
    va="top",
)
ax3.annotate(
    "(c)",
    xy=(-0.35, 1.1),
    xycoords="axes fraction",
    fontsize=12,
    ha="left",
    va="top",
)

plt.tight_layout()
plt.savefig("figure1_final.pdf", format="pdf")
plt.show()
