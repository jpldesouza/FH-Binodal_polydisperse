import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from shapely.geometry import LineString  # currently only used in commented block
import ternary  # imported but not used in the active code; kept for completeness

from SharedFunctions_finalized import (
    generate_laguerre_polynomials,
    compute_poly_binodal,
    approx_poly_binodal0,
    hinv,                     # used inside compute_poly_general_binodal
)

# --------------------------------------------------------------------------------------
# Global styles / colors
# --------------------------------------------------------------------------------------
COLOR_SHADOW = np.array([37, 52, 148]) / 255.0
COLOR_CLOUD = np.array([29, 145, 192]) / 255.0
COLOR_STABLE = "#cb181d"
COLOR_META = "#fcae91"
COLOR_THREE_PHASE = "#fb6a4a"

CHI_TARGET = 0.76555  # chival used throughout


# --------------------------------------------------------------------------------------
# Helper: general binodal for arbitrary w(N)
# --------------------------------------------------------------------------------------
def compute_poly_general_binodal(logeps1vec, Nmulti, wmulti, N1, laguerre_polynomials):
    """
    Generalized binodal computation for a distribution wmulti(N), using the 'z' variable
    and Laguerre-based inverse h-function.
    """
    eps1vec = np.exp(logeps1vec)
    y1vec = 1 - eps1vec

    Nmultimat = np.tile(Nmulti, (len(y1vec), 1)).T
    logeps1mat = np.tile(logeps1vec, (len(Nmulti), 1))
    eps1mat = np.exp(logeps1mat)
    y1mat = 1 - eps1mat

    atanhy1vec = 0.5 * np.log(2 - eps1vec) - 0.5 * logeps1vec
    atanhy1mat = 0.5 * np.log(2 - eps1mat) - 0.5 * logeps1mat

    ymulti = np.tanh((Nmultimat / N1) * atanhy1mat)
    atanhymulti = (Nmultimat / N1) * atanhy1mat

    cond1 = np.abs(y1mat) < 0.01
    cond2 = ~cond1

    I1 = np.sum(np.sort(wmulti, axis=0), axis=0)
    I2 = np.sum(np.sort(wmulti * (atanhymulti / ymulti - 1) / Nmultimat, axis=0), axis=0)
    I3 = np.sum(np.sort(wmulti / ymulti, axis=0), axis=0)

    hz = 1 + I2 / I1
    logepsZvec = hinv(hz, laguerre_polynomials)
    epsZvec = np.exp(logepsZvec)
    zvec = 1 - epsZvec

    beta_multi_times_ymulti = 2 * wmulti / (I1 / zvec + I3)

    phiAmulti = np.zeros_like(ymulti)
    phiBmulti = np.zeros_like(ymulti)

    # General expressions
    phiAmulti[cond2] = beta_multi_times_ymulti[cond2] / (1 - np.exp(-2 * atanhymulti[cond2]))
    phiBmulti[cond2] = (
        beta_multi_times_ymulti[cond2]
        * np.exp(-2 * atanhymulti[cond2])
        / (-np.exp(-2 * atanhymulti[cond2]) + 1)
    )

    # Small y limit (series expansion)
    phiAmulti[cond1] = (
        0.5 * beta_multi_times_ymulti[cond1] / ymulti[cond1] * (1 + ymulti[cond1])
    )
    phiBmulti[cond1] = (
        0.5 * beta_multi_times_ymulti[cond1] / ymulti[cond1] * (1 - ymulti[cond1])
    )

    phiA = np.sum(phiAmulti, axis=0)
    phiB = np.sum(phiBmulti, axis=0)

    chi_vec = (1 / (2 * zvec) + I3 / (2 * I1)) * (
        (1 / N1) * atanhy1vec + np.arctanh(zvec)
    )

    return chi_vec, phiA, phiB, phiAmulti, phiBmulti, zvec


# --------------------------------------------------------------------------------------
# Figure and axes
# --------------------------------------------------------------------------------------
plt.figure(figsize=(10, 10))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

# --------------------------------------------------------------------------------------
# (a) Two polymers cloud/shadow curves (N1=20, N2=1000)
# --------------------------------------------------------------------------------------
eps = 0.0
nu_values = [0 + eps, 1 - eps]  # shadow, cloud
N1, N2 = 20, 1000
Nmulti = np.array([N1, N2])
curves = [np.array([0.99, 0.01])]

param = {"Nmulti": Nmulti}
alphamat = np.ones((len(Nmulti), len(Nmulti)))

logeps1vec = np.concatenate(
    [np.linspace(-100, -10, 100), np.linspace(-9.99, -1e-3, 10000)]
)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros_like(phiAmat)
chi_mat = np.zeros_like(phiAmat)

phiAmat_app0 = np.zeros_like(phiAmat)
phiBmat_app0 = np.zeros_like(phiAmat)
chi_mat_app0 = np.zeros_like(phiAmat)

num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)

for idx, nu in enumerate(nu_values):
    umulti = curves[0]
    i = 0
    param["nu"] = nu
    print("Two-polymer nu =", nu)

    chi_vec, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
        logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
    )
    chi_vec_app0, phiA_app0, phiB_app0, phiAmulti_app0, phiBmulti_app0 = (
        approx_poly_binodal0(logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials)
    )

    phiAmat[i, :] = phiA
    phiBmat[i, :] = phiB
    chi_mat[i, :] = chi_vec

    phiAmat_app0[i, :] = phiA_app0
    phiBmat_app0[i, :] = phiB_app0
    chi_mat_app0[i, :] = chi_vec_app0

    # Cloud vs shadow styling
    if idx == 1:  # nu ~ 1, cloud curve
        ax1.plot(
            phiA,
            chi_vec,
            label="Cloud curve",
            linewidth=3,
            color=COLOR_CLOUD,
            linestyle="dashed",
            dashes=(2, 0),
        )
        ax1.plot(
            phiB,
            chi_vec,
            linewidth=3,
            color=COLOR_SHADOW,
            linestyle="dashed",
            dashes=(2, 0.5),
        )
    else:  # shadow
        ax1.plot(
            phiA,
            chi_vec,
            label="Shadow curve",
            linewidth=3,
            color=COLOR_SHADOW,
            linestyle="dashed",
            dashes=(2, 0.5),
        )
        ax1.plot(
            phiB,
            chi_vec,
            linewidth=3,
            color=COLOR_CLOUD,
            linestyle="dashed",
            dashes=(2, 0),
        )

    ax1.set_ylim(0.5, 1.2)
    ax1.set_xlim(-0.01, 1.0)
    ax1.legend(fontsize=8, frameon=False, loc="lower right")

    # Reference line at chi = 0.76555
    ax1.plot(
        [0, 1],
        [CHI_TARGET, CHI_TARGET],
        linewidth=2,
        color="red",
        linestyle="dashed",
        dashes=(1, 0.5),
    )

    if i == 0:
        ax1.set_title(r"Two polymers, $N_1=20,\,N_2=1000$", fontsize=12)
        ax1.set_xlabel(r"$\phi_t$", fontsize=12)
        ax1.set_ylabel(r"$\chi$", fontsize=12)

    ax1.annotate(
        "(a)",
        xy=(-0.35, 1.1),
        xycoords="axes fraction",
        fontsize=12,
        ha="left",
        va="top",
    )

# Inset for panel (a)
x1, x2 = 0.2, 0.4
y1_min, y1_max = 0.74, 0.87
yticks = [0.75, 0.85]
xticks = [0.25, 0.35]

axins1 = inset_axes(
    ax1,
    width="30%",
    height="30%",
    loc="upper center",
    bbox_to_anchor=(-0.15, 0, 1, 1),
    bbox_transform=ax1.transAxes,
)
axins1.set_yticks(yticks)
axins1.set_xticks(xticks)
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1_min, y1_max)

# Copy lines to inset, preserving dash pattern
for line in ax1.lines:
    axins1.plot(
        line.get_xdata(),
        line.get_ydata(),
        color=line.get_color(),
        linestyle="dashed",
        linewidth=3,
        dashes=line._unscaled_dash_pattern[1],
    )

mark_inset(ax1, axins1, loc1=3, loc2=4, fc="none", ec="black", linestyle="--")


# --------------------------------------------------------------------------------------
# (b) Bimodal cloud/shadow curves (N1=20, N2=1000, frac=0.99)
# --------------------------------------------------------------------------------------
Nmin, Nmax = 1, 1100
Nmulti = np.arange(Nmin, Nmax + 1)

N1, N2 = 20, 1000
std_dev = 20.0
frac_vals = [0.99]

gaussian_curves = [
    frac / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N1) ** 2 / (2 * std_dev**2))
    + (1 - frac)
    / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N2) ** 2 / (2 * std_dev**2))
    for frac in frac_vals
]

param = {"Nmulti": Nmulti}
alphamat = np.ones((len(Nmulti), len(Nmulti)))

eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros_like(phiAmat)
chi_mat = np.zeros_like(phiAmat)

phiAmat_app0 = np.zeros_like(phiAmat)
phiBmat_app0 = np.zeros_like(phiAmat)
chi_mat_app0 = np.zeros_like(phiAmat)

for idx, nu in enumerate(nu_values):
    umulti = gaussian_curves[0]
    i = 0
    param["nu"] = nu

    chi_vec, phiA, phiB, phiAmulti, phiBmulti = compute_poly_binodal(
        logeps1vec, Nmulti, umulti, nu, N1, laguerre_polynomials
    )
    chi_vec_app0, phiA_app0, phiB_app0, phiAmulti_app0, phiBmulti_app0 = (
        approx_poly_binodal0(
            logeps1vec,
            Nmulti,
            umulti,
            nu,
            N1 * frac_vals[0] + (1 - frac_vals[0]) * N2,
            laguerre_polynomials,
        )
    )

    phiAmat[i, :] = phiA
    phiBmat[i, :] = phiB
    chi_mat[i, :] = chi_vec

    phiAmat_app0[i, :] = phiA_app0
    phiBmat_app0[i, :] = phiB_app0
    chi_mat_app0[i, :] = chi_vec_app0

    if idx == 1:
        ax2.plot(
            phiA,
            chi_vec,
            label="Cloud curve",
            linewidth=3,
            color=COLOR_CLOUD,
            linestyle="dashed",
            dashes=(2, 0),
        )
        ax2.plot(
            phiB,
            chi_vec,
            linewidth=3,
            color=COLOR_SHADOW,
            linestyle="dashed",
            dashes=(2, 0.5),
        )
    else:
        ax2.plot(
            phiA,
            chi_vec,
            label="Shadow curve",
            linewidth=3,
            color=COLOR_SHADOW,
            linestyle="dashed",
            dashes=(2, 0.5),
        )
        ax2.plot(
            phiB,
            chi_vec,
            linewidth=3,
            color=COLOR_CLOUD,
            linestyle="dashed",
            dashes=(2, 0),
        )

    ax2.set_ylim(0.5, 1.2)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")

    if i == 0:
        ax2.set_title(rf"Bimodal, $f$={frac_vals[0]}", fontsize=12)
        ax2.set_xlabel(r"$\phi_t$", fontsize=12)
        ax2.set_ylabel(r"$\chi$", fontsize=12)

    ax2.annotate(
        "(b)",
        xy=(-0.35, 1.1),
        xycoords="axes fraction",
        fontsize=12,
        ha="left",
        va="top",
    )

    # Inset for panel (b) (constructed inside loop, but uses same lines)
    x1, x2 = 0.18, 0.38
    y1_min, y1_max = 0.69, 0.82
    yticks = [0.7, 0.8]
    xticks = [0.22, 0.32]

    axins2 = inset_axes(
        ax2,
        width="30%",
        height="30%",
        loc="upper center",
        bbox_to_anchor=(-0.12, 0, 1, 1),
        bbox_transform=ax2.transAxes,
    )
    axins2.set_yticks(yticks)
    axins2.set_xticks(xticks)
    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1_min, y1_max)

    for line in ax2.lines:
        axins2.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle="dashed",
            linewidth=3,
            dashes=line._unscaled_dash_pattern[1],
        )

    mark_inset(ax2, axins2, loc1=3, loc2=4, fc="none", ec="black", linestyle="--")


# --------------------------------------------------------------------------------------
# (c) Composite-space phase diagram in (1−y1, 1−tanh(w2 y1))
# --------------------------------------------------------------------------------------
N1, N2 = 20, 1000
Nmulti = np.array([N1, N2])
param = {"Nmulti": Nmulti}

# Carefully constructed logeps1vec and epstanhw2y1vec grids
logeps1vec = np.flip(
    np.log(
        np.concatenate(
            [
                1 - np.logspace(-6, -1, 500),
                1 - np.linspace(0.101, 0.899, 200),
                np.flip(np.logspace(-200, -1, 200)),
            ]
        )
    )
)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

epstanhw2y1vec = np.concatenate(
    [
        np.logspace(-100, -11, 100),
        np.logspace(-10.9, -1, 200),
        np.linspace(0.101, 0.899, 200),
        np.flip(1 - np.logspace(-10, -1, 400)),
    ]
)
Nw2 = len(epstanhw2y1vec)

logeps1mat = np.tile(logeps1vec, (len(Nmulti), 1))
eps1mat = np.exp(logeps1mat)

Y1, tanhW2Y1 = np.meshgrid(y1vec, 1 - epstanhw2y1vec)
EPS1, epstanhW2Y1 = np.meshgrid(eps1vec, epstanhw2y1vec)

chi_mat = np.zeros_like(Y1)
zmat = np.zeros_like(Y1)
phiA1mat = np.zeros_like(Y1)
phiA2mat = np.zeros_like(Y1)
phiB1mat = np.zeros_like(Y1)
phiB2mat = np.zeros_like(Y1)

for j in range(Nw2):
    wmat = np.ones((len(Nmulti), len(logeps1vec)))

    if epstanhw2y1vec[j] > 0.01:
        wmat[1, :] *= np.arctanh(1 - epstanhw2y1vec[j]) / y1vec
    else:
        wmat[1, :] *= (
            0.5 * np.log(2 / epstanhw2y1vec[j]) - 0.25 * epstanhw2y1vec[j]
        ) / y1vec

    chi_vec, phiA, phiB, phiAmulti, phiBmulti, zvec = compute_poly_general_binodal(
        logeps1vec, Nmulti, wmat, N1, laguerre_polynomials
    )

    # Remove unphysical points
    invalid = (
        (np.min(phiAmulti, axis=0) < 0)
        | (phiA > 1)
        | (np.min(phiBmulti, axis=0) < 0)
        | (phiB > 1)
    )
    chi_vec[invalid] = np.nan

    chi_mat[j, :] = chi_vec
    zmat[j, :] = zvec
    phiA1mat[j, :] = phiAmulti[0, :]
    phiA2mat[j, :] = phiAmulti[1, :]
    phiB1mat[j, :] = phiBmulti[0, :]
    phiB2mat[j, :] = phiBmulti[1, :]

# Clamp χ values
minchi = np.nanmin(chi_mat)
maxchi = 1.0
chi_mat_edit = chi_mat.copy()
chi_mat_edit[chi_mat_edit > maxchi] = maxchi
chi_mat_edit[chi_mat_edit < minchi] = minchi

# Filled contour of χ
contour = ax3.contourf(
    EPS1, epstanhW2Y1, chi_mat_edit, cmap="winter", levels=20, vmin=minchi, vmax=maxchi
)
contourR = ax3.contour(
    EPS1,
    epstanhW2Y1,
    chi_mat_edit,
    levels=[CHI_TARGET],
    colors="red",
    linewidths=1.5,
)

ax3.set_title("Composite Space Phase Diagram")
ax3.set_xlabel(r"$1-y_1$")
ax3.set_ylabel(r"$1-\tanh(w_2 y_1)$")
ax3.annotate(
    "(c)",
    xy=(-0.35, 1.1),
    xycoords="axes fraction",
    fontsize=12,
    ha="left",
    va="top",
)

cbar = plt.colorbar(contour, ax=ax3)
cbar.ax.set_title(r"$\chi$", fontsize=12)
cbar.set_ticks([0.525, 0.6, 0.7, 0.8, 0.9, 1.0])
cbar.set_ticklabels(
    [r"$0.525$", r"$0.6$", r"$0.7$", r"$0.8$", r"$0.9$", r"$1.0$"]
)

# Extract the χ=χ_target contour and reparameterize
contour_paths = contourR.collections[0].get_paths()
eps1values = np.array([p.vertices[:, 0] for p in contour_paths])[0]
epstanhy1w2values = np.array([p.vertices[:, 1] for p in contour_paths])[0]

tanhy1w2values = 1 - epstanhy1w2values
y1values = 1 - eps1values

logeps1values = np.log(eps1values)
wmat = np.ones((len(Nmulti), len(logeps1values)))
indices_thresh = epstanhy1w2values < 0.01

wmat[1, ~indices_thresh] *= np.arctanh(tanhy1w2values[~indices_thresh]) / y1values[
    ~indices_thresh
]
wmat[1, indices_thresh] *= (
    0.5 * np.log(2 / epstanhy1w2values[indices_thresh])
    - 0.25 * epstanhy1w2values[indices_thresh]
) / y1values[indices_thresh]

chi_vec, phiA, phiB, phiAmulti, phiBmulti, zvec = compute_poly_general_binodal(
    logeps1values, Nmulti, wmat, N1, laguerre_polynomials
)

chosen_index = np.abs(chi_vec - CHI_TARGET) < 0.0001

chi_vec = chi_vec[chosen_index]
phiA = phiA[chosen_index]
phiB = phiB[chosen_index]
phiAmulti = phiAmulti[:, chosen_index]
phiBmulti = phiBmulti[:, chosen_index]
y1values = y1values[chosen_index]
tanhy1w2values = tanhy1w2values[chosen_index]
zvalues = zvec[chosen_index]
eps1values = eps1values[chosen_index]
epstanhy1w2values = epstanhy1w2values[chosen_index]

maxindex = 2462
condINT1 = np.arange(0, 539)
condINT2 = np.arange(538, 583)
condINT3 = np.arange(582, 936)
condINT4 = np.arange(935, 1838)
condINT5 = np.arange(1837, maxindex)

# Color segments on panel (c)
ax3.plot(
    eps1values[condINT1],
    epstanhy1w2values[condINT1],
    color=COLOR_STABLE,
    linestyle="-",
    linewidth=2,
)
ax3.plot(
    eps1values[condINT2],
    epstanhy1w2values[condINT2],
    color=COLOR_META,
    linestyle="-",
    linewidth=2,
)
ax3.plot(
    eps1values[condINT3],
    epstanhy1w2values[condINT3],
    color=COLOR_STABLE,
    linestyle="-",
    linewidth=2,
)
ax3.plot(
    eps1values[condINT4],
    epstanhy1w2values[condINT4],
    color=COLOR_META,
    linestyle="-",
    linewidth=2,
)
ax3.plot(
    eps1values[condINT5],
    epstanhy1w2values[condINT5],
    color=COLOR_STABLE,
    linestyle="-",
    linewidth=2,
)

# Three-phase points
TL1y1, TL3y1, TL2y1 = [0.021278655349619962, 0.5131125908173945, 0.5286195945443225]
TL1tanhw2y1, TL2tanhw2y1, TL3tanhw2y1 = [
    0.03639025581086912,
    0.06179949050078531,
    0.007629271499321987,
]

ax3.plot(1 - TL1y1, 1 - TL1tanhw2y1, color=COLOR_THREE_PHASE, marker="o", markersize=5)
ax3.plot(1 - TL2y1, 1 - TL2tanhw2y1, color=COLOR_THREE_PHASE, marker="o", markersize=5)
ax3.plot(1 - TL3y1, 1 - TL3tanhw2y1, color=COLOR_THREE_PHASE, marker="o", markersize=5)


# --------------------------------------------------------------------------------------
# (d) Composition-space picture with tie-lines and classifications
# --------------------------------------------------------------------------------------

# Tie line 1
x1, x2 = 0.0252827, 0.3122872
y1_lin, y2_lin = 0.003018, 0.299274
ax4.plot([x1, y1_lin], [x2, y2_lin], color=COLOR_THREE_PHASE, linewidth=2)

TL1y1 = abs((x2 - y2_lin) / (x2 + y2_lin))
TL1tanhw2y1 = np.tanh((x1 - y1_lin) / (x2 + y2_lin))

# Tie line 2
x1, x2 = 0.0252827, 0.3122872
y1_lin, y2_lin = 7.1045e-28, 0.0963
ax4.plot([x1, y1_lin], [x2, y2_lin], color=COLOR_THREE_PHASE, linewidth=2)
TL2y1 = abs((x2 - y2_lin) / (x2 + y2_lin))
TL2tanhw2y1 = np.tanh((x1 - y1_lin) / (x2 + y2_lin))

# Tie line 3
x1, x2 = 0.003018, 0.299274
y1_lin, y2_lin = 7.1045e-28, 0.0963
ax4.plot([x1, y1_lin], [x2, y2_lin], color=COLOR_THREE_PHASE, linewidth=2)
TL3y1 = abs((x2 - y2_lin) / (x2 + y2_lin))
TL3tanhw2y1 = np.tanh((x1 - y1_lin) / (x2 + y2_lin))

# Phase path in (phi_1000, phi_20)
ax4.plot(
    phiAmulti[1, condINT1],
    phiAmulti[0, condINT1],
    color=COLOR_STABLE,
    linestyle="-",
    linewidth=2,
    label="Two phase",
)
ax4.plot(
    phiBmulti[1, condINT1],
    phiBmulti[0, condINT1],
    color=COLOR_STABLE,
    linestyle="-",
    linewidth=2,
)

ax4.plot(
    phiAmulti[1, condINT2],
    phiAmulti[0, condINT2],
    color=COLOR_META,
    linestyle="-",
    linewidth=2,
)
ax4.plot(
    phiBmulti[1, condINT2],
    phiBmulti[0, condINT2],
    color=COLOR_META,
    linestyle="-",
    linewidth=2,
)

ax4.plot(
    phiAmulti[1, condINT3],
    phiAmulti[0, condINT3],
    color=COLOR_STABLE,
    linestyle="-",
    linewidth=2,
)
ax4.plot(
    phiBmulti[1, condINT3],
    phiBmulti[0, condINT3],
    color=COLOR_STABLE,
    linestyle="-",
    linewidth=2,
)

ax4.plot(
    phiAmulti[1, condINT4],
    phiAmulti[0, condINT4],
    color=COLOR_META,
    linestyle="-",
    linewidth=2,
    label="Metastable",
)
ax4.plot(
    phiBmulti[1, condINT4],
    phiBmulti[0, condINT4],
    color=COLOR_META,
    linestyle="-",
    linewidth=2,
)

ax4.plot(
    phiAmulti[1, condINT5],
    phiAmulti[0, condINT5],
    color=COLOR_STABLE,
    linestyle="-",
    linewidth=2,
)
ax4.plot(
    phiBmulti[1, condINT5],
    phiBmulti[0, condINT5],
    color=COLOR_STABLE,
    linestyle="-",
    linewidth=2,
)

# Three-phase tie-lines + critical point
ax4.plot(
    1.108e-2,
    0.31013,
    marker="o",
    color="black",
    markersize=5,
    linewidth=0,
    label="Critical point",
)
ax4.plot(
    [x1, y1_lin],
    [x2, y2_lin],
    color=COLOR_THREE_PHASE,
    linewidth=2,
    label="Three phase",
)

ax4.set_xlim([1e-9, 0.25])
ax4.set_ylim([0.1, 0.35])
ax4.set_ylabel(r"$\phi_{20}$")
ax4.set_xlabel(r"$\phi_{1000}$")
ax4.legend(frameon=False)
ax4.annotate(
    "(d)",
    xy=(-0.35, 1.1),
    xycoords="axes fraction",
    fontsize=12,
    ha="left",
    va="top",
)

# Inset around the critical point
x1, x2 = 0.0, 0.04
y1_min, y1_max = 0.28, 0.32
yticks = [0.28, 0.32]

axins4 = inset_axes(
    ax4,
    width="30%",
    height="30%",
    loc="lower center",
    bbox_to_anchor=(0, 0.1, 1, 1),
    bbox_transform=ax4.transAxes,
)
axins4.set_yticks(yticks)
axins4.set_xlim(x1, x2)
axins4.set_ylim(y1_min, y1_max)

for line in ax4.lines:
    axins4.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), linewidth=2)
axins4.plot(1.108e-2, 0.31013, marker="o", color="black", markersize=5)

mark_inset(ax4, axins4, loc1=1, loc2=3, fc="none", ec="black", linestyle="--")


# --------------------------------------------------------------------------------------
# Finalize
# --------------------------------------------------------------------------------------
plt.tight_layout()
plt.savefig("Figure6_final.pdf", format="pdf")
plt.show()
