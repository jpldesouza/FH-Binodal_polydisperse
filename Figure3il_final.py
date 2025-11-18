from SharedFunctions_finalized import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ============================================================
# Global settings & colors
# ============================================================
eps = 0
nu_values = [0 + eps, 1 - eps]
chi0 = np.array([0.8])

# Cloud / shadow colors
colorsP = ["#a1dab4", "#41b6c4", "#225ea8"]  # general palette (not always used)
colorshadow = np.array([37, 52, 148]) / 255
colorcloud = np.array([29, 145, 192]) / 255

# Figure and axes
plt.figure(figsize=(6, 5))
axNorm = plt.subplot(2, 2, 1)
axMod = plt.subplot(2, 2, 2)
axBim = plt.subplot(2, 2, 3)
axFS = plt.subplot(2, 2, 4)
plt.suptitle("Cloud and Shadow Curves", fontsize=16)

# ============================================================
# Helper: common Laguerre generator
# ============================================================
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)

# ============================================================
# (i) Normal distribution – cloud and shadow
# ============================================================
N1 = 200
std_devs = [40]

Nmin, Nmax = 1, 450
Nmulti = np.arange(Nmin, Nmax + 1)

gaussian_curves = [
    1
    / (std_dev * np.sqrt(2 * np.pi))
    * np.exp(-(Nmulti - N1) ** 2 / (2 * std_dev**2))
    for std_dev in std_devs
]

param = {"Nmulti": Nmulti}
alphamat = np.ones((len(Nmulti), len(Nmulti)))  # not used, but kept

logeps1vec = np.linspace(-50, -1e-2, 1000)
eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

for idx, nu in enumerate(nu_values):
    for k in range(1):  # only one std_dev
        umulti = gaussian_curves[k]
        i = 0
        param["nu"] = nu
        print(nu)

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

        # Cloud vs shadow assignment
        if idx == 1:
            # nu = 1 - eps: cloud curve labelled, shadow curve unlabeled
            axNorm.plot(
                phiA,
                chi_vec,
                label=r"Cloud curve",
                linewidth=3,
                color=colorcloud,
                linestyle="dashed",
                dashes=(2, 0),
            )
            axNorm.plot(
                phiB,
                chi_vec,
                linewidth=3,
                color=colorshadow,
                linestyle="dashed",
                dashes=(2, 0.5),
            )
        else:
            # nu = 0 + eps: shadow labelled, cloud unlabeled
            axNorm.plot(
                phiA,
                chi_vec,
                label=r"Shadow Curve ",
                linewidth=3,
                color=colorshadow,
                linestyle="dashed",
                dashes=(2, 0.5),
            )
            axNorm.plot(
                phiB,
                chi_vec,
                linewidth=3,
                color=colorcloud,
                linestyle="dashed",
                dashes=(2, 0),
            )

        # Axes formatting (no log scale here)
        axNorm.set_xticks([0, 0.075, 0.15])
        axNorm.set_yticks([0.57, 0.575, 0.58])
        axNorm.set_ylim(0.57, 0.58)
        axNorm.set_xlim(0, 0.15)
        axNorm.legend(fontsize=8, frameon=False, loc="lower left")

        if i == 0:
            labelstring = "(i)"
            axNorm.set_title(rf"Normal, $N_\sigma$ ={std_devs[k]}", fontsize=12)
            axNorm.set_xlabel(r"$\phi_t$", fontsize=12)
            axNorm.set_ylabel(r"$\chi$", fontsize=12)
            nustring = "$\\nu=$" + str(nu)  # not displayed, but kept

        axNorm.annotate(
            labelstring,
            xy=(-0.35, 1.1),
            xycoords="axes fraction",
            fontsize=12,
            ha="left",
            va="top",
        )

# ============================================================
# (l) Flory–Schulz distribution – cloud and shadow
# ============================================================
N1 = 200
b_values = [0.01]

Nmin, Nmax = 1, 800
Nmulti = np.arange(Nmin, Nmax + 1)

# Initial FS curves (then redefined below; kept to preserve original behavior)
curves = [bval**2 * Nmulti * (1 - bval) ** (Nmulti - 1) for bval in b_values]

# Re-define Nmulti for large range (as in original code)
Nmin, Nmax = 1, int(1e5)
Nmulti = np.arange(Nmin, Nmax + 1)

# Recompute curves on new Nmulti
curves = [bval**2 * Nmulti * (1 - bval) ** (Nmulti - 1) for bval in b_values]

param = {"Nmulti": Nmulti}

logeps1vec = -np.logspace(-300, -1e-2, 1000)
logeps1vec = -np.logspace(-4, np.log10(50), 500)  # final override

eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

laguerre_polynomials = generate_laguerre_polynomials(num_terms)

for idx, nu in enumerate(nu_values):
    for k in range(1):
        umulti = curves[k]
        N1 = 2 / b_values[k] - 1
        print(N1)

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

        # Note: i is still 0 here, same as in original script
        phiAmat[i, :] = phiA
        phiBmat[i, :] = phiB
        chi_mat[i, :] = chi_vec

        phiAmat_app0[i, :] = phiA_app0
        phiBmat_app0[i, :] = phiB_app0
        chi_mat_app0[i, :] = chi_vec_app0

        if idx == 1:
            axFS.plot(
                phiA,
                chi_vec,
                label=r"Cloud curve",
                linewidth=3,
                color=colorcloud,
                linestyle="dashed",
                dashes=(2, 0),
            )
            axFS.plot(
                phiB,
                chi_vec,
                linewidth=3,
                color=colorshadow,
                linestyle="dashed",
                dashes=(2, 0.5),
            )
        else:
            axFS.plot(
                phiA,
                chi_vec,
                label=r"Shadow Curve ",
                linewidth=3,
                color=colorshadow,
                linestyle="dashed",
                dashes=(2, 0.5),
            )
            axFS.plot(
                phiB,
                chi_vec,
                linewidth=3,
                color=colorcloud,
                linestyle="dashed",
                dashes=(2, 0),
            )

        axFS.set_xticks([0, 0.2, 0.4])
        axFS.set_yticks([0.5, 0.6, 0.7])
        axFS.set_ylim(0.5, 0.7)
        axFS.set_xlim(0, 0.4)
        axFS.legend(fontsize=8, frameon=False, loc="lower left")

        if i == 0:
            labelstring = "(l)"
            axFS.set_title(rf"Flory-Schulz, $b$ ={1 - b_values[k]}", fontsize=12)
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

        # Cloud/shadow evaluation at chi0 (kept as in original)
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
# (j) Modified Poisson distribution – cloud and shadow
# ============================================================
N1 = 200
b_values = [200]

Nmin, Nmax = 100, 300
Nmulti = np.arange(Nmin, Nmax + 1)

curves = [evaluate_mod_poisson(bval, Nmulti) for bval in b_values]

Nshift = 0  # not used

param = {"Nmulti": Nmulti}

eps1vec = np.exp(logeps1vec)
y1vec = 1 - eps1vec

phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

laguerre_polynomials = generate_laguerre_polynomials(num_terms)

for idx, nu in enumerate(nu_values):
    for k in range(1):
        umulti = curves[k]
        N1 = b_values[k] + 2
        print(N1)

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

        if idx == 1:
            axMod.plot(
                phiA,
                chi_vec,
                label=r"Cloud curve",
                linewidth=3,
                color=colorcloud,
                linestyle="dashed",
                dashes=(2, 0),
            )
            axMod.plot(
                phiB,
                chi_vec,
                linewidth=3,
                color=colorshadow,
                linestyle="dashed",
                dashes=(2, 0.5),
            )
        else:
            axMod.plot(
                phiA,
                chi_vec,
                label=r"Shadow Curve ",
                linewidth=3,
                color=colorshadow,
                linestyle="dashed",
                dashes=(2, 0.5),
            )
            axMod.plot(
                phiB,
                chi_vec,
                linewidth=3,
                color=colorcloud,
                linestyle="dashed",
                dashes=(2, 0),
            )

        axMod.set_xticks([0.05, 0.08])
        axMod.set_yticks([0.573, 0.574])
        axMod.set_ylim(0.5725, 0.574)
        axMod.set_xlim(0.05, 0.08)
        axMod.legend(fontsize=8, frameon=False, loc="lower left")

        if i == 0:
            labelstring = "(j)"
            axMod.set_title(
                rf"Modified-Poisson, $\lambda$ ={b_values[k]}", fontsize=12
            )
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
# (k) Bimodal distribution – cloud and shadow
# ============================================================
N1 = 150
N2 = 250
std_dev = 20.0
frac_vals = [0.5]

chi0 = np.array([0.8])

Nmin, Nmax = 1, 450
Nmax = 2000  # as in original code
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

colorsP = ["#a1dab4", "#41b6c4", "#225ea8"]  # (redeclared, same as before)

Nshift = 0  # unused

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

for idx, nu in enumerate(nu_values):
    for k in range(1):
        umulti = gaussian_curves[k]
        i = 0
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

        if idx == 1:
            axBim.plot(
                phiA,
                chi_vec,
                label=r"Cloud curve",
                linewidth=3,
                color=colorcloud,
                linestyle="dashed",
                dashes=(2, 0),
            )
            axBim.plot(
                phiB,
                chi_vec,
                linewidth=3,
                color=colorshadow,
                linestyle="dashed",
                dashes=(2, 0.5),
            )
        else:
            axBim.plot(
                phiA,
                chi_vec,
                label=r"Shadow Curve ",
                linewidth=3,
                color=colorshadow,
                linestyle="dashed",
                dashes=(2, 0.5),
            )
            axBim.plot(
                phiB,
                chi_vec,
                linewidth=3,
                color=colorcloud,
                linestyle="dashed",
                dashes=(2, 0),
            )

        axBim.set_xticks([0, 0.1, 0.2])
        axBim.set_yticks([0.55, 0.6])
        axBim.set_ylim(0.55, 0.6)
        axBim.set_xlim(0, 0.2)
        axBim.legend(fontsize=8, frameon=False, loc="lower left")

        if i == 0:
            labelstring = "(k)"
            axBim.set_title(rf"Bimodal, $\eta$ ={frac_vals[k]}", fontsize=12)
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

# ============================================================
# Final layout & save
# ============================================================
plt.tight_layout()
plt.savefig("figure3il_final.pdf", format="pdf")
plt.show()
