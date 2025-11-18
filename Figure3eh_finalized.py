from SharedFunctions_finalized import *

# Initialize variables
nu_values = [1.0]
N1 = 200
std_devs = [10, 20, 40]
chi0 = np.array([0.8])

# Create the x-axis values (as integers)
Nmin=1
Nmax=450
Nmulti = np.arange(Nmin, Nmax+1)  # x-axis limits from 320 to 480 (inclusive)



# Generate Gaussian curves
gaussian_curves = [1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(Nmulti - N1)**2 / (2 * std_dev**2)) for std_dev in std_devs]

# Define varying shades of blue, purple, and red
colorsP = ['#a1dab4', '#41b6c4', '#225ea8']  # Different shades of purple


Nshift = 0
# Create three identical plots side by side
plt.figure(figsize=(6, 5))  # Set the overall figure size
axNorm=plt.subplot(2,2,1)
axMod=plt.subplot(2,2,2)
axBim=plt.subplot(2,2,3)
axFS=plt.subplot(2,2,4)
plt.suptitle(r"$\nu=1$", fontsize=16)






# Set up parameters
param = {'Nmulti': Nmulti}
alphamat = np.ones((len(Nmulti), len(Nmulti)))


logeps1vec=np.linspace(-50, -1e-2, 100)

eps1vec=np.exp(logeps1vec)
y1vec=1-eps1vec



# Initialize result matrices
phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

# Generate Laguerre polynomials
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)


for k in range(3):
    umulti=gaussian_curves[k]
    nu=0
    i=0
    param['nu'] = nu

    chi_vec, phiA, phiB, phiAmulti, phiBmulti=compute_poly_binodal(logeps1vec, Nmulti, umulti, nu,N1, laguerre_polynomials)
    chi_vec_app0, phiA_app0, phiB_app0, phiAmulti_app0, phiBmulti_app0=approx_poly_binodal0(logeps1vec, Nmulti, umulti, nu,N1, laguerre_polynomials)

    phiAmat[i,:]=phiA
    phiBmat[i, :] = phiB
    chi_mat[i,:]=chi_vec

    phiAmat_app0[i, :] = phiA_app0
    phiBmat_app0[i, :] = phiB_app0
    chi_mat_app0[i, :] = chi_vec_app0

    if k == 0:
        dashset = (2, 0)
    elif k == 1:
        dashset = (2, 0.25)
    else:
        dashset = (2, 0.1, 1, 0.1)

    axNorm.plot(phiA, chi_vec ,label=rf'$N_\sigma$ ={std_devs[k]}', linewidth=4, color=colorsP[k], linestyle="dashed",
             dashes=dashset)
    axNorm.plot(phiB, chi_vec, linewidth=4, color=colorsP[k], linestyle="dashed",
             dashes=dashset)
    if k>-2:
        axNorm.plot(phiA_app0, chi_vec_app0, linewidth=0.75, color='black', linestyle='-')
        axNorm.plot(phiB_app0, chi_vec_app0, linewidth=0.75, color='black', linestyle='-')

    axNorm.set_xscale('log')
    axNorm.set_xticks([1e-10, 1e-5,1.0])
    axNorm.set_yticks([0.5, 0.7, 0.9])
    axNorm.set_ylim(0.5, 0.9)
    axNorm.set_xlim(1e-10, 1)
    axNorm.legend(fontsize=8, frameon=False, loc='lower left')



    if i == 0:
        labelstring='(e)'
        axNorm.set_title('Normal', fontsize=12)
        axNorm.set_xlabel(r'$\phi_t$', fontsize=12)
        axNorm.set_ylabel(r'$\chi$', fontsize=12)
        nustring = '$\\nu=$'+str(nu)


    axNorm.annotate(labelstring, xy=(-0.35, 1.1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    # plt.annotate(nustring, xy=(0.1, 0.2), xycoords='axes fraction', fontsize=12, ha='left', va='top')



#######################################################################Flory Schulz
# Initialize variables
N1 = 200
b_values=[0.025, 0.02, 0.01]
chi0 = np.array([0.8])

# Create the x-axis values (as integers)
Nmin=1
Nmax=800
Nmulti = np.arange(Nmin, Nmax+1)  # x-axis limits from 320 to 480 (inclusive)



# Generate Gaussian curves
curves = [bval**2*Nmulti*(1-bval)**(Nmulti-1) for bval in b_values]


# Set up parameters
param = {'Nmulti': Nmulti}




logeps1vec=-np.logspace(-300, -1e-2, 1000)
logeps1vec=-np.logspace(-2, np.log10(50), 500)

eps1vec=np.exp(logeps1vec)
y1vec=1-eps1vec



# Initialize result matrices
phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

# Generate Laguerre polynomials
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)


for k in range(3):
    umulti=curves[k]
    N1 = 2 / b_values[k] - 1
    print(N1)
    for i, nu in enumerate(nu_values):
        param['nu'] = nu

        chi_vec, phiA, phiB, phiAmulti, phiBmulti=compute_poly_binodal(logeps1vec, Nmulti, umulti, nu,N1, laguerre_polynomials)
        chi_vec_app0, phiA_app0, phiB_app0, phiAmulti_app0, phiBmulti_app0=approx_poly_binodal0(logeps1vec, Nmulti, umulti, nu,N1, laguerre_polynomials)

        phiAmat[i,:]=phiA
        phiBmat[i, :] = phiB
        chi_mat[i,:]=chi_vec

        phiAmat_app0[i, :] = phiA_app0
        phiBmat_app0[i, :] = phiB_app0
        chi_mat_app0[i, :] = chi_vec_app0

        if k == 0:
            dashset = (2, 0)
        elif k == 1:
            dashset = (2, 0.25)
        else:
            dashset = (2, 0.1, 1, 0.1)

        axFS.plot(phiA, chi_vec,label=rf'$b$ ={1-b_values[k]}', linewidth=3, color=colorsP[k], linestyle="dashed",
                 dashes=dashset)
        axFS.plot(phiB, chi_vec, linewidth=3, color=colorsP[k], linestyle="dashed",
                 dashes=dashset)
        if k>-2:
            axFS.plot(phiA_app0, chi_vec_app0, linewidth=0.75, color='black', linestyle='-')
            axFS.plot(phiB_app0, chi_vec_app0, linewidth=0.75, color='black', linestyle='-')

        axFS.set_xscale('log')
        axFS.set_xticks([1e-10, 1e-5,1.0])
        axFS.set_yticks([0.5, 0.7, 0.9])
        axFS.set_ylim(0.5, 0.9)
        axFS.set_xlim(1e-10, 1)
        axFS.legend(fontsize=8, frameon=True, loc='lower left')



        if i == 0:
            labelstring='(h)'
            axFS.set_title('Flory-Schulz', fontsize=12)
            axFS.set_xlabel(r'$\phi_t$', fontsize=12)
            axFS.set_ylabel(r'$\chi$', fontsize=12)
            nustring = '$\\nu=$'+str(nu)


        axFS.annotate(labelstring, xy=(-0.35, 1.1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
        # plt.annotate(nustring, xy=(0.1, 0.2), xycoords='axes fraction', fontsize=12, ha='left', va='top')



        logeps1vec_at_chi0=find_y1(chi_vec, logeps1vec, chi0)

        logeps1vec_at_chi0_app0 = find_y1(chi_vec_app0, logeps1vec, chi0)

        chival, phiA, phiB, phiAmulti, phiBmulti=compute_poly_binodal(logeps1vec_at_chi0, Nmulti, umulti, nu,N1, laguerre_polynomials)
        print(chival)
        chi_vec_app0, phiA_app0, phiB_app0, phiAmulti_app0, phiBmulti_app0=approx_poly_binodal0(logeps1vec_at_chi0_app0, Nmulti, umulti, nu,N1, laguerre_polynomials)


############################################################################################Poisson
# Initialize variables
N1 = 200
b_values=[180, 200, 220]
chi0 = np.array([0.8])

# Create the x-axis values (as integers)
Nmin=100
Nmax=300
Nmulti = np.arange(Nmin, Nmax+1)  # x-axis limits from 320 to 480 (inclusive)



# Generate Gaussian curves
curves = [evaluate_mod_poisson(bval, Nmulti) for bval in b_values]



Nshift = 0





# Set up parameters
param = {'Nmulti': Nmulti}






eps1vec=np.exp(logeps1vec)
y1vec=1-eps1vec



# Initialize result matrices
phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

# Generate Laguerre polynomials
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)


for k in range(3):
    umulti=curves[k]
    N1 = b_values[k]+2
    print(N1)
    for i, nu in enumerate(nu_values):
        param['nu'] = nu

        chi_vec, phiA, phiB, phiAmulti, phiBmulti=compute_poly_binodal(logeps1vec, Nmulti, umulti, nu,N1, laguerre_polynomials)
        chi_vec_app0, phiA_app0, phiB_app0, phiAmulti_app0, phiBmulti_app0=approx_poly_binodal0(logeps1vec, Nmulti, umulti, nu,N1, laguerre_polynomials)

        phiAmat[i,:]=phiA
        phiBmat[i, :] = phiB
        chi_mat[i,:]=chi_vec

        phiAmat_app0[i, :] = phiA_app0
        phiBmat_app0[i, :] = phiB_app0
        chi_mat_app0[i, :] = chi_vec_app0

        if k == 0:
            dashset = (2, 0)
        elif k == 1:
            dashset = (2, 0.25)
        else:
            dashset = (2, 0.1, 1, 0.1)


        axMod.plot(phiA, chi_vec,label=rf'$\lambda$ ={b_values[k]}', linewidth=3, color=colorsP[k], linestyle="dashed",
                 dashes=dashset)
        axMod.plot(phiB, chi_vec, linewidth=3, color=colorsP[k],linestyle="dashed",
                 dashes=dashset)
        if k>-2:
            axMod.plot(phiA_app0, chi_vec_app0, linewidth=0.75, color='black', linestyle='-')
            axMod.plot(phiB_app0, chi_vec_app0, linewidth=0.75, color='black', linestyle='-')
        axMod.set_xscale('log')
        axMod.set_xticks([1e-10, 1e-5,1.0])
        axMod.set_yticks([0.5, 0.7, 0.9])
        axMod.set_ylim(0.5, 0.9)
        axMod.set_xlim(1e-10, 1)
        axMod.legend(fontsize=8, frameon=False, loc='lower left')

        if i == 0:
            labelstring='(f)'
            axMod.set_title('Modified-Poisson', fontsize=12)
            axMod.set_xlabel(r'$\phi_t$', fontsize=12)
            axMod.set_ylabel(r'$\chi$', fontsize=12)
            nustring = '$\\nu=$'+str(nu)


        axMod.annotate(labelstring, xy=(-0.35, 1.1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
        # plt.annotate(nustring, xy=(0.1, 0.2), xycoords='axes fraction', fontsize=12, ha='left', va='top')



        logeps1vec_at_chi0=find_y1(chi_vec, logeps1vec, chi0)

        logeps1vec_at_chi0_app0 = find_y1(chi_vec_app0, logeps1vec, chi0)


        chival, phiA, phiB, phiAmulti, phiBmulti=compute_poly_binodal(logeps1vec_at_chi0, Nmulti, umulti, nu,N1, laguerre_polynomials)
        chi_vec_app0, phiA_app0, phiB_app0, phiAmulti_app0, phiBmulti_app0=approx_poly_binodal0(logeps1vec_at_chi0_app0, Nmulti, umulti, nu,N1, laguerre_polynomials)

#############################################################################Bimodal
# Initialize variables
N1 =150
N2= 250
std_dev = 20.0
frac_vals=[0.25, 0.5, 0.75]
chi0 = np.array([0.8])

# Create the x-axis values (as integers)
Nmin=1
Nmax=450
Nmulti = np.arange(Nmin, Nmax+1)  # x-axis limits from 320 to 480 (inclusive)



# Generate Gaussian curves
gaussian_curves = [frac / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(Nmulti - N1)**2 / (2 * std_dev**2))+(1-frac) / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(Nmulti - N2)**2 / (2 * std_dev**2)) for frac in frac_vals]

# Define varying shades of blue, purple, and red
colorsP = ['#a1dab4', '#41b6c4', '#225ea8']  # Different shades of purple


Nshift = 0
# Create three identical plots side by side





# Set up parameters
param = {'Nmulti': Nmulti}
alphamat = np.ones((len(Nmulti), len(Nmulti)))



eps1vec=np.exp(logeps1vec)
y1vec=1-eps1vec



# Initialize result matrices
phiAmat = np.zeros((len(nu_values), len(y1vec)))
phiBmat = np.zeros((len(nu_values), len(y1vec)))
chi_mat = np.zeros((len(nu_values), len(y1vec)))

phiAmat_app0 = np.zeros((len(nu_values), len(y1vec)))
phiBmat_app0 = np.zeros((len(nu_values), len(y1vec)))
chi_mat_app0 = np.zeros((len(nu_values), len(y1vec)))

# Generate Laguerre polynomials
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)


for k in range(3):
    umulti=gaussian_curves[k]
    i=0
    param['nu'] = nu

    chi_vec, phiA, phiB, phiAmulti, phiBmulti=compute_poly_binodal(logeps1vec, Nmulti, umulti, nu,N1, laguerre_polynomials)
    chi_vec_app0, phiA_app0, phiB_app0, phiAmulti_app0, phiBmulti_app0=approx_poly_binodal0(logeps1vec, Nmulti, umulti, nu,N1*frac_vals[k]+(1-frac_vals[k])*N2, laguerre_polynomials)

    phiAmat[i,:]=phiA
    phiBmat[i, :] = phiB
    chi_mat[i,:]=chi_vec

    phiAmat_app0[i, :] = phiA_app0
    phiBmat_app0[i, :] = phiB_app0
    chi_mat_app0[i, :] = chi_vec_app0

    if k==0:
        dashset = (2,0)
    elif k==1:
        dashset = (2, 0.25)
    else:
        dashset = (2, 0.1, 1, 0.1)

    axBim.plot(phiA, chi_vec, label=rf'$\eta$ ={frac_vals[k]}',linewidth=3, color=colorsP[k], linestyle = "dashed", dashes=dashset)
    axBim.plot(phiB, chi_vec, linewidth=3, color=colorsP[k], linestyle = "dashed", dashes=dashset)
    if k>-2:
        axBim.plot(phiA_app0, chi_vec_app0, linewidth=0.75, color='black', linestyle='-')
        axBim.plot(phiB_app0, chi_vec_app0, linewidth=0.75, color='black', linestyle='-')

    axBim.set_xscale('log')
    axBim.set_xticks([1e-10, 1e-5,1.0])
    axBim.set_yticks([0.5, 0.7, 0.9])
    axBim.set_ylim(0.5, 0.9)
    axBim.set_xlim(1e-10, 1)
    axBim.legend(fontsize=8, frameon=False, loc='lower left')



    if i == 0:
        labelstring='(g)'
        axBim.set_title('Bimodal', fontsize=12)
        axBim.set_xlabel(r'$\phi_t$', fontsize=12)
        axBim.set_ylabel(r'$\chi$', fontsize=12)
        nustring = '$\\nu=$'+str(nu)


    axBim.annotate(labelstring, xy=(-0.35, 1.1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    # plt.annotate(nustring, xy=(0.1, 0.2), xycoords='axes fraction', fontsize=12, ha='left', va='top')



plt.tight_layout()  # Adjust spacing between subplots

plt.savefig('figure3eh_final.pdf', format='pdf')

plt.show()