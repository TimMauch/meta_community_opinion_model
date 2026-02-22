import numpy as np
import sympy as sp
from pathlib import Path

from scipy.signal import argrelextrema
from scipy.optimize import fsolve

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import opinion_model as om

def bifurcation_alpha(x, beta, gamma, M):
    '''
    One-parameter bifurcation solution for alpha.
    Eq. (3) set to zero and solved for alpha.

    Parameters:
    x       : density of opinion X
    alpha   : spontaneous flipping parameter from X to Y
    beta    : spontaneous flipping parameter from Y to X
    gamma   : interaction based flipping parameter
    M       : total population per patch

    Returns:
    alpha   : value of alpha at the steady state
    '''

    alpha = (beta*(M-x) + gamma*x**2*(M-x) - gamma*x*(M-x)**2) * (1/x)

    return alpha


def one_patch_2p_bifurcation(x_range, beta_eval, M_eval):
    '''
    Derivation of the (alpha,gamma)-parameter local Bifurcation diagram.
    Details are given in Appendix B of the paper.

    Parameters:
    x_range     : range of x values used for the conditions (between 0 and M)
    beta_eval   : evaluation value of beta
    M_eval  : evaluation value of M

    Returns:
    alpha_val   : array of alpha values at the bifurcation points
    gamma_val   : array of gamma values at the bifurcation points
    h_1         : function handle for Condition 1 (used in Fig. A1)
    h_2         : function handle for Condition 2 (used in Fig. A1)
    '''

    # Define local differential equation with sympy
    x, alpha, beta, gamma, M = sp.symbols('x, alpha, beta, gamma, M')
    x_dot = -alpha*x + beta*(M-x) + gamma*x**2*(M-x) - gamma*x*(M-x)**2

    # First Condition: Fixed Point, x_dot=0 -> solve for alpha
    condition_1 = sp.solve(x_dot, alpha)[0]
    # Second Condition: Saddle node bifurcations
    h_1 = sp.lambdify((x, beta, gamma, M), condition_1, 'numpy')
    condition_2 = sp.diff(condition_1, x)
    h_2 = sp.lambdify((x, beta, gamma, M), condition_2, 'numpy')
    condition_2 = sp.solve(condition_2, gamma)[0]
    # Resubstitue Condition 2 in Condition 1
    condition_1 = condition_1.subs(gamma, condition_2)

    # Parametric solution over x_range
    alpha_result = []
    gamma_result = []
    for x_eval in x_range:
        # Substitute fixed parameter values
        cond_1_eval = condition_1.subs({beta: beta_eval, M: M_eval})
        cond_2_eval = condition_2.subs({beta: beta_eval, M: M_eval})
        # Convert into numpy function
        cond_alpha = sp.lambdify((x), cond_1_eval, modules='numpy')
        cond_gamma = sp.lambdify((x), cond_2_eval, modules='numpy')
        # Evaluate at x_eval
        alpha_ = cond_alpha(x_eval)
        gamma_ = cond_gamma(x_eval)
        # Append only positive values
        if (alpha_ >= 0 and gamma_ >= 0):
            alpha_result.append(alpha_)
            gamma_result.append(gamma_)

    return np.array(alpha_result), np.array(gamma_result), h_1, h_2


def multi_patch_2p_bifurcation(x_range, beta_eval, M_eval, mu_x_eval, mu_y_eval, kappa_eval):
    '''
    Derivation of the (alpha,gamma)-parameter Bifurcation diagram for multi-patch system.
    Details are given in Appendix D of the paper.

    Parameters:
    beta_eval   : evaluation value of beta
    M_eval      : evaluation value of M
    mu_x_eval   : evaluation value of mu_x
    mu_y_eval   : evaluation value of mu_y
    kappa_eval  : evaluation value of kappa

    Returns:
    alpha_val   : array of alpha values at the bifurcation points
    gamma_val   : array of gamma values at the bifurcation points
    '''

    # Define symbols
    x, alpha, beta, gamma, M, mu_x, mu_y, kappa = sp.symbols('x, alpha, beta, gamma, M, mu_x, mu_y, kappa')
    
    # Condition 1 for fixpoint
    x_dot = -alpha*x + beta*(M-x) + gamma*x**2*(M-x) - gamma*x*(M-x)**2
    cond_1 = sp.solve(x_dot, alpha)[0] # solve for alpha
    # Condition 2 - determinant at 0
    P11 = -alpha + 2*gamma*x*(M-x) - gamma*(M-x)**2
    P12 = beta - 2*gamma*x*(M-x) + gamma*x**2
    P21 = alpha - 2*gamma*x*(M-x) + gamma*(M-x)**2
    P22 = -beta + 2*gamma*x*(M-x) - gamma*x**2
    det = (P11-kappa*mu_x) * (P22-kappa*mu_y) - P12*P21
    # Substitute condition 1 into determinant
    cond_2 = det.subs(alpha, cond_1)
    cond_2_ = sp.solve(cond_2, gamma)[0] # Solve for gamma
    cond_1_ = cond_1.subs(gamma, cond_2_) # Resubstitue condition 2 into condition 1

    # Substitute fixed parameter values
    cond_1_eval = cond_1_.subs({beta: beta_eval, M:M_eval, mu_x:mu_x_eval, mu_y:mu_y_eval, kappa:kappa_eval})
    cond_2_eval = cond_2_.subs({beta: beta_eval, M:M_eval, mu_x:mu_x_eval, mu_y:mu_y_eval, kappa:kappa_eval})
    # Turn into numpy function
    cond_alpha = sp.lambdify((x), cond_1_eval, modules='numpy')
    cond_gamma = sp.lambdify((x), cond_2_eval, modules='numpy')
    
    # Parametric solution over x_range
    alpha_result = []
    gamma_result = []
    for x in x_range:
        alpha_ = cond_alpha(x)
        gamma_ = cond_gamma(x)
        if (alpha_ >= 0 and gamma_ >= 0):
            alpha_result.append(alpha_)
            gamma_result.append(gamma_)

    return np.array(alpha_result), np.array(gamma_result)

def alpha_gamma_msf_eigv(alpha_vals, gamma_vals, beta, M, mu_x, mu_y, x_root_idx, kappa_vals):

    P_11 = lambda x,y,alpha,gamma: -gamma*y**2 + 2*gamma*x*y - alpha
    P_12 = lambda x,y,beta,gamma: gamma*x**2 - 2*gamma*x*y + beta
    P_21 = lambda x,y,alpha,gamma: gamma*y**2 - 2*gamma*x*y + alpha
    P_22 = lambda x,y,beta,gamma: -gamma*x**2 + 2*gamma*x*y - beta

    max_eigvals = np.zeros([len(alpha_vals), len(gamma_vals)])

    for idx, alpha in enumerate(alpha_vals):
        for idy, gamma in enumerate(gamma_vals):
            local_params = (alpha, beta, gamma, mu_x, mu_y, M)
            roots = om.roots_symbolic(alpha, beta, gamma, M)
            x_root = roots[x_root_idx]
            y_root = M - x_root
            kappa_crit = ((-gamma*y_root**2 + 2*gamma*x_root*y_root - alpha)/mu_x) + ((-gamma*x_root**2 + 2*gamma*x_root*y_root - beta)/mu_y)
            
            if any(value < kappa_crit for value in kappa_vals):
                leading_Msf_eigv = []
                for kappa in kappa_vals:
                    Msf = np.array([[P_11(x_root,y_root,alpha,gamma)-kappa*mu_x,  P_12(x_root, y_root, beta, gamma)], 
                                    [P_21(x_root,y_root,alpha,gamma),             P_22(x_root, y_root, beta, gamma)-kappa*mu_y]])
                    Msf_eigv = np.linalg.eigvals(Msf)
                    leading_Msf_eigv.append(max(Msf_eigv.real))
                max_eigvals[idx, idy] = np.array(leading_Msf_eigv).max()

    return max_eigvals
                

def plot_one_patch_bifurcation_diagram(x_range, one_para_alpha, two_para_alpha, two_para_gamma, beta,
                                figsize=((3+3/8)*1.5, 1.5*1.5), gridspec_kw={'wspace':0.5}, 
                                path=None, filename="bifurcation_diagram.svg"):
    '''
    Plot the one-parameter and two-parameter bifurcation diagrams.

    Parameters:
    x_range            : array of x values for the one-parameter bifurcation diagram 
    one_para_alpha     : array of alpha values for the one-parameter bifurcation diagram (return from bifurcation_alpha)
    two_para_alpha     : array of alpha values for the two-parameter bifurcation diagram (return from two_parameter_bifurcation)
    two_para_gamma     : array of gamma values for the two-parameter bifurcation diagram (return from two_parameter_bifurcation)
    '''

    fig, ax = plt.subplots(2,1, figsize=figsize, gridspec_kw=gridspec_kw)

    # One parameter bifrucation diagram
    local_maxima_idx = argrelextrema(one_para_alpha, np.greater)[0] # upper saddle node point index
    local_minima_idx = argrelextrema(one_para_alpha, np.less)[0] # lower saddle node point index
    ax[0].plot(one_para_alpha[:local_minima_idx[0]], x_range[:local_minima_idx[0]], '-', color='#5d81b4')
    ax[0].plot(one_para_alpha[local_minima_idx[0]:local_maxima_idx[0]], x_range[local_minima_idx[0]:local_maxima_idx[0]], '--',color='#5d81b4')
    ax[0].plot(one_para_alpha[local_maxima_idx[0]:], x_range[local_maxima_idx[0]:], '-',color='#5d81b4')
    #ax[0].plot(one_para_alpha[local_maxima_idx[0]:], x_range[local_maxima_idx[0]:], '-',color='#5d81b4')
    # ax1.set_xlabel(r'$\alpha$')
    ax[0].set_ylabel(r'$X^*$')
    ax[0].set_xlim([0,1])
    ax[0].set_ylim([0,1])
    ax[0].tick_params(axis='x')
    ax[0].tick_params(axis='y')

    # Two-parameter bifurcation diagram
    ax[1].plot(two_para_alpha, two_para_gamma)
    label_x = r"$ \alpha $"
    label_y = r"$ \widetilde{\gamma} $"
    ax[1].set_xlabel(label_x)
    ax[1].set_ylabel(label_y) 
    ax[1].tick_params(axis='x')
    ax[1].tick_params(axis='y')
    ax[1].set_ylim([0,10])
    ax[1].set_xlim([0,1.0])

    # Add additional labels, shadings and arrows
    ax[0].set_xticklabels([])
    ax[0].tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False)   
    ax[1].tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False) 
    # Subfigure Label
    ax[0].text(-0.13, 1.0, "(a)", transform=ax[0].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
    ax[1].text(-0.13, 1.0, "(b)", transform=ax[1].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
    # Shading
    ax[1].fill_between(two_para_alpha, two_para_gamma, 10, color='#e09b24', alpha=0.2, lw=0)
    ax[0].fill_between(one_para_alpha[local_minima_idx[0]:local_maxima_idx[0]], 0, 10, color='#e09b24', alpha=0.2, lw=0)
    # Superior, Inferiority indication
    y_pos = 0.6 
    ax[1].text(beta, y_pos, r'$\beta$', ha='center', va='center', fontsize=10)
    ax[1].annotate('', xy=(beta - 0.05, y_pos), xytext=(0.05, y_pos),
                   arrowprops=dict(arrowstyle='<|-', color='black', lw=0.8))
    ax[1].text(beta/2, y_pos + 0.4, r'$X_{superior}$', ha='center', fontsize=9)
    ax[1].annotate('', xy=(0.95, y_pos), xytext=(beta + 0.05, y_pos),
                   arrowprops=dict(arrowstyle='-|>', color='black', lw=0.8))
    ax[1].text((beta + 1.0)/2, y_pos + 0.4, r'$X_{inferior}$', ha='center', fontsize=9)
    # Arrows
    alpha_start = one_para_alpha[20:local_minima_idx[0]][0]
    x_start = x_range[20:local_minima_idx[0]][0]
    ax[0].plot(alpha_start, x_start, 'ro', markersize=4, color="#eb0909")
    alpha_tipping = one_para_alpha[local_minima_idx[0]]
    x_lower = x_range[20:local_minima_idx[0]][-1]-0.009
    alpha_tipping_idx = (np.abs(one_para_alpha[local_maxima_idx[0]:] - alpha_tipping)).argmin()
    x_upper = (x_range[local_maxima_idx[0]:][alpha_tipping_idx-1]+x_range[local_maxima_idx[0]:][alpha_tipping_idx+1])/2
    x_upper += 0.015
    ax[0].plot(one_para_alpha[20:local_minima_idx[0]], x_range[20:local_minima_idx[0]], '-', color="#eb0909", lw=1.5)
    ax[0].annotate('', xy=(alpha_tipping, x_lower), 
                   xytext=(alpha_tipping, x_upper),
                   arrowprops=dict(arrowstyle="<-", color="#eb0909", lw=1.5))
    alpha_end = 0.23
    gamma = 3.8
    ax[1].plot(alpha_start, gamma, 'ro', markersize=4, color="#eb0909")
    ax[1].annotate('', xy=(alpha_start, gamma), 
                   xytext=(alpha_end, gamma),
                   arrowprops=dict(arrowstyle="<-", color="#eb0909", lw=1.5))

    plt.tight_layout()
    if path:
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        full_path = save_dir / filename
        if full_path.exists():
            print(f"File {filename} already exists in {save_dir}, skipping save.")
        else:
            plt.savefig(full_path, dpi=900, bbox_inches="tight")
    plt.show()


def plot_multi_patch_2p_bifurcation_diagram(x_range, beta_eval, M_eval, mu_x_eval, mu_y_eval, kappa_vals, 
                                            alpha_vals, gamma_vals, max_msf_eigv,
                                            xlim=[0,1.5], ylim=[0,0.05], figsize=((3+3/8)*1.5, 1.5*1.5), gridspec_kw={'wspace':0.5}, 
                                            path=None, filename="multi_patch_bifurcation_diagram.svg"):
    '''
    Plot the one-parameter and two-parameter bifurcation diagrams.

    Parameters:
    x_range         : array of x values for the bifurcation diagram
    beta_eval       : evaluation value of beta
    M_eval          : evaluation value of M
    mu_x_eval       : list of two mu_x, first for axis 1 (mu_y > mu_x), second for axis 2 (mu_x > mu_y) 
    mu_y_eval       : list of two mu_y, first for axis 1 (mu_y > mu_x), second for axis 2 (mu_x > mu_y)
    kappa_vals      : list of two arrays of kappa values, first for axis 1 (mu_y > mu_x), second for axis 2 (mu_x > mu_y)
    ...
    '''

    kappa_eval_ax_1 = kappa_vals[0]
    kappa_eval_ax_2 = kappa_vals[1]

    fig, ax = plt.subplots(1,2,figsize=figsize, gridspec_kw=gridspec_kw, constrained_layout=True)
    # Plot bifurcation line of one-patch system
    one_patch_alpha, one_patch_gamma, _, _ = one_patch_2p_bifurcation(x_range, beta_eval, M_eval)
    ax[0].plot(one_patch_alpha, one_patch_gamma, markersize=4, linestyle='-', label="Saddle-Node", color="black")
    ax[1].plot(one_patch_alpha, one_patch_gamma, markersize=4, linestyle='-', label="Saddle-Node", color="black")
    # Axis 1: mu_y > mu_x diffusion
    for kappa_eval in kappa_eval_ax_1:
        multi_patch_alpha, multi_patch_gamma = multi_patch_2p_bifurcation(x_range, beta_eval, M_eval, mu_x_eval[0], mu_y_eval[0], kappa_eval)
        ax[0].plot(multi_patch_alpha, multi_patch_gamma, markersize=4, linestyle='-', color="#e09b24")
    
    # Axis 2: mu_x > mu_y diffusion
    for kappa_eval in kappa_eval_ax_2:
        multi_patch_alpha, multi_patch_gamma = multi_patch_2p_bifurcation(x_range, beta_eval, M_eval, mu_x_eval[1], mu_y_eval[1], kappa_eval)
        ax[1].plot(multi_patch_alpha, multi_patch_gamma, markersize=4, linestyle='-', color="#5d81b4")

    colors = ["#00000027", "#00000088", "#000000FF"]
    grey_cmap = mcolors.LinearSegmentedColormap.from_list("custom_orange", colors)

    extent = [alpha_vals.min(), alpha_vals.max(), gamma_vals.min(), gamma_vals.max()]
    plt.sca(ax[1])
    im = plt.imshow(max_msf_eigv.T, 
            extent=extent,
            origin='lower', 
            cmap=grey_cmap,
            aspect='auto',
            norm=mcolors.PowerNorm(gamma=1))
            # norm=mcolors.LogNorm())

    # Add labels and title
    cbar = plt.colorbar(im, orientation='vertical') 
    cbar.set_label(r'$m(\kappa)$', labelpad=5)
    cbar.minorticks_off()
    ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel(r'$\gamma$')
    ax[1].set_xlabel(r'$\alpha$')
    ax[0].set_ylim(ylim)
    ax[0].set_xlim(xlim)
    ax[1].set_ylim(ylim)
    ax[1].set_xlim(xlim)

    ax[1].set_yticklabels([])
    ax[1].tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False)   
    ax[0].tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False) 

    ax[0].text(-0.35, 1.0, "(a)", transform=ax[0].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
    ax[1].text(-0.02, 1.0, "(b)", transform=ax[1].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

    # plt.tight_layout()
    if path:
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        full_path = save_dir / filename
        if full_path.exists():
            print(f"File {filename} already exists in {save_dir}, skipping save.")
        else:
            plt.savefig(full_path, dpi=900, bbox_inches="tight")
    plt.show()



def plot_phase_portraits(x_range, alpha_vals, beta_vals, gamma_vals, M, h_1, h_2,
                         figsize=((3+3/8)*1.5, 1.5*1.5), gridspec_kw={'wspace':0.3}, 
                         path=None, filename="phase_portrait.svg"):
    '''
    Plot phase portraits for different parameter values.

    Parameters:
    alpha_vals  : list of alpha values for different scenarios
    beta_vals   : list of beta values for different scenarios
    gamma_vals  : list of gamma values for different scenarios
    M           : total population per patch
    x_range     : range of x values for plotting
    h_1         : function handle for Condition 1 (used in Fig. A1)
    h_2         : function handle for Condition 2 (used in Fig. A1)
    '''
    # Create a figure and axis for the plot
    fig, [ax1, ax2] = plt.subplots(1,2,figsize=figsize, gridspec_kw=gridspec_kw)

    # Function to find steady states
    def roots_numerical(alpha, beta, gamma , M):
        steady_states = fsolve(om.one_patch_model, x0=[0.1, 0.5, 0.9], args=(alpha, beta, gamma , M))
        return steady_states

    # Function to plot arrows along the curve
    def plot_arrows_on_curve(x_vals, y_vals, a, b, c, N, n_arrows=10):
        # Select n_arrows positions evenly spaced
        arrow_positions = np.linspace(3, len(x_vals)-4, n_arrows, dtype=int)
        for i in arrow_positions:
            x = x_vals[i]
            y = y_vals[i]
            # Calculate slope to determine arrow direction
            dx = 0.01
            dy = om.one_patch_model(x + dx, a, b, c, N) - om.one_patch_model(x, a, b, c, N)
            # Normalize the direction
            norm = np.sqrt(dx**2 + dy**2)
            dx, dy = dx / norm, dy / norm
            # Reverse arrow direction if below x-axis
            if y < 0:
                dx, dy = -dx, -dy
            # Plot the arrow on the curve
            ax1.arrow(x, y, dx*0.025, dy*0.025, head_width=0.01, head_length=0.01, color='black', zorder=10)

    # Loop through different parameter values and plot the phase portraits
    colors = ['#5d81b4', '#e09b24', '#8eb031', '#eb6235', '#8678b2', '#c46e1a']
    labels = [r"$\alpha=\beta=0$", r"$\alpha=\beta>0$", r"$\alpha \neq \beta$"]
    for idx, (a, b, c) in enumerate(zip(alpha_vals, beta_vals, gamma_vals)):
        y_vals = om.one_patch_model(x_range, a, b, c, M)
        # Plot the dynamics curve
        ax1.plot(x_range, y_vals, label=labels[idx], color=colors[idx], linestyle="-")
        # Find steady states and mark them
        steady_states = roots_numerical(a, b, c, M)
        for ss in steady_states:
            ax1.scatter(ss, 0, zorder=5, marker="x", s=15, color=colors[3])  # Mark steady states
        # Plot arrows along the curve
        plot_arrows_on_curve(x_range, y_vals, a, b, c, M)

    # Customize the plot
    ax1.axhline(0, color='black', linewidth=0.5)  # X-axis
    ax1.axvline(0, color='black', linewidth=0.5)  # Y-axis
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\dot{x}$')
    ax1.set_xlim([-0.05,1.05])
    ax1.set_ylim([-0.3,0.75])
    ax1.legend(loc="upper right", )

    ax2.axhline(0.6, color=colors[0], linestyle="-", label=r"$h_1$")
    ax2.plot(x_range, h_1, label=r"$h_2$", color=colors[1], linestyle="-")
    ax2.plot(x_range, h_2, label=r"${d_xh_2}$", color=colors[2], linestyle="-")
    ax2.set_xlabel(r'$x^*$')
    ax2.set_ylabel(r'$\alpha$')
    ax2.axhline(0,linewidth=0.5,color="black")
    ax2.axvline(0,linewidth=0.5, color="black")
    ax2.set_ylim([-0.1, 3])
    ax2.set_xlim([-0.05,1.05])
    ax2.legend()

    ax1.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False)   
    ax2.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False) 
    ax1.text(-0.2, 1.0, "(a)", transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
    ax2.text(-0.13, 1.0, "(b)", transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

    # Show the plot
    plt.tight_layout()
    if path:
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        full_path = save_dir / filename
        if full_path.exists():
            print(f"File {filename} already exists in {save_dir}, skipping save.")
        else:
            plt.savefig(full_path, dpi=900, bbox_inches="tight")
    plt.show()


def master_stability_function(alpha_vals, beta_vals, gamma_vals, mu_x_vals, mu_y_vals,
                              kappa_vals, x_root_idx, M):
    '''
    Compute the master stability function for given parameter values.

    Parameters:
    alpha_vals  : list of alpha values for different scenarios
    beta_vals   : list of beta values for different scenarios
    gamma_vals  : list of gamma values for different scenarios
    mu_x_vals   : list of mu_x values for different scenarios
    mu_y_vals   : list of mu_y values for different scenarios
    kappa_vals  : array of kappa values for the Laplacian eigenvalues
    x_root_idx  : index of the root to be evaluated
    M           : total population per patch

    Returns:
    real_eigvals    : list of arrays of real parts of eigenvalues for each scenario
    im_eigvals      : list of arrays of imaginary parts of eigenvalues for each scenario
    intersect       : list of intersection points for each scenario
    '''
    
    # Jacobian matrix elements
    P_11 = lambda x,y,alpha,gamma: -gamma*y**2 + 2*gamma*x*y - alpha
    P_12 = lambda x,y,beta,gamma: gamma*x**2 - 2*gamma*x*y + beta
    P_21 = lambda x,y,alpha,gamma: gamma*y**2 - 2*gamma*x*y + alpha
    P_22 = lambda x,y,beta,gamma: -gamma*x**2 + 2*gamma*x*y - beta

    real_eigvals    = []
    im_eigvals      = []
    intersect       = []

    for alpha, beta, gamma, mu_x, mu_y in zip(alpha_vals, beta_vals, gamma_vals, mu_x_vals, mu_y_vals):

        roots = om.roots_symbolic(alpha, beta, gamma, M)
        root_x = roots[x_root_idx]
        root_y = M - root_x

        leading_eigv = []
        eigval = []

        for kappa in kappa_vals:
            Msf = np.array([[P_11(root_x,root_y,alpha,gamma)-kappa*mu_x,  P_12(root_x,root_y,beta,gamma)], 
                          [P_21(root_x,root_y,alpha,gamma),             P_22(root_x, root_y, beta, gamma)-kappa*mu_y]])
            eigenvalues = np.linalg.eigvals(Msf)
            leading_eigv.append(max(eigenvalues.real))
            eigval.append(eigenvalues)

        real_eigvals.append(np.real(eigval))
        im_eigvals.append(np.imag(eigval))

        intersect.append((P_11(root_x,root_y,alpha,gamma)/mu_x) + (P_22(root_x, root_y, beta, gamma)/mu_y))

    return real_eigvals, im_eigvals, intersect





def plot_master_stability_function(kappa_vals, real_eigvals, im_eigvals, intersect, params_list,
                                   xlim=[-0.4,2], ylim=[-1.5,2.5], figsize=((3+3/8)*1.9, 1.5*2), gridspec_kw={'wspace':0.2,'hspace':0.2},
                                   path=None, filename="master_stability_function.svg"):

    
    fig, ax = plt.subplots(2,2, figsize=figsize, gridspec_kw=gridspec_kw, sharey=True, sharex=True, constrained_layout=True)

    idx = 0
    label = ["(a)", "(b)", "(c)", "(d)"]
    for i in [0,1]:
        for e in [0,1]:
            ax[i,e].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax[i,e].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            p1=ax[i,e].plot(kappa_vals, real_eigvals[idx][:,1], "-", label=r'$\Re(\lambda_1)$', color="#5d81b4", linewidth=1)
            p2=ax[i,e].plot(kappa_vals, real_eigvals[idx][:,0], "-", label=r'$\Re(\lambda_2)$', color="#e09b24", linewidth=1)
            p3=ax[i,e].plot(kappa_vals, im_eigvals[idx][:,0], "--", label=r'$\Im(\lambda_1)$', color="#5d81b4", linewidth=1)
            p4=ax[i,e].plot(kappa_vals, im_eigvals[idx][:,1], "--", label=r'$\Im(\lambda_2)$', color="#e09b24", linewidth=1)
            
            stats_text = (fr"$\alpha={params_list[0][idx]}$," + "\n" + \
                          fr"$\beta={params_list[1][idx]}$," + "\n" + \
                          fr"$\gamma={params_list[2][idx]}$," + "\n" + \
                          fr"$\mu_x={params_list[4][idx]}$," +  "\n" + \
                          fr"$\mu_y={params_list[5][idx]}$")
            
            
            # Use bbox to create the box effect
            ax[i,e].text(0.8, 0.9, stats_text, transform=ax[i,e].transAxes, 
                         fontsize=6, verticalalignment='top', horizontalalignment='left')
                         # bbox=dict(boxstyle='square', facecolor='white', alpha=1, edgecolor="black", pad=1))
            
            ax[i,e].set_xlim(xlim)
            ax[i,e].set_ylim(ylim)
            
            # Add this inside your for e in [0,1] loop:
            if idx == 0 or idx == 2:
                ax[i,e].text(-0.1, 1.0, label[idx], transform=ax[i,e].transAxes, 
                            fontsize=10, fontweight='bold', va='top', ha='right')
            else:
                ax[i,e].text(-0.02, 1.0, label[idx], transform=ax[i,e].transAxes, 
                            fontsize=10, fontweight='bold', va='top', ha='right')

            idx += 1



    ax[0,0].set_ylabel(r'$m(\kappa)$')
    ax[1,0].set_ylabel(r'$m(\kappa)$')
    ax[1,0].set_xlabel(r'$\kappa$')
    ax[1,1].set_xlabel(r'$\kappa$')

    ax[1, 1].axvline(x=intersect[-1], color='black', linestyle='--')
    ax[1, 1].annotate(r'$\tilde{\kappa}$', xy=(intersect[-1], 0), xytext=(0.3, 1.5), 
                    arrowprops=dict(arrowstyle='->', color='black'), color='black')

    for a in ax.flat:
        a.minorticks_off()

    handles = [p1[0], p2[0], p3[0], p4[0]]  # Handles to be included in the legend
    labels = [r'$\Re(\lambda_1)$', r'$\Re(\lambda_2)$', r'$\Im(\lambda_1)$', r'$\Im(\lambda_2)$'] # Custom labels
    ax[0,0].legend(handles, labels, loc='upper left', ncol=2,  frameon=True)

    if path:
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        full_path = save_dir / filename
        if full_path.exists():
            print(f"File {filename} already exists in {save_dir}, skipping save.")
        else:
            plt.savefig(full_path, dpi=900, bbox_inches="tight")
    plt.show()
