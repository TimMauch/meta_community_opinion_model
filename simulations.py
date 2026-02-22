import numpy as np
import sympy as sp
import networkx as nx
from pathlib import Path

from scipy.signal import argrelextrema
from scipy.optimize import fsolve
from scipy.integrate import odeint
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


import opinion_model as om
import bifurcation as bfc


def pattern_conditions(A, local_params, x_root, y_root):
    '''
    Check the conditions for pattern formation in the opinion model on a network.

    Parameters:
    A              : patch network adjacency matrix
    local_params   : tuple of local parameters (alpha, beta, gamma, mu_x, mu_y, M)
    x_root         : density of opinion X at the homogeneous steady state
    y_root         : density of opinion Y at the homogeneous steady state
    '''

    alpha, beta, gamma, mu_x, mu_y, M = local_params

    # Laplacian
    D = np.diag(np.sum(A, axis=1))
    L = np.array(D - A)
    eigv = np.linalg.eigh(L)[0]
    print(f"Lalacian Matrix Eigenvalues are: {np.round(eigv,4)}")

    # Check if conditions for pattern formation are met (kappa critical value from Eq (10) in the paper)
    kappa_crit = ((-gamma*y_root**2 + 2*gamma*x_root*y_root - alpha)/mu_x) + ((-gamma*x_root**2 + 2*gamma*x_root*y_root - beta)/mu_y)
    print(f'Critical value for kappa: {np.round(kappa_crit, 4)}')
    print(f'Any Laplacian Eigenvalue below kappa? {any(value < kappa_crit for value in eigv[1:])}')

    return eigv


def initial_conditions(A, local_params, x_root, y_root, noise_strength=1, int_noise=False, seed=None):
    '''
    Find initial conditions around the homogeneous steady state with added noise.

    Parameters:
    A              : patch network adjacency matrix
    local_params   : tuple of local parameters (alpha, beta, gamma, mu_x, mu_y, M)
    x_root         : density of opinion X at the homogeneous steady state
    y_root         : density of opinion Y at the homogeneous steady state
    noise_strength : strength of the added noise (only for float noise)
    int_noise      : whether to use integer noise, used for Gillespie algorithm (default: False)

    Returns:
    z0             : initial conditions array of length 2*num_patches
    '''

    _, _, _, _, _, M = local_params
    
    # Find maximum and minimum noise to keep initial conditions within bounds
    rng = np.random.default_rng(seed)
    if int_noise:
        max_noise = np.min([M-np.max([np.ceil(x_root), np.ceil(y_root)]), np.min([np.floor(x_root), np.floor(y_root)])])
        noise = rng.integers(low=-max_noise, high=max_noise+1, size=A.shape[0])
        x0 = np.full(A.shape[0], np.ceil(x_root)) + noise
        y0 = np.full(A.shape[0], np.floor(y_root)) - noise

    else:
        max_noise = np.min([M-np.max([x_root,y_root]), np.min([x_root,y_root])])
        noise = -0.5 * noise_strength + rng.uniform(0, 1, size=A.shape[0]) * noise_strength
        noise[noise > max_noise] = max_noise
        noise[noise < -max_noise] = -max_noise
        x0 = np.full(A.shape[0], x_root) + noise
        y0 = np.full(A.shape[0], y_root) - noise

    z0 = np.concatenate([x0, y0])

    return z0


def run_simulation(t, z0, x_root, y_root, local_params, A, normalize=True, ylim=[0,1],
                   figsize=(1.7*1.5, 1.5*1.5), path=None, filename="time_series.svg", plot=True):
    '''
    Run the simulation of the opinion model on a network and plot the time series.

    Parameters:
    t              : time array for the integration
    z0             : initial conditions array of length 2*num_patches
    x_root         : density of opinion X at the homogeneous steady state
    y_root         : density of opinion Y at the homogeneous steady state
    local_params   : tuple of local parameters (alpha, beta, gamma, mu_x, mu_y, M)
    A              : patch network adjacency matrix
    normalize      : whether to normalize the results by M (default: True)

    Returns:
    solution : array of shape (len(t), 2*num_patches) with the time evolution of opinions X and Y
    '''

    alpha, beta, gamma, mu_x, mu_y, M = local_params
    solution = odeint(om.model, z0, t, args=(A, alpha, beta, gamma, mu_x, mu_y))
    if normalize:
        x_result = solution[:, :A.shape[0]]/M
        y_result = solution[:, A.shape[0]:]/M
    else:
        x_result = solution[:, :A.shape[0]]
        y_result = solution[:, A.shape[0]:]

    if plot:
        plt.figure(figsize=figsize)
        for i in range(0, A.shape[0]):
            plt.plot(t, x_result[:,i], color="#5d81b4", linestyle="-")
            plt.plot(t, y_result[:,i], color="#e09b24", linestyle="-")
        plt.axhline(x_root/M, color="black", linestyle="--")
        plt.axhline(y_root/M, color="black", linestyle="--")
        plt.plot(t, y_result[:,i], color="#e09b24", linestyle="-")
        plt.xlabel(r'$t$')
        plt.ylabel(r'$X_i$ (Normalised)')
        plt.ylim(ylim)
        red_patch = plt.Line2D([0], [0], color='#5d81b4', linestyle='-', label=r'$X$')
        blue_patch = plt.Line2D([0], [0], color='#e09b24', linestyle='-', label=r'$Y$')
        plt.legend(handles=[red_patch, blue_patch], loc="upper left")

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

    return solution


def flipping_parameter(A, t, alpha_vals, gamma_vals, beta, M, mu_x, mu_y, x_root_idx, seed=10, int_noise=False):
    '''
    Checks the number of nodes where the majority opinion flips (compared to global majority) for a ragne of alpha and gamma values.

    Parameters:
    A           : Network structure
    t           : time interval for numerical integration of differential equation.
    alpha_vals  : range of alpha values to evaluate
    gamma_vals  : range of gamma values to evaluate
    beta        : beta value for evaluation
    M           : total mass for evaluation
    mu_x        : diffusion rate of x for evaluation
    mu_y        : diffuions rate of y for evaluation
    x_root_idx  : idx of x root to evaluate at
    seed        : seed

    Returns:
    solutions       : dictionary with time series for each alpha-gamma parameter combination
    flipped_nodes   : matrix with number of flipped nodes for each alpha-gamma combination
    '''

    P_11 = lambda x,y,alpha,gamma: -gamma*y**2 + 2*gamma*x*y - alpha
    P_12 = lambda x,y,beta,gamma: gamma*x**2 - 2*gamma*x*y + beta
    P_21 = lambda x,y,alpha,gamma: gamma*y**2 - 2*gamma*x*y + alpha
    P_22 = lambda x,y,beta,gamma: -gamma*x**2 + 2*gamma*x*y - beta

    flipped_nodes = np.zeros([len(alpha_vals), len(gamma_vals)])
    solutions = {}
    max_eigvals = {}

    D = np.diag(np.sum(A, axis=1))
    L = np.array(D - A)
    L_eigv = np.linalg.eigh(L)[0]

    for idx, alpha in enumerate(alpha_vals):
        for idy, gamma in enumerate(gamma_vals):
            local_params = (alpha, beta, gamma, mu_x, mu_y, M)
            roots = om.roots_symbolic(alpha, beta, gamma, M)
            x_root = roots[x_root_idx]
            y_root = M - x_root
            kappa_crit = ((-gamma*y_root**2 + 2*gamma*x_root*y_root - alpha)/mu_x) + ((-gamma*x_root**2 + 2*gamma*x_root*y_root - beta)/mu_y)
            kappa_indexes = np.where(L_eigv<kappa_crit)[0]
            if any(value < kappa_crit for value in L_eigv[1:]):
                leading_Msf_eigv = []
                for kappa in L_eigv[:len(kappa_indexes)]:
                    Msf = np.array([[P_11(x_root,y_root,alpha,gamma)-kappa*mu_x,  P_12(x_root, y_root, beta, gamma)], 
                                    [P_21(x_root,y_root,alpha,gamma),             P_22(x_root, y_root, beta, gamma)-kappa*mu_y]])
                    Msf_eigv = np.linalg.eigvals(Msf)
                    leading_Msf_eigv.append(max(Msf_eigv.real))
                max_eigvals[(alpha,gamma,kappa_crit)] = np.array(leading_Msf_eigv)

                z0 = initial_conditions(A, local_params, x_root, y_root, noise_strength=1, int_noise=int_noise, seed=seed)
                solution = run_simulation(t, z0, x_root, y_root, local_params, A, normalize=True, 
                                          figsize=(1.7*1.5, 1.5*1.5), path=None, plot=False)
                x_final = solution[-1, :A.shape[0]]
                y_final = solution[-1, A.shape[0]:]
                solutions[(alpha,gamma,kappa_crit)] = solution

                flipped_nodes[idx, idy] = len(np.where((x_final-y_final )* (x_final.sum() - y_final.sum()) < 0)[0])

    return flipped_nodes, solutions, max_eigvals


def plot_flipping(A, x_range, alpha_range, gamma_range, beta_eval, M_eval, mu_x_eval, mu_y_eval, num_L_eigv,
                flipped_nodes, alpha_msf, gamma_msf, x_root_idx,
                xlim=[[0,1.5],[0,1]], ylim=[[0,0.05],[0,0.1]], figsize=((3+3/8)*1.5, 1.5*1.5), gridspec_kw={'wspace':0.5}, 
                marker_size = 20, 
                path=None, filename="multi_patch_bifurcation_diagram.svg"):
    '''
    Plot the number of nodes where the majority opinion switches in favor of the global minority in the alpha gamma bifurcation plane.
    Plot the master stability function for different parameter combinations.

    Parameters:
    A               : Community Network Adjacency mattrix
    x_range         : array of x values for the bifurcation diagram
    alpha_range     : alpha values used for the flipping experiment (needed for heat map plot)
    gamma_range     : gamma values used for the flipping experiment (needed for heat map plot)
    beta_eval       : evaluation value of beta
    M_eval          : evaluation value of M
    mu_x_eval       : list of two mu_x, first for axis 1 (mu_y > mu_x), second for axis 2 (mu_x > mu_y) 
    mu_y_eval       : list of two mu_y, first for axis 1 (mu_y > mu_x), second for axis 2 (mu_x > mu_y)
    num_L_eigv      : number of laplacian eigenvalues from A that should for which the stability region in the bifurcation space should be plotted
    flipped_nodes   : matrix containing number of flipped nodes for each alpha gamma parameter combination
    alpha_msf       : alpha values that should be represented in the msf plot
    gamma_msf       : gamma values that should be represented in the msf plot
    x_root_idx      : index of the root of opinion X to initialize around
    '''
    D = np.diag(np.sum(A, axis=1))
    L = np.array(D - A)
    L_eigv = np.linalg.eigh(L)[0]

    # Get master stability functions for different alpha-gamma realizations
    kappa_msf = np.linspace(0, 1, 1000)
    beta_msf = [beta_eval] * len(alpha_msf)
    mu_x_msf = [mu_x_eval] * len(alpha_msf)
    mu_y_msf = [mu_y_eval] * len(alpha_msf)
    real_eigvals, _, intersect = bfc.master_stability_function(alpha_msf, beta_msf, gamma_msf, mu_x_msf, mu_y_msf, 
                                                               kappa_msf, x_root_idx, M_eval)

    fig, ax = plt.subplots(2,1,figsize=figsize, gridspec_kw=gridspec_kw, constrained_layout=True)
    extent = [min(alpha_range), max(alpha_range), min(gamma_range), max(gamma_range)]
    plt.sca(ax[0])
    im = plt.imshow(flipped_nodes.T, 
            extent=extent,
            origin='lower', 
            cmap='Greys',
            aspect='auto')
    # Plot bifurcation line of one-patch system
    one_patch_alpha, one_patch_gamma, _, _ = bfc.one_patch_2p_bifurcation(x_range, beta_eval, M_eval)
    # Axis 1: mu_y > mu_x diffusion
    marker_kappa = ['o', 's', '^', 'X', 'd']
    for idx, kappa_eval in enumerate(L_eigv[1:num_L_eigv]):
        multi_patch_alpha, multi_patch_gamma = bfc.multi_patch_2p_bifurcation(x_range, beta_eval, M_eval, mu_x_eval, mu_y_eval, kappa_eval)
        ax[0].plot(multi_patch_alpha, multi_patch_gamma, markersize=4, linestyle='-', marker=marker_kappa[idx], color="#5d81b4", markevery=20, label=rf'$\kappa_{idx+1}$')
        ax[1].scatter(kappa_eval, 0, color="#5d81b4", marker=marker_kappa[idx], s=marker_size)
    ax[0].plot(one_patch_alpha, one_patch_gamma, markersize=4, linestyle='-', color="black")
    marker_msf = ['P', '*', 'v', 'h', '|']
    for idx in range(len(alpha_msf)):
        ax[0].scatter(alpha_msf[idx], gamma_msf[idx], color="#e09b24", marker=marker_msf[idx], s=marker_size)
        ax[1].plot(kappa_msf, real_eigvals[idx][:,1], "-", markersize=4, marker=marker_msf[idx], color='#e09b24', linewidth=1, markevery=20, label=rf"$(\alpha,\gamma)_{idx}$")

    # Add labels and title
    cbar = plt.colorbar(im, location='top') 
    cbar.set_label("Num. Minority Pockets", labelpad=10)
    ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel(r'$\gamma$')
    ax[0].set_ylim(ylim[0])
    ax[0].set_xlim(xlim[0])
    ax[0].legend(loc="upper right", frameon=True, ncol=1)

    ax[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax[1].set_ylabel(r'$m(\kappa)$')
    ax[1].set_xlabel(r'$\kappa$')
    ax[1].legend(loc="upper right", frameon=True)
    ax[1].set_xlim(xlim[1])
    ax[1].set_ylim(ylim[1])

    ax[1].tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False)   
    ax[0].tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False) 

    ax[0].text(-0.15, 1.0, "(a)", transform=ax[0].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
    ax[1].text(-0.15, 1.0, "(b)", transform=ax[1].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

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


def create_pie_chart(G, pos, ax, prop_x_per_node, prop_y_per_node, node_sizes, total_x, total_y):
    '''
    Create pie charts on the network nodes using low-level Wedges to ensure 
    perfect coordinate alignment with NetworkX edges.

    This function was originally called in draw_graph_dist() to create the pi-chart diagram in Fig. 1
    Since the draw_graph_dist() function was updated, this funciton is not used anymore. 

    Parameters:
    G               : networkx graph
    pos             : positions of the nodes
    ax              : matplotlib axis to draw on
    prop_x_per_node : dictionary with proportion of opinion X per node
    prop_y_per_node : dictionary with proportion of opinion Y per node
    node_sizes     : dictionary with sizes of the nodes
    total_x        : total abundance of opinion X in the network
    total_y        : total abundance of opinion Y in the network

    Returns:
    ax              : matplotlib axis with the pie charts drawn
    '''
    
    for node in G.nodes():
        # 1. Get coordinates and size
        x, y = pos[node]
        radius = node_sizes[node]
        
        # Create Wedge patches
        # Slice 1: Opinion X (Starts at 0, ends at theta_x)
        theta_x = 360 * prop_x_per_node[node] # Calculate angle of the first slice (360 degrees * proportion)
        wedge_x = mpatches.Wedge(center=(x, y), r=radius, theta1=0, theta2=theta_x, 
                                 facecolor='#5d81b4', alpha=0.8,edgecolor='black',linewidth=0.5)
        # Slice 2: Opinion Y (Starts at theta_x, ends at 360)
        wedge_y = mpatches.Wedge(center=(x, y), r=radius, theta1=theta_x, theta2=360, 
                                 facecolor='#e09b24', alpha=0.8, edgecolor='black',linewidth=0.5)
        ax.add_patch(wedge_x)
        ax.add_patch(wedge_y)

        # Check for local majority condition (from your original code)
        if ((prop_x_per_node[node] - prop_y_per_node[node]) * (total_x - total_y)) < 0: 
            circle = plt.Circle((x, y), radius*1.5, color="grey", fill=True, alpha=0.4, zorder=-1)
            ax.add_patch(circle)
            
    return ax


def draw_graph_dist(G, t, solution, frames, size_scale=[10,75], layout="RGG", path=None, 
                    seed=15, laplace_eigv=1, prop="X", weights=None, kappa_0=False):
    '''
    Draw the network at different time points with pie charts representing the proportions of the two opinions.

    Parameters:
    G            : networkx graph
    t            : time array for the integration
    solution     : array of shape (len(t), 2*num_patches) with the time evolution of opinions X and Y
    frames       : list of time frame indices to draw
    size_scale   : scaling factor for the node sizes
    layout       : layout type for the network visualization ("RGG" or "spring")
    path         : path to save the figures (if None, figures are not saved)
    seed         : seed for the spring layout (if layout is "spring")
    laplace_eigv : index of the Laplacian eigenvector to use for node coloring (if int), 
                   or threshold for unstable eigenvalues (if float)
    prop         : which opinion proportion to use for coloring ("X" or "Y")
    weights      : weights for the unstable eigenvectors (if laplace_eigv is a float)
    kappa_0      : whether to exclude the zero eigenvalue from the unstable eigenvector calculation (default: False)

    Returns:
    filenames   : list of filenames of the saved figures (potentially needed for GIF creation)
    '''

    if layout == "RGG":
        pos = nx.get_node_attributes(G, "pos")
    else:
        pos = nx.spring_layout(G, seed=seed)
       #  pos = nx.kamada_kawai_layout(G)

    filenames = []
    patches = len(G.nodes)

    # Draw eigenvector of laplacian
    L = nx.laplacian_matrix(G).toarray()
    if type(laplace_eigv) == int:
        _, eigvecs = np.linalg.eigh(L)
        eig_vector = eigvecs[:, laplace_eigv]
    else:
        eigvals, eigvecs = np.linalg.eigh(L)
        unstable_indices = np.where(eigvals<laplace_eigv)[0][not kappa_0:]
        # eig_vector = np.sum(eigvecs[:, unstable_indices], axis=1)
        eig_vector = np.dot(eigvecs[:, unstable_indices], weights[not kappa_0:])
    norm = mpl.colors.Normalize(vmin=np.min(eig_vector), vmax=np.max(eig_vector))
    norm_eigenvector = norm(eig_vector)
    if path:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_aspect('equal')
        nx.draw(G, pos, node_color=norm_eigenvector, cmap=plt.cm.viridis, edge_color='gray', ax=ax)
        Path(path).mkdir(parents=True, exist_ok=True)
        full_path = Path(path) / "laplacian_ev.svg"
        plt.savefig(full_path, dpi=900, bbox_inches="tight")
        plt.close()
    else:
        print("No path specified, figures not saved.")
        

    # Loop over specified time frames
    for idx in frames:
        t_idx = t[idx] # time point
        # Calculate x and y abundacne and proportions
        x_abundance = solution[idx,:patches]
        y_abundance = solution[idx,patches:]
        x_per_node = {i: x_abundance[i] for i in G.nodes()}
        y_per_node = {i: y_abundance[i] for i in G.nodes()}
        total_per_node = {i: x_per_node[i] + y_per_node[i] for i in G.nodes()}
        total_x = np.sum(x_abundance)
        total_y = np.sum(y_abundance)
        prop_x_per_node = {i: x_per_node[i] / total_per_node[i] for i in G.nodes()}
        prop_y_per_node = {i: y_per_node[i] / total_per_node[i] for i in G.nodes()}
        # derive node size
        max_val = max(total_per_node.values())
        # max_val = np.mean(list(total_per_node.values()))
        # node_sizes = [(total_per_node[i] / (max_val*size_scale)) for i in G.nodes()] 
        node_sizes = {i: (total_per_node[i] / (max_val*size_scale[1])) for i in G.nodes()}

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_aspect('equal')
        nx.draw_networkx_edges(G, pos, alpha=0.8, ax=ax)
        ax = create_pie_chart(G, pos, ax, prop_x_per_node, prop_y_per_node, node_sizes, total_x, total_y)
        plt.axis("off")

        if path:
            filename = f'{path}/idx_{idx}_t_{round(t_idx,2)}.png'
            plt.savefig(filename, dpi=900, bbox_inches="tight")
            plt.close()
        else:
            plt.close()

        if idx == frames[-1]:
            colors = ["#AFAFAFFF", "#727272FF", "#000000FF"]
            grey_cmap = mcolors.LinearSegmentedColormap.from_list("custom_orange", colors)
            fig, axes = plt.subplots(1,3,figsize=((3+(3/8))*2.3, 5))
            # Draw eigenvector of laplacian
            nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.6, edge_color='black', node_size=0, arrows=False)
            network_1 = nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=norm_eigenvector, cmap=grey_cmap, node_size=size_scale[0])
            cbar1 = fig.colorbar(network_1, ax=axes[0], pad=0.04, orientation='horizontal')
            # cbar1.set_label(label=r"Aggregated $\mathbf{v}_i$ for $\kappa_i<\tilde{\kappa}$")
            # cbar1.set_label(label=r"$\sum_{i:m(\kappa_i)>0}m(\kappa_i)\mathbf{v}_i$")
            cbar1.set_label(label=r"$s$")
            # Draw network distribution at final time point
            if prop=="X":
                # norm = mpl.colors.Normalize(vmin=min(prop_x_per_node.values()), vmax=max(prop_x_per_node.values()))
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                norm_opinions = norm(np.array(list(prop_x_per_node.values())))
                norm_opinions = np.array(list(prop_x_per_node.values()))
                c_label = r'$X_i/(X_i+Y_i)$'
                # axes[1].set_title(f"Proportion X", fontsize=16)
            else:
                # norm = mpl.colors.Normalize(vmin=min(prop_y_per_node.values()), vmax=max(prop_y_per_node.values()))
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                norm_opinions = norm(np.array(list(prop_y_per_node.values())))
                norm_opinions = np.array(list(prop_y_per_node.values()))
                c_label = r'$Y_i/(X_i+Y_i)$'
                # axes[1].set_title(f"Proportion Y", fontsize=16)
            nx.draw_networkx_edges(G, pos, ax=axes[1], alpha=0.6, edge_color='black', node_size=0, arrows=False)
            network_2 = nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=norm_opinions, cmap=grey_cmap, node_size=size_scale[0])
            cbar2 = fig.colorbar(network_2, ax=axes[1], pad=0.04, orientation='horizontal')
            cbar2.set_label(label=c_label)

            node_sizes = np.array(list(node_sizes.values()))
            node_sizes_norm = (node_sizes - np.min(node_sizes)) / (np.max(node_sizes)-np.min(node_sizes))
            node_sizes_norm = 0.1 + (node_sizes_norm * (1-0.1))
            prop_x_per_node = np.array(list(prop_x_per_node.values()))
            total_per_node = np.array(list(total_per_node.values()))
            colors = [('#e09b24','#5d81b4')[int(prop_x>0.5)] for prop_x in prop_x_per_node]
            nx.draw_networkx_edges(G, pos, ax=axes[2], alpha=0.6, edge_color='black', node_size=0, arrows=False)
            nx.draw_networkx_nodes(G, pos, ax=axes[2], node_color=colors, node_size=node_sizes_norm*size_scale[1])
            network_3 = nx.draw_networkx_nodes(G, pos, ax=axes[2], node_color=total_per_node, cmap=plt.cm.viridis, node_size=0)
            cbar3 = fig.colorbar(network_3, ax=axes[2], pad=0.04, orientation='horizontal')
            cbar3.set_label(label=r"$X_i+Y_i$ Opinion Majority $X>Y$ $X<Y$")

            axes[0].axis("off")
            axes[1].axis("off")
            axes[2].axis("off")

            axes[0].set_aspect('equal')
            axes[1].set_aspect('equal')
            axes[2].set_aspect('equal')

            axes[0].text(-0.13, 1.0, "(a)", transform=axes[0].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
            axes[1].text(-0.13, 1.0, "(b)", transform=axes[1].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
            axes[2].text(-0.13, 1.0, "(c)", transform=axes[2].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

            if path:
                filename = f'{path}/network_overview.svg'
                plt.savefig(filename, dpi=900, bbox_inches="tight")
            else:
                plt.show()

    # return filenames



def draw_graph_dist(G, t, solution, frames, size_scale=[10,75], layout="RGG", path=None, 
                    seed=15, laplace_eigv=1, prop="X", weights=None, kappa_0=False, gillespie=None):
    '''
    Draw the network at different time points with pie charts representing the proportions of the two opinions.

    Parameters:
    G            : networkx graph
    t            : time array for the integration
    solution     : array of shape (len(t), 2*num_patches) with the time evolution of opinions X and Y
    frames       : list of time frame indices to draw
    size_scale   : scaling factor for the node sizes
    layout       : layout type for the network visualization ("RGG" or "spring")
    path         : path to save the figures (if None, figures are not saved)
    seed         : seed for the spring layout (if layout is "spring")
    laplace_eigv : index of the Laplacian eigenvector to use for node coloring (if int), 
                   or threshold for unstable eigenvalues (if float)
    prop         : which opinion proportion to use for coloring ("X" or "Y")
    weights      : weights for the unstable eigenvectors (if laplace_eigv is a float)
    kappa_0      : whether to exclude the zero eigenvalue from the unstable eigenvector calculation (default: False)
    gillespie    : if not None, will plot probability of a node being flipped in stochastic simulations
    '''

    if layout == "RGG":
        pos = nx.get_node_attributes(G, "pos")
    else:
        pos = nx.spring_layout(G, seed=seed)
       #  pos = nx.kamada_kawai_layout(G)

    filenames = []
    patches = len(G.nodes)

    # Draw eigenvector of laplacian
    L = nx.laplacian_matrix(G).toarray()
    if type(laplace_eigv) == int:
        _, eigvecs = np.linalg.eigh(L)
        eig_vector = eigvecs[:, laplace_eigv]
    else:
        eigvals, eigvecs = np.linalg.eigh(L)
        unstable_indices = np.where(eigvals<laplace_eigv)[0][not kappa_0:]
        # eig_vector = np.sum(eigvecs[:, unstable_indices], axis=1)
        eig_vector = np.dot(eigvecs[:, unstable_indices], weights[not kappa_0:])
    norm = mpl.colors.Normalize(vmin=np.min(eig_vector), vmax=np.max(eig_vector))
    norm_eigenvector = norm(eig_vector)
    if path:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_aspect('equal')
        nx.draw(G, pos, node_color=norm_eigenvector, cmap=plt.cm.viridis, edge_color='gray', ax=ax)
        Path(path).mkdir(parents=True, exist_ok=True)
        full_path = Path(path) / "laplacian_ev.svg"
        plt.savefig(full_path, dpi=900, bbox_inches="tight")
        plt.close()
    else:
        print("No path specified, figures not saved.")
        

    # Loop over specified time frames
    for idx in frames:
        t_idx = t[idx] # time point
        # Calculate x and y abundacne and proportions
        x_abundance = solution[idx,:patches]
        y_abundance = solution[idx,patches:]
        x_per_node = {i: x_abundance[i] for i in G.nodes()}
        y_per_node = {i: y_abundance[i] for i in G.nodes()}
        total_per_node = {i: x_per_node[i] + y_per_node[i] for i in G.nodes()}
        total_x = np.sum(x_abundance)
        total_y = np.sum(y_abundance)
        prop_x_per_node = {i: x_per_node[i] / total_per_node[i] for i in G.nodes()}
        prop_y_per_node = {i: y_per_node[i] / total_per_node[i] for i in G.nodes()}
        # derive node size
        max_val = max(total_per_node.values())
        # max_val = np.mean(list(total_per_node.values()))
        # node_sizes = [(total_per_node[i] / (max_val*size_scale)) for i in G.nodes()] 
        node_sizes = {i: (total_per_node[i] / (max_val*size_scale[1])) for i in G.nodes()}

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_aspect('equal')
        nx.draw_networkx_edges(G, pos, alpha=0.8, ax=ax)
        ax = create_pie_chart(G, pos, ax, prop_x_per_node, prop_y_per_node, node_sizes, total_x, total_y)
        plt.axis("off")

        if path:
            filename = f'{path}/idx_{idx}_t_{round(t_idx,2)}.png'
            plt.savefig(filename, dpi=900, bbox_inches="tight")
            plt.close()
        else:
            plt.close()

        if idx == frames[-1]:
            colors = ["#AFAFAFFF", "#727272FF", "#000000FF"]
            grey_cmap = mcolors.LinearSegmentedColormap.from_list("custom_orange", colors)
            fig, axes = plt.subplots(1,3,figsize=((3+(3/8))*2.3, 5))
            # Draw eigenvector of laplacian
            custom_cmap = mcolors.LinearSegmentedColormap.from_list("BlueOrangeBlue", ["#5d81b4", "#e09b24", "#5d81b4"])
            nx.draw_networkx_edges(G, pos, ax=axes[2], alpha=0.6, edge_color='black', node_size=0, arrows=False)
            network_1 = nx.draw_networkx_nodes(G, pos, ax=axes[2], node_color=norm_eigenvector, cmap=custom_cmap, node_size=size_scale[0])
            cbar1 = fig.colorbar(network_1, ax=axes[2], pad=0.04, orientation='horizontal')
            # cbar1.set_label(label=r"Aggregated $\mathbf{v}_i$ for $\kappa_i<\tilde{\kappa}$")
            # cbar1.set_label(label=r"$\sum_{i:m(\kappa_i)>0}m(\kappa_i)\mathbf{v}_i$")
            cbar1.set_label(label=r"$s$")
            # Draw network distribution at final time point
            if gillespie == None:
                if prop=="X":
                    # norm = mpl.colors.Normalize(vmin=min(prop_x_per_node.values()), vmax=max(prop_x_per_node.values()))
                    norm = mpl.colors.Normalize(vmin=0, vmax=1)
                    norm_opinions = norm(np.array(list(prop_x_per_node.values())))
                    norm_opinions = np.array(list(prop_x_per_node.values()))
                    c_label = r'$X_i/(X_i+Y_i)$'
                    # axes[1].set_title(f"Proportion X", fontsize=16)
                else:
                    # norm = mpl.colors.Normalize(vmin=min(prop_y_per_node.values()), vmax=max(prop_y_per_node.values()))
                    norm = mpl.colors.Normalize(vmin=0, vmax=1)
                    norm_opinions = norm(np.array(list(prop_y_per_node.values())))
                    norm_opinions = np.array(list(prop_y_per_node.values()))
                    c_label = r'$Y_i/(X_i+Y_i)$'
                    # axes[1].set_title(f"Proportion Y", fontsize=16)
                nx.draw_networkx_edges(G, pos, ax=axes[1], alpha=0.6, edge_color='black', node_size=0, arrows=False)
                network_2 = nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=norm_opinions, cmap=grey_cmap, node_size=size_scale[0])
                cbar2 = fig.colorbar(network_2, ax=axes[1], pad=0.04, orientation='horizontal')
                cbar2.set_label(label=c_label)
            else: # Draw gillespie distribution
                custom_cmap = mcolors.LinearSegmentedColormap.from_list("BlueToOrange", ["#e09b24", "#5d81b4"])
                norm = mpl.colors.Normalize(vmin=np.min(gillespie[0]), vmax=np.max(gillespie[0]))
                norm_flipped = norm(gillespie[0])
                nx.draw_networkx_edges(G, pos, ax=axes[1], alpha=0.6, edge_color='black', node_size=0, arrows=False)
                network_1 = nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=norm_flipped, cmap=custom_cmap, node_size=gillespie[1])
                cbar1 = fig.colorbar(network_1, pad=0.04, orientation='horizontal')
                cbar1.set_label(label=r"P(Minority Pocket)")

            node_sizes = np.array(list(node_sizes.values()))
            node_sizes_norm = (node_sizes - np.min(node_sizes)) / (np.max(node_sizes)-np.min(node_sizes))
            node_sizes_norm = 0.1 + (node_sizes_norm * (1-0.1))
            prop_x_per_node = np.array(list(prop_x_per_node.values()))
            total_per_node = np.array(list(total_per_node.values()))
            colors = [('#e09b24','#5d81b4')[int(prop_x>0.5)] for prop_x in prop_x_per_node]
            nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.6, edge_color='black', node_size=0, arrows=False)
            nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=colors, node_size=node_sizes_norm*size_scale[1])
            network_3 = nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=total_per_node, cmap=plt.cm.viridis, node_size=0)
            cbar3 = fig.colorbar(network_3, ax=axes[0], pad=0.04, orientation='horizontal')
            cbar3.set_label(label=r"$X_i+Y_i$ Opinion Majority $X>Y$ $X<Y$")

            axes[0].axis("off")
            axes[1].axis("off")
            axes[2].axis("off")

            axes[0].set_aspect('equal')
            axes[1].set_aspect('equal')
            axes[2].set_aspect('equal')

            axes[0].text(-0.13, 1.0, "(a)", transform=axes[0].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
            axes[1].text(-0.13, 1.0, "(b)", transform=axes[1].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
            axes[2].text(-0.13, 1.0, "(c)", transform=axes[2].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

            if path:
                filename = f'{path}/network_overview.svg'
                plt.savefig(filename, dpi=900, bbox_inches="tight")
            else:
                plt.show()

    # return filenames




def gillespie_algorithm(cycles, steps, z0, A, local_params):

    # Extract model parameters
    node_degrees = np.sum(A, axis=1)
    num_patches = A.shape[0]
    alpha, beta, gamma, mu_x, mu_y, _ = local_params
    # Set up holder arrays
    x_result = np.zeros((cycles, steps + 1, num_patches))
    y_result = np.zeros((cycles, steps + 1, num_patches))
    t_result = np.zeros((cycles, steps + 1))

    for cycle_idx in range(cycles):
        # Reset Time and State
        x = z0[:num_patches].copy()
        y = z0[num_patches:].copy()
        t = 0.0

        x_result[cycle_idx, 0] = x
        y_result[cycle_idx, 0] = y
        t_result[cycle_idx, 0] = t

        for step_idx in range(steps):
            
            # Calculate Rates
            r_flip_x = x * alpha
            r_flip_y = y * beta
            r_adj_x = x * (y**2) * gamma
            r_adj_y = (x**2) * y * gamma
            r_dif_x = x * node_degrees * mu_x
            r_dif_y = y * node_degrees * mu_y
            # Cumulative sums for selection of next event
            all_rates = np.concatenate([
                r_flip_x, r_flip_y, r_adj_x, r_adj_y, r_dif_x, r_dif_y
            ])
            cumsum_rates = np.cumsum(all_rates)
            r_total = cumsum_rates[-1]
            # Break when rates are zero
            if r_total < 1e-12:
                x_result[cycle_idx, step_idx+1:] = x_result[cycle_idx, step_idx]
                y_result[cycle_idx, step_idx+1:] = y_result[cycle_idx, step_idx]
                t_result[cycle_idx, step_idx+1:] = t_result[cycle_idx, step_idx]
                break

            # Draw next time step
            t += np.random.exponential(scale=1/r_total)            
            # Find next event
            u = np.random.rand() * r_total # Draw event selection number
            event_idx = np.searchsorted(cumsum_rates, u)
            event_idx = min(event_idx, len(all_rates) - 1)
            # Execute event
            if event_idx < num_patches: # Flip X->Y
                node = event_idx
                x[node] -= 1; y[node] += 1
            elif event_idx < 2*num_patches: # Flip Y->X
                node = event_idx - num_patches
                y[node] -= 1; x[node] += 1
            elif event_idx < 3*num_patches: # Adj X->Y
                node = event_idx - 2*num_patches
                x[node] -= 1; y[node] += 1
            elif event_idx < 4*num_patches: # Adj Y->X
                node = event_idx - 3*num_patches
                y[node] -= 1; x[node] += 1
            elif event_idx < 5*num_patches: # Dif X
                node = event_idx - 4*num_patches
                neighbors = np.nonzero(A[node])[0]
                if len(neighbors) > 0:
                    dest = np.random.choice(neighbors)
                    x[node] -= 1; x[dest] += 1
            else: # Dif Y
                node = event_idx - 5*num_patches
                neighbors = np.nonzero(A[node])[0]
                if len(neighbors) > 0:
                    dest = np.random.choice(neighbors)
                    y[node] -= 1; y[dest] += 1

            # 4. Record State
            x_result[cycle_idx, step_idx+1] = x
            y_result[cycle_idx, step_idx+1] = y
            t_result[cycle_idx, step_idx+1] = t

    return t_result, x_result, y_result


            
    