import numpy as np
import sympy as sp


# ------ Opinion Model ODEs ------ #

def f(x,y,alpha,beta,gamma):
    "Opinion Change Term of X, Compare Eq. 1a in Paper"
    return -alpha*x + beta*y + gamma*(x**2)*y - gamma*x*(y**2)

def g(x,y,alpha,beta,gamma):
    "Opinion Change Term of Y, Compare Eq. 1b in Paper"
    return alpha*x - beta*y - gamma*(x**2)*y + gamma*x*(y**2)

def E_x(x, mu_x):
    "Diffusion Term of X, Compare Eq. 2a in Paper"
    return x*mu_x

def E_y(y, mu_y):
    "Diffusion Term of Y, Compare Eq. 2b in Paper"
    return y*mu_y

def model(z, t, A, alpha, beta, gamma, mu_x, mu_y):
    '''
    System of ODEs for the opinion model on a network

    z       : initial conditions
    A       : patch network adjacency matrix
    alpha   : spontaneous flipping parameter from X to Y
    beta    : spontaneous flipping parameter from Y to X
    gamma   : interaction based flipping parameter
    mu_x    : diffusion rate of opinion X
    mu_y    : diffusion rate of opinion Y
    t       : time (not used, but required for odeint)

    Returns:
    [x,y]   : array of lenght 2*num_patches with time derivatives of opinions X and Y
    '''

    omega = np.sum(A, axis=1) # degree of each patch

    num_patches = len(z) // 2
    x_ = z[:num_patches]
    y_ = z[num_patches:]

    x = f(x_,y_,alpha, beta, gamma) - omega*E_x(x_, mu_x) + np.sum(A*E_x(x_,mu_x),axis=1)
    y = g(x_,y_,alpha, beta, gamma) - omega*E_y(y_, mu_y) + np.sum(A*E_y(y_,mu_y),axis=1)

    return np.concatenate([x,y])

def one_patch_model(x, alpha, beta, gamma, M):
    '''
    Differential equation for a single patch expressed only in terms of x, compare Eq. (3) in Paper.

    Parameters:
    x       : density of opinion X
    alpha   : spontaneous flipping parameter from X to Y
    beta    : spontaneous flipping parameter from Y to X
    gamma   : interaction based flipping parameter
    M       : total population per patch

    Returns:
    x_dot   : time derivative of x
    '''

    x_dot = -alpha*x + beta*(M-x) + gamma*x**2*(M-x) - gamma*x*(M-x)**2

    return x_dot


def roots_symbolic(alpha, beta, gamma, M):
    '''
    Find the roots of the dynamical system on a single patch.

    Parameters:
    alpha   : spontaneous flipping parameter from X to Y
    beta    : spontaneous flipping parameter from Y to X
    gamma   : interaction based flipping parameter
    M       : total population per patch

    Returns:
    real_roots  : list of real parts of the roots
    '''
    
    # Coefficients of the polynomial
    a = -2*gamma
    b = (gamma*M+2*gamma*M)
    c = (-alpha-beta-gamma*M**2)
    d = beta*M

    x = sp.symbols('x')
    roots = sp.solve(a*x**3 + b*x**2 + c*x + d, x)
    real_roots = []
    for x in roots:
        if abs(sp.im(x).evalf()) < 10**(-15):
            x = sp.re(x)
        if x.is_real:
            x = float(x)
            real_roots.append(x)
    
    return real_roots