import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

# Parameters
omega0 = 1.0       # Natural frequency
h = 1.0            # Modulation strength (0 to 1)
a = 0.5            # Weierstrass parameter (0 < a < 1)
b = 7              # Weierstrass parameter (b odd, ab > 1 + 3π/2)
n_terms = 30       # Truncated series terms
t_max = 100        # Simulation time
dt = 0.01          # Time step

# Weierstrass function (Numba-optimized)
@njit
def weierstrass(t, a, b, n_terms):
    W = 0.0
    for n in range(n_terms):
        W += a**n * np.cos(b**n * t)
    return W

# Modified solver to return W(t) values
@njit
def solve_oscillator(t, omega0, h, a, b, n_terms, dt):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    W_vals = np.zeros(n_steps)
    x[0], v[0] = 1.0, 0.0  # Initial conditions
    
    for i in range(1, n_steps):
        W = weierstrass(t[i-1], a, b, n_terms)
        W_vals[i-1] = W
        x[i] = x[i-1] + v[i-1] * dt - 0.5 * (omega0**2 * (1 + h * W) * x[i-1]) * dt**2
        W_new = weierstrass(t[i], a, b, n_terms)
        v[i] = v[i-1] - 0.5 * (omega0**2 * (1 + h * W) * x[i-1] + omega0**2 * (1 + h * W_new) * x[i]) * dt
    
    W_vals[-1] = weierstrass(t[-1], a, b, n_terms)
    return x, v, W_vals

# Time array
t = np.arange(0, t_max, dt)

# Solve system
x, v, W = solve_oscillator(t, omega0, h, a, b, n_terms, dt)

# Energy calculation
E = 0.5 * (v**2 + omega0**2 * (1 + h * W) * x**2)

def plot_phase_space(x, v, W, h):
    """Adaptive visualization switching between direct/density plots"""
    plt.figure(figsize=(12, 5))
    
    if h == 0:
        # Direct plot for h=0
        plt.subplot(1, 2, 1)
        plt.plot(x, v, 'b', linewidth=0.5)
        plt.axis('equal')
        plt.title('Phase Space (Direct Plot)')
    else:
        # Density plot for h>0
        plt.subplot(1, 2, 1)
        hist, xedges, vedges = np.histogram2d(x, v, bins=300, density=True)
        plt.imshow(hist.T, extent=[xedges[0], xedges[-1], vedges[0], vedges[-1]], 
                   origin='lower', aspect='equal', cmap='inferno', norm=LogNorm())
        plt.colorbar(label='log(Density)')
        plt.title('Phase Space Density')
    
    # 3D phase space for all h
    plt.subplot(1, 2, 2, projection='3d')
    skip = max(1, len(x)//10000)  # Downsample for performance
    ax = plt.gca()
    ax.scatter(x[::skip], v[::skip], W[::skip], c=t[::skip], cmap='viridis', s=1)
    ax.set_xlabel('x'); ax.set_ylabel('v'); ax.set_zlabel('W(t)')
    plt.title('3D Phase Space')
    plt.tight_layout()

def plot_diagnostics(t, x, E):
    """Time series and energy plots"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, x, 'r')
    plt.xlabel('Time'); plt.ylabel('x(t)')
    plt.title('Position Time Series')
    
    plt.subplot(1, 2, 2)
    plt.plot(t, E, 'g')
    plt.ylim(0.99*np.min(E), 1.01*np.max(E))
    plt.xlabel('Time'); plt.ylabel('Energy')
    plt.title('Total Energy')
    plt.tight_layout()

# Generate all plots
plot_phase_space(x, v, W, h)
plot_diagnostics(t, x, E)
plt.show()
