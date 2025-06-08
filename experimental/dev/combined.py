import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

# Parameters
omega0 = 1.0       # Natural frequency
h_k = 0.7          # Modulation strength for stiffness
h_d = 0.7          # Modulation strength for damping
a_k = 0.7          # Weierstrass parameter for stiffness
a_d = 0.73         # Weierstrass parameter for damping
b = 9              # Common b for both (must be odd)
n_terms = 30       # Series terms
t_max = 100
dt = 0.01

# Weierstrass function (Numba-optimized)
@njit
def weierstrass(t, a, b, n_terms):
    W = 0.0
    for n in range(n_terms):
        W += a**n * np.cos(b**n * t)
    return W

@njit
def solve_modulated_oscillator(t, omega0, h_k, h_d, a_k, a_d, b, n_terms, dt):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    Wk_vals = np.zeros(n_steps)
    Wd_vals = np.zeros(n_steps)

    x[0], v[0] = 1.0, 0.0

    for i in range(1, n_steps):
        if h_k == 0.0:
            Wk = 0.0
            Wk_new = 0.0
        else:
            Wk = weierstrass(t[i - 1], a_k, b, n_terms)
            Wk_new = weierstrass(t[i], a_k, b, n_terms)

        if h_d == 0.0:
            Wd = 0.0
        else:
            Wd = weierstrass(t[i - 1], a_d, b, n_terms)

        Wk_vals[i - 1] = Wk
        Wd_vals[i - 1] = Wd

        # Effective stiffness and damping
        stiffness = omega0 ** 2 * (1 + h_k * Wk)
        damping = h_d * Wd  # Linear damping with fractal envelope

        # Velocity Verlet-like update
        x[i] = x[i - 1] + v[i - 1] * dt - 0.5 * stiffness * x[i - 1] * dt**2
        v_half = v[i - 1] - 0.5 * stiffness * x[i - 1] * dt
        v[i] = (v_half - damping * v_half * dt) - 0.5 * stiffness * x[i] * dt

    # Final values
    Wk_vals[-1] = weierstrass(t[-1], a_k, b, n_terms) if h_k != 0.0 else 0.0
    Wd_vals[-1] = weierstrass(t[-1], a_d, b, n_terms) if h_d != 0.0 else 0.0

    return x, v, Wk_vals, Wd_vals

# Time array
t = np.arange(0, t_max, dt)

# Solve system with separate stiffness and damping modulation
x, v, Wk, Wd = solve_modulated_oscillator(t, omega0, h_k, h_d, a_k, a_d, b, n_terms, dt)

# Energy calculation
E = 0.5 * (v**2 + omega0**2 * (1 + h_k * Wk) * x**2)

# Visualization
def plot_combined(t, x, v, Wk, Wd, E, h_k, h_d):
    plt.figure(figsize=(16, 10))

    # 1. Position Time Series
    plt.subplot(2, 2, 1)
    plt.plot(t, x, 'r', linewidth=0.8)
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title('Position Time Series')
    plt.grid(alpha=0.3)

    # 2. Total Energy
    plt.subplot(2, 2, 2)
    plt.plot(t, E, 'g', linewidth=0.8)
    plt.ylim(0.99*np.min(E), 1.01*np.max(E))
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Total Energy')
    plt.grid(alpha=0.3)

    # 3. Phase Space Density
    plt.subplot(2, 2, 3)
    if h_k == 0 and h_d == 0:
        plt.plot(x, v, 'b', linewidth=0.5)
        plt.axis('equal')
        plt.title('Phase Space (Direct Plot)')
    else:
        hist, xedges, vedges = np.histogram2d(x, v, bins=300, density=True)
        plt.imshow(hist.T, extent=[xedges[0], xedges[-1], vedges[0], vedges[-1]],
                   origin='lower', aspect='equal', cmap='inferno', norm=LogNorm())
        plt.colorbar(label='log(Density)')
        plt.title('Phase Space Density')
    plt.xlabel('x')
    plt.ylabel('v')

    # 4. 3D Phase Space
    ax = plt.subplot(2, 2, 4, projection='3d')
    skip = max(1, len(x) // 10000)
    ax.scatter(x[::skip], v[::skip], Wk[::skip],
               c=t[::skip], cmap='plasma', s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_zlabel('Wk(t)')
    plt.title('3D Phase Space (with Stiffness Modulation)')
    plt.tight_layout()
    plt.show()

# Plot results
plot_combined(t, x, v, Wk, Wd, E, h_k, h_d)
