import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.colors import LogNorm

# Parameters
omega0 = 1.0       # Natural frequency
h = 0.3            # Modulation strength (0 to 1)
a = 0.5            # Weierstrass amplitude decay (0 < a < 1)
b = 7              # Weierstrass frequency scaling (odd integer ≥3)
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

# Oscillator solver (returns x, v, W_vals)
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

# Visualization
def plot_results(t, x, v, W, h):
    fig = plt.figure(figsize=(15, 5))
    
    # Time Series
    plt.subplot(1, 3, 1)
    plt.plot(t, x, 'r')
    plt.xlabel('Time')
    plt.ylabel('Position (x)')
    plt.title(f'$x(t)$, $h={h}$')
    
    # Phase Space
    plt.subplot(1, 3, 2)
    if h == 0:
        plt.plot(x, v, 'b', linewidth=0.5)
        plt.axis('equal')
    else:
        hist, xedges, vedges = np.histogram2d(x, v, bins=300, density=True)
        plt.imshow(hist.T, extent=[xedges[0], xedges[-1], vedges[0], vedges[-1]], 
                   origin='lower', aspect='equal', cmap='inferno', norm=LogNorm())
        plt.colorbar(label='Density')
    plt.xlabel('Position (x)')
    plt.ylabel('Velocity (v)')
    plt.title(f'Phase Space, $h={h}$')
    
    # Energy
    plt.subplot(1, 3, 3)
    plt.plot(t, E, 'g')
    plt.ylim(0.99*np.min(E), 1.01*np.max(E))
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Total Energy')
    
    plt.tight_layout()
    plt.show()

# Plot results
plot_results(t, x, v, W, h)
