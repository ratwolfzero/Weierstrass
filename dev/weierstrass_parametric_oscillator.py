import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.colors import LogNorm

# Parameters
omega0 = 1.0       # Natural frequency
h = 0.2            # Modulation strength
a = 0.5            # Weierstrass parameter (0 < a < 1)
b = 7              # Weierstrass parameter (b odd, ab > 1 + 3π/2)
n_terms = 30       # Truncated series terms
t_max = 1000       # Simulation time
dt = 0.01          # Time step

# Weierstrass function (corrected definition)
@njit
def weierstrass(t, a, b, n_terms):
    W = 0.0
    for n in range(n_terms):
        W += a**n * np.cos(b**n * t)
    return W

# Parametric oscillator solver (velocity Verlet)
@njit
def solve_oscillator(t, omega0, h, a, b, n_terms, dt):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    x[0], v[0] = 1.0, 0.0  # Initial conditions
    
    for i in range(1, n_steps):
        # Current Weierstrass value
        W = weierstrass(t[i-1], a, b, n_terms)
        
        # Verlet integration
        x[i] = x[i-1] + v[i-1] * dt - 0.5 * (omega0**2 * (1 + h * W) * x[i-1]) * dt**2
        
        # Update velocity with new position
        W_new = weierstrass(t[i], a, b, n_terms)
        v[i] = v[i-1] - 0.5 * (
            omega0**2 * (1 + h * W) * x[i-1] + 
            omega0**2 * (1 + h * W_new) * x[i]
        ) * dt
    return x, v

# Time array
t = np.arange(0, t_max, dt)

# Solve
x, v = solve_oscillator(t, omega0, h, a, b, n_terms, dt)

# Plot phase space density
def plot_phase_space(x, v, bins=1000):
    plt.figure(figsize=(10, 8))
    hist, xedges, vedges = np.histogram2d(x, v, bins=bins, density=True)
    plt.imshow(hist.T, extent=[xedges[0], xedges[-1], vedges[0], vedges[-1]], 
               origin='lower', aspect='auto', cmap='inferno', norm=LogNorm())
    plt.colorbar(label='Density (log scale)')
    plt.xlabel('Position (x)')
    plt.ylabel('Velocity (v)')
    plt.title('Phase Space Density (Weierstrass-Modulated Oscillator)')
    plt.show()

plot_phase_space(x, v)

# Plot time series
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, [weierstrass(tt, a, b, n_terms) for tt in t], 'b', alpha=0.7)
plt.title('Weierstrass Modulation Signal $W(t)$')

plt.subplot(2, 1, 2)
plt.plot(t, x, 'r', alpha=0.7)
plt.title('Oscillator Position $x(t)$')
plt.xlabel('Time')
plt.tight_layout()
plt.show()
