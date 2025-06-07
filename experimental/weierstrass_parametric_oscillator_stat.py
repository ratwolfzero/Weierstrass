import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from numba import njit

@njit
def weierstrass(t, a, b, n_terms):
    total = 0.0
    for n in range(n_terms):
        total += a**n * np.cos(np.pi * b**n * t)
    return total

@njit
def solve_oscillator(t, omega0, h, a, b, n_terms, dt):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    x[0] = 1.0
    W = np.zeros(n_steps)
    for i in range(1, n_steps):
        W[i-1] = weierstrass(t[i-1], a, b, n_terms)
        omega_sq = omega0**2 * (1 + h * W[i-1])
        x[i] = x[i-1] + v[i-1]*dt - 0.5 * omega_sq * x[i-1] * dt**2
        omega_sq_next = omega0**2 * (1 + h * weierstrass(t[i], a, b, n_terms))
        v[i] = v[i-1] - 0.5 * (omega_sq + omega_sq_next) * x[i] * dt
    return x, v, W

@njit
def compute_lyapunov(t, omega0, h, a, b, n_terms, dt, delta0=1e-8):
    n_steps = len(t)
    x = 1.0
    x_perturbed = x + delta0
    v = 0.0
    v_perturbed = v
    sum_log_divergence = 0.0

    for i in range(1, n_steps):
        W = weierstrass(t[i-1], a, b, n_terms)

        x_new = x + v * dt - 0.5 * omega0**2 * (1 + h * W) * x * dt**2
        v_new = v - 0.5 * omega0**2 * (1 + h * W) * (x + x_new) * dt

        x_perturbed_new = x_perturbed + v_perturbed * dt - 0.5 * omega0**2 * (1 + h * W) * x_perturbed * dt**2
        v_perturbed_new = v_perturbed - 0.5 * omega0**2 * (1 + h * W) * (x_perturbed + x_perturbed_new) * dt

        dx = x_perturbed_new - x_new
        dv = v_perturbed_new - v_new
        delta = np.sqrt(dx*dx + dv*dv)
        if delta == 0.0:
            continue
        sum_log_divergence += np.log(delta / delta0)
												 
        scale = delta0 / delta
        x_perturbed_new = x_new + dx * scale
        v_perturbed_new = v_new + dv * scale

        x, v = x_new, v_new
        x_perturbed, v_perturbed = x_perturbed_new, v_perturbed_new

    return sum_log_divergence / (t[-1] - t[0])

def get_maxima(x, t, discard_ratio=0.5):
    idx_peaks = argrelextrema(x, np.greater)[0]
    discard = int(discard_ratio * len(t))
    return x[idx_peaks[idx_peaks > discard]]

def plot_bifurcation_and_lyapunov(h_vals, t, omega0, a, b, n_terms, dt):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Bifurcation Diagram on left subplot
    for h in h_vals:
        x, v, W = solve_oscillator(t, omega0, h, a, b, n_terms, dt)
        peaks = get_maxima(x, t)
        axs[0].plot([h]*len(peaks), peaks, ',k', alpha=0.5)
    axs[0].set_xlabel('h (modulation strength)')
    axs[0].set_ylabel('x(t) maxima')
    axs[0].set_title('Bifurcation Diagram')
    axs[0].grid(True)

    # Lyapunov Exponent on right subplot
    exponents = []
    for h in h_vals:
        lce = compute_lyapunov(t, omega0, h, a, b, n_terms, dt)
        exponents.append(lce)
    axs[1].plot(h_vals, exponents, 'r-')
    axs[1].axhline(0, color='gray', linestyle='--')
    axs[1].set_xlabel('h (modulation strength)')
    axs[1].set_ylabel('Lyapunov Exponent')
    axs[1].set_title('Lyapunov Exponent vs h')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
h_vals = np.linspace(0, 1.0, 1000)
t = np.arange(0, 100, 0.001)
om = 1.0

plot_bifurcation_and_lyapunov(h_vals, t, omega0=om, a=0.5, b=7, n_terms=30, dt=0.01)


