import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from numba import njit


@njit
def precompute_powers(a, b, n_terms):
    a_powers = np.empty(n_terms)
    b_powers = np.empty(n_terms)
    a_val = 1.0
    b_val = 1.0
    for n in range(n_terms):
        a_powers[n] = a_val
        b_powers[n] = b_val
        a_val *= a
        b_val *= b
    return a_powers, b_powers


@njit
def precompute_weierstrass(t, a_powers, b_powers):
    n_terms = len(a_powers)
    W = np.empty(len(t))
    for i in range(len(t)):
        total = 0.0
        for n in range(n_terms):
            total += a_powers[n] * np.cos(np.pi * b_powers[n] * t[i])
        W[i] = total
    return W


@njit
def solve_oscillator_with_W(t, omega0, h, W, dt):
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    x[0] = 1.0
    v[0] = 0.0
    for i in range(1, n_steps):
        omega_sq = omega0**2 * (1 + h * W[i-1])
        x[i] = x[i-1] + v[i-1]*dt - 0.5 * omega_sq * x[i-1] * dt**2
        omega_sq_next = omega0**2 * (1 + h * W[i])
        v[i] = v[i-1] - 0.5 * (omega_sq + omega_sq_next) * x[i] * dt
    return x, v


@njit
def compute_lyapunov_with_W(t, omega0, h, W, dt, delta0=1e-8):
    n_steps = len(t)
    x = 1.0
    x_perturbed = x + delta0
    v = 0.0
    v_perturbed = v
    sum_log_divergence = 0.0

    for i in range(1, n_steps):
        omega_sq = omega0**2 * (1 + h * W[i-1])
        x_new = x + v * dt - 0.5 * omega_sq * x * dt**2
        v_new = v - 0.5 * omega_sq * (x + x_new) * dt
        x_perturbed_new = x_perturbed + v_perturbed * \
            dt - 0.5 * omega_sq * x_perturbed * dt**2
        v_perturbed_new = v_perturbed - 0.5 * omega_sq * \
            (x_perturbed + x_perturbed_new) * dt

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
    if len(idx_peaks) == 0:
        return np.array([])
    discard_time = t[0] + discard_ratio * (t[-1] - t[0])
    filtered_peaks = [x[i] for i in idx_peaks if t[i] > discard_time]
    return np.array(filtered_peaks)


def plot_bifurcation_and_lyapunov(h_vals, t, omega0, a, b, n_terms, dt):
    a_powers, b_powers = precompute_powers(a, b, n_terms)
    W = precompute_weierstrass(t, a_powers, b_powers)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Precompute Lyapunov exponents for coloring
    exponents = np.array([compute_lyapunov_with_W(t, omega0, h, W, dt) for h in h_vals])
    
    # Create a diverging colormap: blue (negative) -> white (0) -> red (positive)
    cmap = plt.cm.coolwarm  
    
    # Normalize exponents with 0 at the center
    max_abs = max(abs(exponents.min()), abs(exponents.max()))
    norm = plt.Normalize(vmin=-max_abs, vmax=max_abs)

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Plot bifurcation diagram with colored points
    for i, h in enumerate(h_vals):
        x, v = solve_oscillator_with_W(t, omega0, h, W, dt)
        peaks = get_maxima(x, t)
        if len(peaks) > 0:
            color = cmap(norm(exponents[i]))
            axs[0].plot([h]*len(peaks), peaks, '.', color=color, markersize=1, alpha=0.7)
    
    axs[0].set_xlabel('h (modulation strength)')
    axs[0].set_ylabel('x(t) maxima')
    axs[0].set_title('Bifurcation Diagram (colored by Lyapunov exponent)')
    axs[0].grid(True)
    
    # Add colorbar to the bifurcation plot
    cbar = fig.colorbar(sm, ax=axs[0])
    cbar.set_label('Lyapunov Exponent')

    # Plot Lyapunov exponents
    axs[1].plot(h_vals, exponents, 'k-', linewidth=1)  # Now in black for clarity
    axs[1].axhline(0, color='gray', linestyle='--')
    
    # Color the area under the curve based on exponent sign
    axs[1].fill_between(h_vals, exponents, where=(exponents>=0), 
                       color='red', alpha=0.3, interpolate=True)
    axs[1].fill_between(h_vals, exponents, where=(exponents<0), 
                       color='blue', alpha=0.3, interpolate=True)
    
    axs[1].set_xlabel('h (modulation strength)')
    axs[1].set_ylabel('Lyapunov Exponent')
    axs[1].set_title('Lyapunov Exponent vs h')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# Example usage
omega0 = 1.0
a = 0.5
b = 7
n_terms = 30
dt = 0.001
T_max = 1000
t = np.arange(0, T_max, dt)
h_vals = np.linspace(0.0, 1.0, 1000)

plot_bifurcation_and_lyapunov(h_vals, t, omega0, a, b, n_terms, dt)
