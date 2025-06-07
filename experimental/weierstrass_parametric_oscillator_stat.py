import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from numba import njit

@njit
def weierstrass(t, a, b, n_terms):
    """
    Calculates the Weierstrass function at a given time t.

    Args:
        t (float): The time at which to evaluate the function.
        a (float): Parameter of the Weierstrass function.
        b (float): Parameter of the Weierstrass function.
        n_terms (int): Number of terms to sum in the series.

    Returns:
        float: The value of the Weierstrass function.
    """
    total = 0.0
    for n in range(n_terms):
        total += a**n * np.cos(np.pi * b**n * t)
    return total

@njit
def solve_oscillator(t, omega0, h, a, b, n_terms, dt):
    """           
    Solves the modified harmonic oscillator equation using a numerical method.

    Args:
        t (np.ndarray): Array of time points.
        omega0 (float): Natural frequency of the oscillator.
        h (float): Modulation strength.
        a (float): Parameter for Weierstrass function.
        b (float): Parameter for Weierstrass function.
        n_terms (int): Number of terms for Weierstrass function.
        dt (float): Time step size.

    Returns:
        tuple: x (np.ndarray), v (np.ndarray), W (np.ndarray) -
               position, velocity, and Weierstrass function values over time.
    """
    n_steps = len(t)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    x[0] = 1.0  # Initial position
    v[0] = 0.0  # Initial velocity
    W = np.zeros(n_steps)

    for i in range(1, n_steps):
        # Calculate Weierstrass function at current time step
        W_current = weierstrass(t[i-1], a, b, n_terms)
        W[i-1] = W_current

        # Calculate omega_sq for the current step
        omega_sq = omega0**2 * (1 + h * W_current)

        # Update position using a modified Euler-Cromer like method
        # This is an approximation for the second-order differential equation
        x[i] = x[i-1] + v[i-1]*dt - 0.5 * omega_sq * x[i-1] * dt**2

        # Calculate omega_sq for the next step (for velocity update)
        W_next = weierstrass(t[i], a, b, n_terms)
        omega_sq_next = omega0**2 * (1 + h * W_next)

        # Update velocity
        # This update uses an average of current and next omega_sq for better accuracy
        v[i] = v[i-1] - 0.5 * (omega_sq + omega_sq_next) * x[i] * dt
    return x, v, W

@njit
def compute_lyapunov(t, omega0, h, a, b, n_terms, dt, delta0=1e-8):
    """
    Computes the largest Lyapunov exponent for the modified harmonic oscillator.
    Uses the Benettin algorithm for calculating the Lyapunov exponent.

    Args:
        t (np.ndarray): Array of time points.
        omega0 (float): Natural frequency of the oscillator.
        h (float): Modulation strength.
        a (float): Parameter for Weierstrass function.
        b (float): Parameter for Weierstrass function.
        n_terms (int): Number of terms for Weierstrass function.
        dt (float): Time step size.
        delta0 (float, optional): Initial perturbation magnitude. Defaults to 1e-8.

    Returns:
        float: The computed Lyapunov exponent.
    """
    n_steps = len(t)
    x = 1.0  # Initial position of reference trajectory
    x_perturbed = x + delta0 # Initial position of perturbed trajectory
    v = 0.0  # Initial velocity of reference trajectory
    v_perturbed = v # Initial velocity of perturbed trajectory (same as reference for dv=0)
    sum_log_divergence = 0.0

    for i in range(1, n_steps):
        # Evaluate Weierstrass function at the previous time step for omega_sq calculation
        W_current = weierstrass(t[i-1], a, b, n_terms)
        omega_sq_current = omega0**2 * (1 + h * W_current)

        # Propagate reference trajectory
        x_new = x + v * dt - 0.5 * omega_sq_current * x * dt**2
        # For velocity, use current omega_sq and the new position (x_new)
        v_new = v - 0.5 * omega_sq_current * (x + x_new) * dt

        # Propagate perturbed trajectory
        x_perturbed_new = x_perturbed + v_perturbed * dt - 0.5 * omega_sq_current * x_perturbed * dt**2
        v_perturbed_new = v_perturbed - 0.5 * omega_sq_current * (x_perturbed + x_perturbed_new) * dt

        # Calculate the divergence vector
        dx = x_perturbed_new - x_new
        dv = v_perturbed_new - v_new

        # Calculate the current divergence magnitude
        delta = np.sqrt(dx*dx + dv*dv)

        # Avoid division by zero if delta becomes zero (should ideally not happen with proper dynamics)
        if delta == 0.0:
            # If divergence is zero, re-perturb to delta0 to avoid issues, or continue
            # For robust Lyapunov calculation, it's better to re-perturb
            # For this simple example, we'll continue, but a more rigorous approach would handle this
            continue

        # Add logarithm of divergence ratio to sum
        sum_log_divergence += np.log(delta / delta0)

        # Rescale the perturbed trajectory back to delta0 distance
        scale = delta0 / delta
        x_perturbed_new = x_new + dx * scale
        v_perturbed_new = v_new + dv * scale

        # Update states for the next iteration
        x, v = x_new, v_new
        x_perturbed, v_perturbed = x_perturbed_new, v_perturbed_new

    # Calculate Lyapunov exponent as the average log divergence per unit time
    return sum_log_divergence / (t[-1] - t[0])

def get_maxima(x, t, discard_ratio=0.5):
    """
    Finds local maxima in the position array x, discarding initial transient points.

    Args:
        x (np.ndarray): Array of position values.
        t (np.ndarray): Array of time points corresponding to x.
        discard_ratio (float, optional): Fraction of initial time to discard
                                         to avoid transient behavior. Defaults to 0.5.

    Returns:
        np.ndarray: Array of filtered maxima values.
    """
    # Find indices of local maxima
    idx_peaks = argrelextrema(x, np.greater)[0]
    if len(idx_peaks) == 0:
        return np.array([])  # Return empty array if no peaks found

    # Determine the time after which to consider peaks
    discard_time = t[0] + discard_ratio * (t[-1] - t[0])

    # Filter peaks that occur after the discard time
    filtered_peaks = [x[i] for i in idx_peaks if t[i] > discard_time]
    return np.array(filtered_peaks)

def plot_bifurcation_and_lyapunov(h_vals, t, omega0, a, b, n_terms, dt):
    """
    Plots the bifurcation diagram and the Lyapunov exponent as a function of h.

    Args:
        h_vals (np.ndarray): Array of modulation strengths (h) to test.
        t (np.ndarray): Array of time points for simulation.
        omega0 (float): Natural frequency of the oscillator.
        a (float): Parameter for Weierstrass function.
        b (float): Parameter for Weierstrass function.
        n_terms (int): Number of terms for Weierstrass function.
        dt (float): Time step size.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # --- Bifurcation Diagram ---
    # Iterate through each h value to simulate and find maxima
    for h in h_vals:
        x, v, W = solve_oscillator(t, omega0, h, a, b, n_terms, dt)
        peaks = get_maxima(x, t, discard_ratio=0.5)
        if len(peaks) > 0:
            # CORRECTED: Plot h on x-axis and peaks on y-axis
            axs[0].plot([h]*len(peaks), peaks, ',k', alpha=0.5)

    axs[0].set_xlabel('h (modulation strength)')
    axs[0].set_ylabel('x(t) maxima')
    axs[0].set_title('Bifurcation Diagram')
    axs[0].grid(True)

    # --- Lyapunov Exponent ---
    exponents = []
    # Iterate through each h value to compute the Lyapunov exponent
    for h in h_vals:
        lce = compute_lyapunov(t, omega0, h, a, b, n_terms, dt)
        exponents.append(lce)
    axs[1].plot(h_vals, exponents, 'r-', linewidth=1)
    axs[1].axhline(0, color='gray', linestyle='--') # Add a horizontal line at y=0
    axs[1].set_xlabel('h (modulation strength)')
    axs[1].set_ylabel('Lyapunov Exponent')
    axs[1].set_title('Lyapunov Exponent vs h')
    axs[1].grid(True)

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.show()

# --- Example Usage ---
if __name__ == '__main__':
    # Define parameters for the oscillator and Weierstrass function
    omega0 = 1.0
    a = 0.5
    b = 7
    n_terms = 30
    dt = 0.001
    T_max = 500  # Total simulation time
    t = np.arange(0, T_max, 0.001)

    # Define the range of h values for the bifurcation diagram and Lyapunov exponent plot
    h_vals = np.linspace(0, 1.0, 500) # Increased resolution for h_vals

    # Call the plotting function
    plot_bifurcation_and_lyapunov(h_vals, t, omega0, a, b, n_terms, dt)

