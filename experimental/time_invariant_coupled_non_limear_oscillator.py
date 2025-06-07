import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit

# Weierstrass Function
@njit
def weierstrass(t, a=0.5, b=3.0, n_terms=30):
    result = 0.0
    for n in range(n_terms):
        result += a ** n * np.cos(np.pi * b ** n * t)
    return result

# Generate both versions
@njit
def generate_comparison(n_steps, scale=2.0, a_amp=2.15, b_amp=-1.1):
    # Original oscillator arrays
    xs_orig = np.zeros(n_steps)                                  
    ys_orig = np.zeros(n_steps)
    
    # Weierstrass-driven arrays
    xs_wei = np.zeros(n_steps)                                                 
    ys_wei = np.zeros(n_steps)
    
    x_orig, y_orig = 0.0, 0.0
    x_wei, y_wei = 0.0, 0.0

    for i in range(n_steps):
        t = i / n_steps
        
        # Original oscillator (constant parameters)
        x_new_orig = np.sin(scale * y_orig)
        y_new_orig = np.cos(scale * x_orig)
        xs_orig[i] = x_new_orig
        ys_orig[i] = y_new_orig
        x_orig, y_orig = x_new_orig, y_new_orig
        
        # Weierstrass-driven oscillator
        a = scale + a_amp * weierstrass(t)                       
        b = scale + b_amp * weierstrass(t + 0.1)
        x_new_wei = np.sin(a * y_wei)
        y_new_wei = np.cos(b * x_wei)                                                           
        xs_wei[i] = x_new_wei
        ys_wei[i] = y_new_wei
        x_wei, y_wei = x_new_wei, y_new_wei

    return xs_orig, ys_orig, xs_wei, ys_wei

# Plotting function
def plot_comparison(xs_orig, ys_orig, xs_wei, ys_wei, bins=1000):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)  # Moderate size

    # Original oscillator plot
    heatmap_orig, _, _ = np.histogram2d(xs_orig, ys_orig, bins=bins)
    im1 = ax1.imshow(heatmap_orig.T, origin='lower', cmap='viridis', norm=LogNorm())
    ax1.set_title('Original Oscillator\n(Constant Parameters)', fontsize=13, fontweight='semibold')
    ax1.axis('off')

    # Weierstrass-driven oscillator plot
    heatmap_wei, xedges, yedges = np.histogram2d(xs_wei, ys_wei, bins=bins)
    im2 = ax2.imshow(heatmap_wei.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                     origin='lower', cmap='magma', norm=LogNorm())
    ax2.set_title('Weierstrass-Driven Oscillator\n(Fractal-Modulated Parameters)', fontsize=13, fontweight='semibold')
    ax2.axis('off')

    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.05, top=0.92)  # More room for widgets at bottom
    plt.show()



# Generate and plot
n_steps = 2_000_000  # Reduced for faster testing
xs_orig, ys_orig, xs_wei, ys_wei = generate_comparison(n_steps)
plot_comparison(xs_orig, ys_orig, xs_wei, ys_wei)

