import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import LogNorm
from numba import njit

# ---- True Weierstrass Function ----
@njit
def weierstrass(t, a=0.5, b=3.0, n_terms=30):
    result = 0.0
    for n in range(n_terms):
        result += a ** n * np.cos(np.pi * b ** n * t)
    return result

# ---- Parametric Oscillator with Weierstrass Modulation ----
@njit
def generate_points(n_steps, scale=2.0, a_amp=2.15, b_amp=-1.15):
    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)
    x, y = 0.0, 0.0

    for i in range(n_steps):
        t = i / n_steps
        # Use Weierstrass modulation for a(t) and b(t)
        a = scale + a_amp * weierstrass(t, a=0.5, b=3.0)
        b = scale + b_amp * weierstrass(t + 0.1, a=0.5, b=3.0)
        x_new = np.sin(a * y)
        y_new = np.cos(b * x)
        xs[i] = x_new
        ys[i] = y_new
        x, y = x_new, y_new                                                                               

    return xs, ys

# ---- Histogram Renderer ----
def plot_histogram(xs, ys, bins=1000, cmap='magma'):
    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower',
               cmap=cmap, norm=LogNorm(), interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ---- CLI Entry Point ----
def main():
    parser = argparse.ArgumentParser(description="Weierstrass-Modulated Parametric Oscillator (Histogram Rendering)")
    parser.add_argument("--steps", type=int, default=8_000_000, help="Number of iterations (default: 8 million)")
    parser.add_argument("--cmap", type=str, default="magma", help="Colormap (e.g., magma, viridis, inferno)")
    args = parser.parse_args()

    print(f"[i] Generating {args.steps} points...")
    xs, ys = generate_points(args.steps)

    print(f"[i] Rendering histogram using colormap: {args.cmap}")
    plot_histogram(xs, ys, cmap=args.cmap)

if __name__ == "__main__":
    main()








