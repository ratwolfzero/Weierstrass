import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from numba import njit

# --- Numba-accelerated 2D Weierstrass function using precomputed terms ---


@njit
def compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs):
    W = np.zeros_like(X)
    for n in range(len(a_powers)):
        W += a_powers[n] * np.cos(b_freqs[n] * X) * np.cos(b_freqs[n] * Y)
    return W

# --- Density approximation via histogram ---


def compute_density_approx(values, bins=500):
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_indices = np.digitize(values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
    return hist[bin_indices]


# --- Parameters ---
size = 300
N = 20
bins = 500
odd_b_values = np.arange(3, 21, 2)

x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)

init_a = 0.5
init_b = 5

# Initial precomputation
a_powers = np.array([init_a ** n for n in range(N)], dtype=np.float64)
b_freqs = np.array([(init_b ** n) * np.pi for n in range(N)], dtype=np.float64)

Z = compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs)
Z_norm = Z / np.max(np.abs(Z))
density = compute_density_approx(Z.flatten(), bins).reshape(Z.shape)

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.1, bottom=0.35)

im = ax.imshow(density, extent=(-1, 1, -1, 1), cmap='inferno')
cbar = plt.colorbar(im, ax=ax, label='Density')
title = ax.set_title(f'Weierstrass Density (a={init_a:.2f}, b={init_b})')

# --- Sliders ---
ax_a = plt.axes([0.1, 0.25, 0.65, 0.03])
slider_a = Slider(ax_a, 'a', 0.01, 0.99, valinit=init_a, valstep=0.01)

ax_b = plt.axes([0.1, 0.2, 0.65, 0.03])
slider_b = Slider(ax_b, 'b', odd_b_values[0], odd_b_values[-1],
                  valinit=init_b, valstep=2)

# --- Checkbox ---
ax_check = plt.axes([0.1, 0.1, 0.2, 0.05])
check = CheckButtons(ax_check, ['Show Density'], [True])

# --- Update function ---


def update(val):
    a = slider_a.val
    b = slider_b.val

    # Precompute powers and frequencies
    a_powers = np.array([a ** n for n in range(N)], dtype=np.float64)
    b_freqs = np.array([(b ** n) * np.pi for n in range(N)], dtype=np.float64)

    Z = compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs)
    Z_norm = Z / np.max(np.abs(Z))
    show_density = check.get_status()[0]

    if show_density:
        Z_data = compute_density_approx(Z.flatten(), bins).reshape(Z.shape)
        im.set_clim(vmin=0, vmax=np.max(Z_data))
        im.set_cmap('hot')
        cbar.set_label('Density')
        title.set_text(f'Weierstrass Density (a={a:.2f}, b={int(b)})')
    else:
        Z_data = Z_norm
        im.set_clim(vmin=-1, vmax=1)
        im.set_cmap('coolwarm')
        cbar.set_label('Raw Value')
        title.set_text(f'Weierstrass Function (a={a:.2f}, b={int(b)})')

    im.set_data(Z_data)
    fig.canvas.draw_idle()


# --- Bind interactions ---
slider_a.on_changed(update)
slider_b.on_changed(update)
check.on_clicked(update)

plt.show()
