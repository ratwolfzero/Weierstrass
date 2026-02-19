import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from numba import njit

# --- Numba-accelerated 2D Weierstrass function ---
@njit
def compute_weierstrass_2d(X, Y, a_powers, b_freqs):
    W = np.zeros_like(X)
    for n in range(len(a_powers)):
        W += a_powers[n] * np.cos(b_freqs[n] * X) * np.cos(b_freqs[n] * Y)
    return W

# --- Density approximation via histogram ---
def compute_density_approx(values, bins=500):
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_indices = np.searchsorted(bin_edges, values, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
    return hist[bin_indices]

# --- Parameters ---
size = 500       # Grid resolution
N = 40           # Number of terms
bins = 500       # Histogram bins for density view
odd_b_values = np.arange(3, 21, 2)

# Coordinate grids
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
x_edges = np.linspace(-1, 1, size + 1)
y_edges = np.linspace(-1, 1, size + 1)
X, Y = np.meshgrid(x, y)

init_a = 0.22
init_b = 5

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.1, bottom=0.3)

# Visualization artists
im_raw = ax.imshow(
    np.zeros((size, size)),
    extent=(-1, 1, -1, 1),
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    origin='lower'
)
im_density = ax.pcolormesh(
    x_edges,
    y_edges,
    np.zeros((size, size)),
    cmap='inferno',
    shading='auto'
)

cbar = plt.colorbar(im_raw, ax=ax)
title = ax.set_title('', fontsize=12)
ax.set_xlabel('X Coordinate', fontsize=10)
ax.set_ylabel('Y Coordinate', fontsize=10)
ax.set_aspect('equal')

# --- Sliders ---
ax_a = plt.axes([0.1, 0.18, 0.65, 0.03])
slider_a = Slider(ax_a, '', 0.01, 0.99, valinit=init_a, valstep=0.01)

ax_b = plt.axes([0.1, 0.11, 0.65, 0.03])
slider_b = Slider(ax_b, '', odd_b_values[0], odd_b_values[-1], valinit=init_b, valstep=2)

fig.text(0.1, 0.22, 'Amplitude Decay (a)', fontsize=10, weight='bold')
fig.text(0.1, 0.15, 'Frequency Growth (b)', fontsize=10, weight='bold')

# --- Radio Buttons ---
ax_radio = plt.axes([0.1, 0.05, 0.8, 0.05])
radio_buttons = RadioButtons(ax_radio, ['Raw Values', 'Show Density'], active=1)
ax_radio.set_frame_on(False)
ax_radio.set_facecolor('none')

# --- Validity Indicator ---
indicator_ax = plt.axes([0.8, 0.18, 0.1, 0.03])
indicator_text = indicator_ax.text(
    0.5, 0.5, '', fontsize=10, ha='center', va='center', transform=indicator_ax.transAxes)
indicator_ax.set_xticks([])
indicator_ax.set_yticks([])
indicator_ax.set_frame_on(True)
indicator_ax.set_facecolor('white')
fig.text(0.8, 0.22, 'Constraint: ab ≥ 1', fontsize=8)
					 
# --- Update function ---
def update_plot(val):
    a = slider_a.val
    b = int(slider_b.val)
    ab = a * b
    view_mode = radio_buttons.value_selected

    # Update validity indicator
    indicator_text.set_text(f"a·b = {ab:.2f}")
    indicator_ax.set_facecolor('lightgreen' if ab >= 1 else 'lightcoral')

    # --- Precompute powers and frequencies only when sliders change ---
    a_powers = np.array([a ** n for n in range(N)], dtype=np.float64)
    b_freqs = np.array([b ** n * np.pi for n in range(N)], dtype=np.float64)

    # Compute 2D Weierstrass function
    Z = compute_weierstrass_2d(X, Y, a_powers, b_freqs)
    Z_norm = Z / np.max(np.abs(Z))

    # --- Update visualization ---
    if view_mode == 'Raw Values':
        im_raw.set_data(Z_norm)
        im_raw.set_visible(True)
        im_density.set_visible(False)
        cbar.update_normal(im_raw)
        cbar.set_label('Normalized Value', fontsize=10)
        title.set_text(f'Normalized 2D Weierstrass Function (a={a:.2f}, b={b})')
    else:
        density = compute_density_approx(Z_norm.flatten(), bins).reshape((size, size))
        im_density.set_array(density.ravel())
        im_density.set_clim(0, np.max(density))
        im_density.set_visible(True)
        im_raw.set_visible(False)
        cbar.update_normal(im_density)
        cbar.set_label('Probability Density', fontsize=10)
        title.set_text(f'Value Probability Density (a={a:.2f}, b={b})')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    fig.canvas.draw_idle()

# --- Bind events ---
slider_a.on_changed(update_plot)
slider_b.on_changed(update_plot)
radio_buttons.on_clicked(update_plot)

# --- Initial plot ---
update_plot(None)
plt.show()
