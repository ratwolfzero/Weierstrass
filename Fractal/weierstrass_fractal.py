import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
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

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.1, bottom=0.3) # Adjust bottom to make space for radio buttons and new labels

# Initialize with a dummy plot to be updated
im = ax.imshow(np.zeros((size, size)), extent=(-1, 1, -1, 1))
cbar = plt.colorbar(im, ax=ax)
title = ax.set_title('') # Title will be set in update_plot

# --- Sliders ---
# Define slider axes positions
ax_a_pos = [0.1, 0.18, 0.65, 0.03] # [left, bottom, width, height]
ax_b_pos = [0.1, 0.11, 0.65, 0.03]

ax_a = plt.axes(ax_a_pos)
slider_a = Slider(ax_a, '', 0.01, 0.99, valinit=init_a, valstep=0.01) # Empty label for the slider itself

ax_b = plt.axes(ax_b_pos)
slider_b = Slider(ax_b, '', odd_b_values[0], odd_b_values[-1],
                  valinit=init_b, valstep=2) # Empty label for the slider itself

# --- Custom Labels Above Sliders ---
# Calculate positions for the custom labels
# We'll place them just above the slider's axes
label_a_y = ax_a_pos[1] + ax_a_pos[3] + 0.01 # Slider bottom + height + small offset
label_b_y = ax_b_pos[1] + ax_b_pos[3] + 0.01

label_a_x = ax_a_pos[0] # Aligned with the left edge of the slider axes
label_b_x = ax_b_pos[0]

# Add text labels using fig.text for global positioning
fig.text(label_a_x, label_a_y, 'Amplitude Decay (a)',
         verticalalignment='bottom', horizontalalignment='left',
         fontsize=10, weight='bold')

fig.text(label_b_x, label_b_y, 'Frequency Growth (b)',
         verticalalignment='bottom', horizontalalignment='left',
         fontsize=10, weight='bold')

# --- Radio Buttons for View Mode ---
ax_radio = plt.axes([0.3, 0.05, 0.8, 0.05]) # Position for two radio buttons
radio_buttons = RadioButtons(ax_radio, ['Raw Values', 'Show Density'], active=0) # 'Raw Values' is default

# --- Remove the box frame around the RadioButtons ---
ax_radio.set_frame_on(False) # Hides the entire box frame
ax_radio.set_facecolor('none') # Makes the background transparent

# --- Update function ---
def update_plot(val):
    a = slider_a.val
    b = slider_b.val
    view_mode = radio_buttons.value_selected

    # Precompute powers and frequencies
    a_powers = np.array([a ** n for n in range(N)], dtype=np.float64)
    b_freqs = np.array([(b ** n) * np.pi for n in range(N)], dtype=np.float64)

    Z = compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs)
    Z_norm = Z / np.max(np.abs(Z)) # Normalized for Raw Values

    if view_mode == 'Raw Values':
        data = Z_norm
        cmap = 'coolwarm'
        clim = (-1, 1)
        label = 'Normalized Value'
        current_title = f'2D Weierstrass Function (a={a:.2f}, b={int(b)})'
    elif view_mode == 'Show Density':
        data = compute_density_approx(Z.flatten(), bins).reshape(Z.shape)
        cmap = 'inferno'
        # Set clim based on the actual max density, which can vary
        clim = (0, np.max(data) + 1e-9 if np.max(data) == 0 else np.max(data))
        label = 'Density'
        current_title = f'Weierstrass Density (a={a:.2f}, b={int(b)})'

    im.set_data(data)
    im.set_clim(*clim)
    im.set_cmap(cmap)
    cbar.set_label(label)
    title.set_text(current_title)
    fig.canvas.draw_idle()

# --- Bind interactions ---
slider_a.on_changed(update_plot)
slider_b.on_changed(update_plot)
radio_buttons.on_clicked(update_plot)

# Perform initial plot update
update_plot(None) # Call once to set up the initial display

plt.show()
