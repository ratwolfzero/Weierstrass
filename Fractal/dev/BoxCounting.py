import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
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
    

# --- Box-counting dimension calculation ---
@njit
def box_counting_dimension(Z, epsilons):
    size = Z.shape[0]
    counts = np.zeros(len(epsilons))
    
    # Normalize Z to [0,1] range
    z_min = np.min(Z)
    z_max = np.max(Z)
    z_range = z_max - z_min + 1e-9
    Z_norm = (Z - z_min) / z_range
    
    for i in range(len(epsilons)):
        eps = epsilons[i]
        
        # Calculate box sizes (same relative size in all dimensions)
        box_size_xy = max(1, int(np.ceil(eps * size)))
        box_size_z = eps  # Same relative size in Z dimension
        
        # Number of boxes in each dimension
        grid_x = (size + box_size_xy - 1) // box_size_xy
        grid_y = (size + box_size_xy - 1) // box_size_xy
        grid_z = max(1, int(np.ceil(1.0 / box_size_z)))
        
        occupied = np.zeros((grid_x, grid_y, grid_z), dtype=np.bool_)
        
        for x in range(size):
            for y in range(size):
                # Spatial boxes
                bx = x // box_size_xy
                by = y // box_size_xy
                
                # Height box
                bz = int(Z_norm[x,y] / box_size_z)
                bz = min(bz, grid_z - 1)  # Ensure within bounds
                
                if not occupied[bx, by, bz]:
                    occupied[bx, by, bz] = True
        
        counts[i] = np.sum(occupied)
    
    # Manual linear regression
    sum_x = sum_y = sum_xy = sum_x2 = 0.0
    n = len(epsilons)
    for i in range(n):
        log_eps = np.log(epsilons[i])
        log_count = np.log(counts[i])
        sum_x += log_eps
        sum_y += log_count
        sum_xy += log_eps * log_count
        sum_x2 += log_eps * log_eps
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    return -slope
    

# --- Parameters ---
size = 600
N = 500
bins = 500 
odd_b_values = np.arange(3, 21, 2)

x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)

init_a = 0.22
init_b = 5

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.1, bottom=0.35)  # Adjusted for button space

im = ax.imshow(np.zeros((size, size)), extent=(-1, 1, -1, 1))
cbar = plt.colorbar(im, ax=ax)
title = ax.set_title('')

# --- Sliders ---
ax_a_pos = [0.1, 0.25, 0.65, 0.03]
ax_b_pos = [0.1, 0.18, 0.65, 0.03]

ax_a = plt.axes(ax_a_pos)
slider_a = Slider(ax_a, '', 0.01, 0.99, valinit=init_a, valstep=0.01)

ax_b = plt.axes(ax_b_pos)
slider_b = Slider(ax_b, '', odd_b_values[0], odd_b_values[-1], valinit=init_b, valstep=2)

# --- Custom Labels Above Sliders ---
label_a_y = ax_a_pos[1] + ax_a_pos[3] + 0.01
label_b_y = ax_b_pos[1] + ax_b_pos[3] + 0.01

fig.text(ax_a_pos[0], label_a_y, 'Amplitude Decay (a)', verticalalignment='bottom',
         horizontalalignment='left', fontsize=10, weight='bold')

fig.text(ax_b_pos[0], label_b_y, 'Frequency Growth (b)', verticalalignment='bottom',
         horizontalalignment='left', fontsize=10, weight='bold')

# --- Radio Buttons for View Mode ---
ax_radio = plt.axes([0.3, 0.12, 0.4, 0.05])
radio_buttons = RadioButtons(ax_radio, ['Raw Values', 'Show Density'], active=1)
ax_radio.set_frame_on(False)
ax_radio.set_facecolor('none')

# --- Validity Indicator Box ---
indicator_ax = plt.axes([0.8, 0.25, 0.1, 0.03])
indicator_text = indicator_ax.text(0.5, 0.5, '', fontsize=10, ha='center', va='center',
                                   transform=indicator_ax.transAxes)
indicator_ax.set_xticks([])
indicator_ax.set_yticks([])
indicator_ax.set_frame_on(True)
indicator_ax.set_facecolor('white')

fig.text(0.8, 0.29, 'Constraint ab ≥ 1', fontsize=8)

# --- Box-Counting Button ---
ax_button = plt.axes([0.1, 0.05, 0.3, 0.05])
button = Button(ax_button, 'Calculate Box-Counting Dimension', color='lightgoldenrodyellow')

# --- Global variable to store current Z_norm ---
current_Z_norm = None

# --- Update function ---
def update_plot(val):
    global current_Z_norm
    a = slider_a.val
    b = slider_b.val
    ab = a * b
    view_mode = radio_buttons.value_selected

    # Update validity indicator
    indicator_text.set_text(f"a·b = {ab:.2f}")
    if ab >= 1:
        indicator_ax.set_facecolor('lightgreen')
    else:
        indicator_ax.set_facecolor('lightcoral')

    # Precompute terms
    a_powers = np.array([a ** n for n in range(N)], dtype=np.float64)
    b_freqs = np.array([(b ** n) * np.pi for n in range(N)], dtype=np.float64)

    Z = compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs)
    Z_norm = Z / np.max(np.abs(Z))
    current_Z_norm = Z_norm  # Store for button callback

    if view_mode == 'Raw Values':
        data = Z_norm
        cmap = 'coolwarm'
        clim = (-1, 1)
        label = 'Normalized Value'
        current_title = f'2D Weierstrass Function (a={a:.2f}, b={int(b)})'
    else:
        data = compute_density_approx(Z.flatten(), bins).reshape(Z.shape)
        cmap = 'inferno'
        clim = (0, np.max(data) + 1e-9 if np.max(data) == 0 else np.max(data))
        label = 'Density'
        current_title = f'Weierstrass Density (a={a:.2f}, b={int(b)})'

    im.set_data(data)
    im.set_clim(*clim)
    im.set_cmap(cmap)
    cbar.set_label(label)
    title.set_text(current_title)
    fig.canvas.draw_idle()

# --- Button callback ---
def calculate_dimension(event):
    global current_Z_norm
    if current_Z_norm is None:
        return
    
    a = slider_a.val
    b = slider_b.val
    ab = a * b
    
    if ab < 1:
        title.set_text("Box-counting requires ab ≥ 1 (fractal condition not met)")
        fig.canvas.draw_idle()
        return
    
    epsilons = np.linspace(0.02, 0.2, 10)
    dimension = box_counting_dimension(current_Z_norm, epsilons)
    
    current_title = title.get_text()
    title.set_text(f"{current_title}\nBox-Counting Dimension: {dimension:.3f}")
    fig.canvas.draw_idle()

# --- Bind events ---
slider_a.on_changed(update_plot)
slider_b.on_changed(update_plot)
radio_buttons.on_clicked(update_plot)
button.on_clicked(calculate_dimension)

# Initial plot
update_plot(None)

plt.show()
