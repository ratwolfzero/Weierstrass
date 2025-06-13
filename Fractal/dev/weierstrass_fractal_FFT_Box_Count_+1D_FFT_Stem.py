import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from numba import njit
import matplotlib.gridspec as gridspec

# --- Numba-accelerated 1D Weierstrass function ---
@njit
def compute_weierstrass_1d(y, a_powers, b_freqs):
    total = np.zeros_like(y)
    for n in range(len(a_powers)):
        total += a_powers[n] * np.cos(b_freqs[n] * y)
    return total

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

# --- FFT computation ---
def compute_fft(Z):
    fft_Z = np.fft.fft2(Z)
    fft_shifted = np.fft.fftshift(fft_Z)
    magnitude = np.abs(fft_shifted)
    return np.log10(magnitude + 1e-10)

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

        # Calculate box sizes
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
                bz = int(Z_norm[x, y] / box_size_z)
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
size = 500
N = 30
bins = 500
odd_b_values = np.arange(3, 21, 2)

x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)

# For density view using pcolormesh
x_edges = np.linspace(-1, 1, size+1)
y_edges = np.linspace(-1, 1, size+1)

init_a = 0.22
init_b = 5

# Frequency coordinates (radians/sample)
freq_x = np.fft.fftshift(np.fft.fftfreq(size, d=2/size)) * 2 * np.pi
freq_y = np.fft.fftshift(np.fft.fftfreq(size, d=2/size)) * 2 * np.pi
extent_freq = [freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]]

# --- Plot setup with improved layout ---
fig = plt.figure(figsize=(16, 10))
# Create grid: 2 rows, 2 columns, with width ratios 1.4:0.6 (70%:30%)
gs = gridspec.GridSpec(2, 2, width_ratios=[1.4, 0.6], height_ratios=[1, 1],
                       top=0.95, bottom=0.35, hspace=0.3, wspace=0.3)

# Left: Larger 2D plot (spans both rows)
ax0 = plt.subplot(gs[:, 0])
im = ax0.imshow(np.zeros((size, size)), extent=(-1, 1, -1, 1))
cbar = plt.colorbar(im, ax=ax0)
title_2d = ax0.set_title('')

# Right top: 1D Weierstrass
ax1 = plt.subplot(gs[0, 1])
line_1d, = ax1.plot(y, np.zeros_like(y), 'b-')
ax1.set_title('1D Weierstrass Function (x=0)')
ax1.set_xlabel('y')
ax1.set_ylabel('W(y)')
ax1.grid(True)

# Right bottom: FFT of 1D function - STEM PLOT (FIXED)
ax2 = plt.subplot(gs[1, 1])
ax2.set_title('FFT of 1D Weierstrass (Stem Plot)')
ax2.set_xlabel('Frequency (cycles/sample)')
ax2.set_ylabel('Magnitude (log scale)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True)

# Initialize with dummy stem plot that will be updated later
fft_stem = None

# --- Sliders ---
ax_a_pos = [0.1, 0.25, 0.65, 0.03]
ax_b_pos = [0.1, 0.18, 0.65, 0.03]

ax_a = plt.axes(ax_a_pos)
slider_a = Slider(ax_a, '', 0.01, 0.99, valinit=init_a, valstep=0.01)

ax_b = plt.axes(ax_b_pos)
slider_b = Slider(
    ax_b, '', odd_b_values[0], odd_b_values[-1], valinit=init_b, valstep=2)

# --- Custom Labels Above Sliders ---
label_a_y = ax_a_pos[1] + ax_a_pos[3] + 0.01
label_b_y = ax_b_pos[1] + ax_b_pos[3] + 0.01

fig.text(ax_a_pos[0], label_a_y, 'Amplitude Decay (a)', verticalalignment='bottom',
         horizontalalignment='left', fontsize=10, weight='bold')

fig.text(ax_b_pos[0], label_b_y, 'Frequency Growth (b)', verticalalignment='bottom',
         horizontalalignment='left', fontsize=10, weight='bold')

# --- Radio Buttons for View Mode ---
ax_radio = plt.axes([0.1, 0.12, 0.8, 0.05])
radio_buttons = RadioButtons(
    ax_radio, ['Raw Values', 'Show Density', 'Show FFT'], active=1)
ax_radio.set_frame_on(False)
ax_radio.set_facecolor('none')

# --- Validity Indicator Box ---
indicator_ax = plt.axes([0.8, 0.20, 0.1, 0.03])
indicator_text = indicator_ax.text(0.5, 0.5, '', fontsize=10, ha='center', va='center',
                                   transform=indicator_ax.transAxes)
indicator_ax.set_xticks([])
indicator_ax.set_yticks([])
indicator_ax.set_frame_on(True)
indicator_ax.set_facecolor('white')

fig.text(0.8, 0.25, 'Constraint ab ≥ 1', fontsize=8)

# --- Box-Counting Button ---
ax_button = plt.axes([0.1, 0.05, 0.3, 0.05])
button = Button(ax_button, 'Display Box-Counting (Fractal) Dimension',
                color='lightgoldenrodyellow')

# --- Dimension Display ---
ax_dim = plt.axes([0.42, 0.05, 0.18, 0.05])
dim_text = ax_dim.text(0.5, 0.5, 'Fractal Dimension: --', fontsize=10,
                       ha='center', va='center', transform=ax_dim.transAxes)
ax_dim.set_xticks([])
ax_dim.set_yticks([])
ax_dim.set_frame_on(True)
ax_dim.set_facecolor('lightblue')

# --- Global variables ---
current_Z_norm = None
current_1d_data = None
current_dimension = None
current_plot = None
dimension_calculated = False
last_a = init_a
last_b = init_b
fft_stem = None  # To store stem plot artists

# --- Update function ---
def update_plot(val):
    global current_Z_norm, current_1d_data, current_dimension, current_plot, dimension_calculated, last_a, last_b, fft_stem
    a = slider_a.val
    b = slider_b.val
    ab = a * b
    view_mode = radio_buttons.value_selected

    # Check if parameters changed
    params_changed = (a != last_a) or (b != last_b)
    if params_changed:
        dimension_calculated = False
        last_a = a
        last_b = b
        button.color = 'lightgoldenrodyellow'
        button.hovercolor = 'lightgoldenrodyellow'
        if ab >= 1:
            dim_text.set_text('Fractal Dimension: --')
        else:
            dim_text.set_text('ab < 1: Not fractal')

    # Update validity indicator
    indicator_text.set_text(f"a·b = {ab:.2f}")
    if ab >= 1:
        indicator_ax.set_facecolor('lightgreen')
        if not dimension_calculated:
            button.color = 'lightgoldenrodyellow'
            button.hovercolor = 'lightgoldenrodyellow'
    else:
        indicator_ax.set_facecolor('lightcoral')
        button.color = 'lightgray'
        button.hovercolor = 'lightgray'

    # Precompute terms
    a_powers = np.array([a ** n for n in range(N)], dtype=np.float64)
    b_freqs = np.array([(b ** n) * np.pi for n in range(N)], dtype=np.float64)

    # Compute 2D surface
    Z = compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs)
    Z_norm = Z / np.max(np.abs(Z))
    current_Z_norm = Z_norm
    
    # Compute 1D slice at x=0 (center of grid)
    center_idx = size // 2
    y_vals = y
    W_1d = compute_weierstrass_1d(y_vals, a_powers, b_freqs)
    W_1d_norm = W_1d / np.max(np.abs(W_1d))
    current_1d_data = W_1d_norm
    
    # Compute FFT of 1D function
    fft_vals = np.fft.fft(W_1d_norm)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.fftfreq(len(W_1d_norm), d=2/size)
    
    # Only show positive frequencies
    pos_freqs = freqs[:len(freqs)//2]
    pos_fft = fft_mag[:len(fft_mag)//2]
    
    # Update 1D plot
    line_1d.set_data(y_vals, W_1d_norm)
    ax1.relim()
    ax1.autoscale_view()
    
    # Update FFT plot with stem plot
    # Clear previous stem plot if it exists
    if fft_stem is not None:
        # Remove all components of the stem plot
        for artist in fft_stem:
            try:
                artist.remove()
            except:
                pass
    
    # Create new stem plot
    if len(pos_freqs) > 0:
        fft_stem = ax2.stem(
            pos_freqs, 
            pos_fft, 
            linefmt='r-', 
            markerfmt='ro', 
            basefmt='k-'
        )
        
        # Style adjustments for better visibility
        # Access the stem components correctly
        plt.setp(fft_stem[0], markersize=3, markerfacecolor='r', markeredgecolor='r')  # Markers
        plt.setp(fft_stem[1], linewidth=0.7)  # Stem lines
        plt.setp(fft_stem[2], linewidth=0.7)  # Baseline
        
        # Set axis limits
        ax2.set_xlim(0, np.max(pos_freqs))
        ax2.set_ylim(np.min(pos_fft[pos_fft > 0]) * 0.9, np.max(pos_fft) * 1.1)
    
    # Clear previous 2D plot if it exists
    if current_plot:
        current_plot.remove()
        current_plot = None

    if view_mode == 'Raw Values':
        data = Z_norm
        cmap = 'coolwarm'
        clim = (-1, 1)
        label = 'Normalized Value'
        current_plot = ax0.imshow(data, cmap=cmap, extent=(-1, 1, -1, 1))
        ax0.set_xlim(-1, 1)
        ax0.set_ylim(-1, 1)
        ax0.set_xlabel('X Coordinate')
        ax0.set_ylabel('Y Coordinate')
        ax0.set_aspect('equal')
        current_title = f'Normalized 2D Weierstrass Function (a={a:.2f}, b={int(b)})'
    elif view_mode == 'Show Density':
        data = compute_density_approx(Z.flatten(), bins).reshape(Z.shape)
        cmap = 'inferno'
        clim = (0, np.max(data) + 1e-9 if np.max(data) == 0 else np.max(data))
        label = 'Probability Density'
        current_plot = ax0.pcolormesh(
            x_edges, y_edges, data, cmap=cmap, shading='auto')
        ax0.set_xlim(-1, 1)
        ax0.set_ylim(-1, 1)
        ax0.set_xlabel('X Coordinate')
        ax0.set_ylabel('Y Coordinate')
        ax0.set_aspect('equal')
        current_title = f'Value Probability Density (a={a:.2f}, b={int(b)})'
    else:  # Show FFT
        data = compute_fft(Z)
        cmap = 'inferno'
        clim = (np.min(data), np.max(data))
        label = 'Log-Magnitude (dB)'
        current_plot = ax0.imshow(data, cmap='viridis', extent=extent_freq)
        ax0.set_xlim(extent_freq[0], extent_freq[1])
        ax0.set_ylim(extent_freq[2], extent_freq[3])
        ax0.set_xlabel('Angular Frequency ω_x (rad/sample)')
        ax0.set_ylabel('Angular Frequency ω_y (rad/sample)')
        ax0.set_aspect('equal')
        current_title = f'Frequency Spectrum (a={a:.2f}, b={int(b)})'

    current_plot.set_clim(*clim)
    cbar.update_normal(current_plot)
    cbar.set_label(label)
    title_2d.set_text(current_title)
    ax1.set_title(f'1D Weierstrass (x=0, a={a:.2f}, b={int(b)})')
    fig.canvas.draw_idle()

# --- Button callback ---
def calculate_dimension(event):
    global current_Z_norm, current_dimension, dimension_calculated
    if current_Z_norm is None or dimension_calculated:
        return

    a = slider_a.val
    b = slider_b.val
    ab = a * b

    if ab < 1:
        dim_text.set_text('ab < 1: Not fractal')
        return

    # Switch to raw values view for dimension visualization
    radio_buttons.set_active(0)
    
    epsilons = np.linspace(0.02, 0.2, 10)
    current_dimension = box_counting_dimension(current_Z_norm, epsilons)
    
    dim_text.set_text(f'Fractal Dimension: {current_dimension:.3f}')
    
    dimension_calculated = True
    button.color = 'lightgray'
    button.hovercolor = 'lightgray'

# --- Bind events ---
slider_a.on_changed(update_plot)
slider_b.on_changed(update_plot)
radio_buttons.on_clicked(update_plot)
button.on_clicked(calculate_dimension)

# Initial plot
update_plot(None)
plt.show()
