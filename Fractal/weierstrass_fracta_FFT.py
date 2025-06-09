import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from numba import njit

# --- Numba-accelerated Weierstrass function ---
@njit
def compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs):
    W = np.zeros_like(X)
    for n in range(len(a_powers)):
        W += a_powers[n] * np.cos(b_freqs[n] * X) * np.cos(b_freqs[n] * Y)
    return W

# --- Density approximation ---
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

# --- Parameters ---
size = 300
N = 20
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)
init_a, init_b = 0.5, 5

# Frequency coordinates (radians/sample)
freq_x = np.fft.fftshift(np.fft.fftfreq(size, d=2/size)) * 2 * np.pi
freq_y = np.fft.fftshift(np.fft.fftfreq(size, d=2/size)) * 2 * np.pi
extent_freq = [freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]]

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

# Initialize
a_powers = np.array([init_a ** n for n in range(N)], dtype=np.float64)
b_freqs = np.array([(init_b ** n) * np.pi for n in range(N)], dtype=np.float64)
Z = compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs)
Z_norm = Z / np.max(np.abs(Z))
im = ax.imshow(Z_norm, extent=(-1, 1, -1, 1), cmap='coolwarm')
cbar = plt.colorbar(im, ax=ax, label='Value')
title = ax.set_title(f'Weierstrass Function (a={init_a:.2f}, b={init_b})')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# --- Widgets ---
ax_a = plt.axes([0.1, 0.25, 0.65, 0.03])
slider_a = Slider(ax_a, 'a', 0.01, 0.99, valinit=init_a, valstep=0.01)
ax_b = plt.axes([0.1, 0.2, 0.65, 0.03])
slider_b = Slider(ax_b, 'b', 3, 19, valinit=init_b, valstep=2)
ax_check = plt.axes([0.1, 0.1, 0.2, 0.05])
check = CheckButtons(ax_check, ['Show Density'], [False])
ax_fft = plt.axes([0.4, 0.1, 0.2, 0.05])
fft_button = Button(ax_fft, 'Toggle FFT')

# --- State ---
show_density = False
show_fft = False

# --- Update function ---
def update(val):
    global show_density, show_fft
    a = slider_a.val
    b = int(slider_b.val)
    
    a_powers = np.array([a ** n for n in range(N)], dtype=np.float64)
    b_freqs = np.array([(b ** n) * np.pi for n in range(N)], dtype=np.float64)
    Z = compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs)
    Z_norm = Z / np.max(np.abs(Z))
    
    show_density = check.get_status()[0]
    
    if show_fft:
        Z_data = compute_fft(Z)
        im.set_cmap('inferno')
        im.set_clim(vmin=np.min(Z_data), vmax=np.max(Z_data))
        im.set_extent(extent_freq)
        ax.set_xlabel('Frequency (radians/sample)')
        ax.set_ylabel('Frequency (radians/sample)')
        cbar.set_label('Log-Magnitude')
        title.set_text(f'FFT (a={a:.2f}, b={b})')
    else:
        im.set_extent((-1, 1, -1, 1))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if show_density:
            Z_data = compute_density_approx(Z.flatten(), bins=500).reshape(Z.shape)
            im.set_cmap('hot')
            im.set_clim(vmin=0, vmax=np.max(Z_data))
            cbar.set_label('Density')
            title.set_text(f'Weierstrass Density (a={a:.2f}, b={b})')
        else:
            Z_data = Z_norm
            im.set_cmap('coolwarm')
            im.set_clim(vmin=-1, vmax=1)
            cbar.set_label('Raw Value')
            title.set_text(f'Weierstrass Function (a={a:.2f}, b={b})')
    
    im.set_data(Z_data)
    fig.canvas.draw_idle()

# --- Toggle FFT ---
def toggle_fft(event):
    global show_fft
    show_fft = not show_fft
    update(None)

# --- Bind interactions ---
slider_a.on_changed(update)
slider_b.on_changed(update)
check.on_clicked(update)
fft_button.on_clicked(toggle_fft)

plt.show()
