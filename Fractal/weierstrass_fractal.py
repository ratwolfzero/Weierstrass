import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numba import njit

# --- Numba-accelerated 2D Weierstrass function ---
@njit
def compute_weierstrass_2d(X, Y, a, b, N):
    W = np.zeros_like(X)
    pi = np.pi
    for n in range(N):
        freq = (b ** n) * pi
        cos_x = np.cos(freq * X)
        cos_y = np.cos(freq * Y)
        W += (a ** n) * cos_x * cos_y
    return W

# --- Density approximation via histogram ---
def compute_density_approx(values, bins=500):
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_indices = np.digitize(values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
    return hist[bin_indices]

# --- Parameters ---
size = 300
N = 20               # Fixed depth
bins = 500           # Histogram bins for density
odd_b_values = np.arange(3, 21, 2)  # Allowed b values: 3, 5, ..., 19

x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)

# --- Initial computation ---
init_a = 0.5
init_b = 5  # Start with b=5

Z = compute_weierstrass_2d(X, Y, init_a, init_b, N)
density = compute_density_approx(Z.flatten(), bins).reshape(Z.shape)

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.1, bottom=0.2)
im = ax.imshow(density, extent=(-1, 1, -1, 1), cmap='inferno')
cbar = plt.colorbar(im, ax=ax, label='Density')
ax.set_title(f'Weierstrass Density (a={init_a:.2f}, b={init_b})')

# --- Sliders ---
ax_a = plt.axes([0.1, 0.1, 0.65, 0.03])
slider_a = Slider(ax_a, 'a', 0.01, 0.99, valinit=init_a, valstep=0.01)

ax_b = plt.axes([0.1, 0.05, 0.65, 0.03])
# Create slider with actual b values
slider_b = Slider(ax_b, 'b', odd_b_values[0], odd_b_values[-1], 
                 valinit=init_b, valstep=2)  # step=2 for odd numbers

# --- Update function ---
def update(val):
    a = slider_a.val
    b = slider_b.val
    Z = compute_weierstrass_2d(X, Y, a, b, N)
    density = compute_density_approx(Z.flatten(), bins).reshape(Z.shape)
    im.set_data(density)
    ax.set_title(f'Weierstrass Density (a={a:.2f}, b={int(b)}')  # Show as integer
    fig.canvas.draw_idle()

# --- Connect sliders to update ---
slider_a.on_changed(update)
slider_b.on_changed(update)

plt.show()
