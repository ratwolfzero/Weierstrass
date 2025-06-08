import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import numpy as np

import numpy as np


def weierstrass(x, a=0.7, b=9, n_terms=30):
    assert 0 < a < 1, "a must be in (0, 1)"
    assert isinstance(
        b, int) and b > 1, "b must be a positive integer greater than 1"
    assert a * b > 1 + \
        (3 * np.pi / 2), "ab must exceed 1 + 3π/2 for nowhere differentiability (Weierstrass condition)"

    result = np.zeros_like(x, dtype=float)
    for n in range(n_terms):
        result += (a ** n) * np.cos((b ** n) * np.pi * x)
    return result


# Parameters
a, b, n_terms = 0.7, 9, 30
samples_per_unit = 10000
min_zoom = 0.0005
max_zoom = 4
initial_zoom = (min_zoom + max_zoom) / 2  # ~2.00025

# Setup plot
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)
line, = ax.plot([], [], color='#D4A017', linewidth=1.2)
ax.set_facecolor('#F8F0E3')
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title("Manual Deep Zoom into the Weierstrass Function", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("W(x)")

# Slider axis and slider
ax_zoom = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='#F8F0E3')
zoom_slider = Slider(
    ax=ax_zoom,
    label='Zoom Window Width',
    valmin=min_zoom,
    valmax=max_zoom,
    valinit=initial_zoom,
    valstep=0.0005,
    color='#D4A017'
)

# Update function for slider interaction


def update(val):
    zoom = zoom_slider.val
    x = np.linspace(-zoom/2, zoom/2, int(samples_per_unit * zoom))
    y = weierstrass(x, a=a, b=b, n_terms=n_terms)
    line.set_data(x, y)
    ax.set_xlim(-zoom/2, zoom/2)
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
    ax.set_title(f"Zoom Level: ±{zoom/2:.5f}")
    fig.canvas.draw_idle()


# Connect the slider to the update function
zoom_slider.on_changed(update)

# Initial plot draw
update(initial_zoom)

plt.show()
