
# 🔍 Manual Deep Zoom Visualization of the Weierstrass Function

This project visualizes the **Weierstrass function**, a classic example of a **continuous but nowhere differentiable function**, using a manual zoom slider that acts like a microscope revealing intricate fractal details.

![Weierstrass](weierstrass_1.png)

## ✨ Features

- 🔎 Zoom slider allowing deep zoom from a wide view (±2 units) down to extremely fine scales (±0.00025 units).
- 🎯 High sample density adapts dynamically to zoom level to maintain smoothness.
- 📈 Dynamic y-axis scaling to reveal the local amplitude fluctuations at each zoom level.
- 🌿 Clear visualization of fractal self-similarity and infinite oscillations at every scale.

## 📐 Mathematical Background

The Weierstrass function is defined as:

$$
W(x) = \sum_{n=0}^{\infty} a^n \cos\left(b^n \pi x\right)
$$

with constraints:

- \( 0 < a < 1 \)
- \( b \) is an odd integer
- \( a b > 1 + $\frac{3\pi}{2} \$) (ensures nowhere differentiability)

This function is continuous everywhere but differentiable nowhere, exhibiting fractal-like oscillations at arbitrarily small scales.

## 🔍 Visual Constraints at High Zoom Levels

When zoomed in deeply (e.g., ±0.00025 units), you may notice the function appears as a single smooth peak or wave. This is consistent with the mathematical nature of the function because:

- The viewing window is extremely narrow, showing a very small portion of the fractal.
- The function remains fractal and nowhere differentiable, but visual resolution and sampling density limit visible details.
- Increasing sample density further would reveal more micro-oscillations, but practical limits exist.

Thus, the “single peak” appearance at highest zoom levels reflects the fractal tip rather than smoothness.

## 🚀 Usage

Run the Python script, and use the slider to zoom in and out manually. Observe how the local shape of the Weierstrass function changes with scale, illustrating its fractal complexity.

---

## 📁 References

- K. Weierstrass, *Mathematische Werke*, Vol 2, 1872. [Archive.org](https://archive.org/details/mathematischewer02weieuoft/page/n101)  
- E.W. Weisstein, “Weierstrass Function”, [MathWorld](https://mathworld.wolfram.com/WeierstrassFunction.html)

---

Enjoy exploring this fascinating mathematical object! 🍻
