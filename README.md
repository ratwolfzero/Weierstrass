# 🔍 **Exploring the Weierstrass Function with Deep Zoom**

✅ Continuous everywhere | ❌ Differentiable nowhere

![Weierstrass](weierstrass_1.png)

The Weierstrass function stands as one of the most iconic examples in mathematical analysis: a curve that is continuous at every point yet differentiable nowhere. In other words, it lacks a tangent anywhere — it’s infinitely “rough,” exhibiting intricate wiggles at every scale.

---

## 📐 **The Mathematics Behind It**

Formally, the function is defined as an infinite sum:

$$
\Huge
W(x) = \sum_{n=0}^{\infty} a^{n} \cos\left(b^{n} \pi x\right)
$$

subject to the conditions:

* $\large 0 < a < 1$
* $\large b$ is an odd integer
* $\large a \times b > 1 + \frac{3\pi}{2}$

These constraints guarantee the function’s fractal, nowhere differentiable behavior, ensuring it oscillates wildly no matter how closely you zoom in.

---

## 🎨 **Visualizing the Function**

Visualizing such a fractal object poses challenges — zooming in reveals ever more oscillations without any smoothening. To explore this phenomenon interactively, I developed a Python visualization featuring a manual zoom slider functioning like a microscope.

Starting from a broad view (±2 units), you can zoom in as deeply as ±0.00025 units. The plot dynamically updates its sampling density to keep the curve smooth at every scale. The y-axis rescales automatically to highlight the local amplitude of oscillations clearly.

---

## 🔍 **What Happens at Extreme Zoom?**

At the highest zoom levels, the function’s graph may visually appear as a single peak or smooth segment. This, however, does **not** imply differentiability. Instead:

* The narrow viewing window captures just one local wavelet.
* Finer oscillations exist beyond the visible resolution or sample density.
* Increasing the sample density further reveals smaller-scale oscillations, endlessly unfolding the fractal complexity.

Thus, the visualization remains mathematically rigorous, respecting the infinite intricacy of the Weierstrass function, while providing an intuitive, microscope-like exploration experience.

---

## 🚀 **Try It Yourself**

Run the provided Python script to interactively explore the Weierstrass function’s fractal structure. Move the zoom slider and watch this elegant mathematical object reveal detail at every scale — a vivid demonstration of how continuity and roughness coexist in analysis.

This hands-on visualization bridges abstract theory and tangible intuition, highlighting why the Weierstrass function continues to captivate mathematicians and educators alike.

---

## 📁 **References**

* K. Weierstrass, *Mathematische Werke*, Vol 2, 1872. [Archive.org](https://archive.org/details/mathematischewer02weieuoft/page/n101)
* E.W. Weisstein, “Weierstrass Function,” [MathWorld](https://mathworld.wolfram.com/WeierstrassFunction.html)
