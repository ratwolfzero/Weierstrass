# Weierstrass Function Density Visualization

![Weierstrass Fractal](fractal_1.png)

*Example output for* `a = 0.3`, `b = 5`

---

## 📌 Overview

This project visualizes the **2D Weierstrass function**—a fascinating extension of the classic 1D function known for being **continuous everywhere but differentiable nowhere**. The 2D version creates rich, fractal-like textures that reveal hidden mathematical complexity in both dimensions.

---

## Key Features

* Interactive sliders for parameters `a` (amplitude decay) and `b` (frequency scaling, **odd integers only**)
* Real-time updates via **Numba-accelerated computation**
* **Histogram-based density approximation** to highlight value distributions in a more perceptual way

---

## 📐 Mathematical Definition

The 2D Weierstrass function is defined as:

$$
W(x, y) = \sum_{n=0}^{N} a^n \cdot \cos(\pi b^n x) \cdot \cos(\pi b^n y)
$$

Where:

* $a \in (0,1)$ controls **amplitude decay**
* $b \in \{3, 5, 7, \dots\}$ (odd integers) controls **frequency growth**
* $N = 20$ is the number of terms used for finite approximation

---

## Key Properties

| Property                   | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| **Continuity**             | Uniformly convergent sum → continuous surface                |
| **Nowhere differentiable** | No well-defined tangent plane at any point                   |
| **Self-similar**           | Fractal behavior at all zoom levels                          |
| **Parameter-sensitive**    | Small changes in `a` or `b` yield visually distinct patterns |

---

## Why 2D?

The 1D Weierstrass function produces jagged lines—interesting, but limited. The 2D version, however, produces stunning fractal surfaces, useful for:

* Terrain generation
* Procedural textures
* Studying chaotic or self-similar systems
* Aesthetic/organic mathematical art

---

## 📊 Interpreting the Visualization

### What You’re Seeing

* **Color intensity = density of values**

  * **Darker** = more frequently occurring values
  * **Brighter** = rarer, more extreme values

### Parameter Effects

| Parameter | Role                      | Visual Impact                                |
| --------- | ------------------------- | -------------------------------------------- |
| `a`       | Controls term amplitude   | ↑ `a` = rougher terrain, more sharp contrast |
| `b`       | Controls frequency growth | ↑ `b` = more fine-grained fractal detail     |

---

### Example Behaviors

* **Low `a` (0.1–0.3)** → Smooth, flowing waveforms
* **High `a` (0.7–0.9)** → Jagged, mountainous terrain
* **Low `b` (3–7)** → Coarse structure dominates
* **High `b` (15–19)** → Rich detail, complex self-similarity