# Fractal-Driven Chaotic Oscillator

![Comparison](oscillator_comparison.png)

## 🔍 Overview

This project demonstrates how **Weierstrass fractal modulation** transforms a simple nonlinear oscillator into a complex, chaotic system. The side-by-side visualization contrasts a smooth, quasi-periodic baseline against the fractally driven system exhibiting highly irregular behavior.

> 💡 **Note:** The "original oscillator" is *not* a parametric oscillator in the strict physics sense, since it uses constant parameters. Only the **Weierstrass-driven version**, with time-dependent modulation, qualifies as a **parametrically modulated nonlinear oscillator**.

---

## 📊 Mathematical Core

### Original Oscillator (Non-Autonomous Benchmark)

A simple coupled nonlinear oscillator:

$$
\begin{aligned}
\frac{dx}{dt} &= \sin(\alpha y) \\\\
\frac{dy}{dt} &= \cos(\alpha x)
\end{aligned}
$$

with $\alpha = 2.0$ constant. This produces **smooth, bounded, quasi-periodic motion** due to its autonomous structure.

---

### Weierstrass Modulation (Fractal Parametric Oscillator)

We introduce **time-dependent parameters** using the **Weierstrass function**, a classic example of a continuous but nowhere-differentiable function:

$$
W(t) = \sum_{n=0}^{30} a^n \cos(\pi b^n t)
$$

Using $a = 0.5$, $b = 3.0$, this creates wild, fractal behavior in time. Parameters evolve as:

$$
\begin{aligned}
\alpha(t) &= 2 + 2.15 \cdot W(t) \\\\
\beta(t)  &= 2 - 1.1 \cdot W(t + 0.1)
\end{aligned}
$$

The modified system becomes:

$$
\begin{aligned}
\frac{dx}{dt} &= \sin(\alpha(t) \cdot y) \\\\
\frac{dy}{dt} &= \cos(\beta(t) \cdot x)
\end{aligned}
$$

---

## 🧠 Physics Interpretation

In physics, **parametric oscillators** involve systems whose parameters (e.g., frequency or stiffness) change with time, often used in:

* **Quantum optics** (e.g., parametric down-conversion \[Boyd, *Nonlinear Optics*, 3rd ed.])
* **Classical mechanics** (e.g., Mathieu oscillator with a time-varying spring)
* **Plasma waves**, **Josephson junctions**, and **NEMS/MEMS** systems

By driving the system with a **fractal function**, we simulate a highly irregular "environmental coupling," mimicking phenomena like:

* **Quasiperiodic forcing** with no characteristic scale
* **Chaotic modulation** beyond sinusoidal or random noise
* **Anomalous diffusion** and **irregular attractor structures**

This opens a path for analyzing **non-smooth dynamical systems**, relevant to:

* **Fractal time crystals**
* **Stochastic control systems**
* **Biophysical oscillations in complex environments**

---

## 📈 Key Effects

| Feature               | Original Oscillator         | Fractal-Driven Oscillator       |
| --------------------- | --------------------------- | ------------------------------- |
| **Time Dependence**   | Constant $\alpha$           | Fractal $\alpha(t), \beta(t)$   |
| **Phase Space**       | Smooth Lissajous-like loops | Fractal, chaotic attractors     |
| **Differentiability** | Smooth everywhere           | Nowhere-differentiable forcing  |
| **Stability**         | Regular                     | Sensitive to initial conditions |

---

## 🎨 Customization

You can customize behavior by adjusting amplitudes and scaling:

```python
# For sparse, fragmented attractors
generate_comparison(a_amp=1.5, b_amp=-0.8)

# For dense, organic chaos
generate_comparison(a_amp=3.0, b_amp=-2.0, scale=1.8)
```

---

## 📚 References

* G.H. Hardy, *Weierstrass’s Non-Differentiable Function*, Cambridge (1916)
* L.E. Reichl, *The Transition to Chaos: Conservative Classical Systems and Quantum Manifestations*, Springer (2004)
* R.C. Hilborn, *Chaos and Nonlinear Dynamics*, Oxford University Press (2000)
* R.W. Boyd, *Nonlinear Optics*, 3rd ed., Academic Press (2008)
* K. Falconer, *Fractal Geometry: Mathematical Foundations and Applications*, Wiley (2003)

---

