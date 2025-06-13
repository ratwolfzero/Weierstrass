# Weierstrass Function Visualization Toolkit

![Weierstrass Fractal](overview.png)

## 📌 Overview

This interactive Python tool visualizes the **2D Weierstrass function**—a fascinating extension of the classic 1D function known for being **continuous everywhere but differentiable nowhere**. The application provides multiple perspectives on this mathematical curiosity, including:

1. Raw function visualization
2. Density distribution mapping
3. Frequency spectrum analysis (FFT)
4. Fractal dimension calculation

---

## 🚀 Key Features

* **Interactive controls** for parameters `a` (amplitude decay) and `b` (frequency scaling)
* **Multiple visualization modes**:
  * Raw function values
  * Value density distribution
  * Fast Fourier Transform (FFT) spectrum
* **Box-counting dimension** calculation for fractal analysis
* **Real-time updates** via Numba-accelerated computation
* **Visual heuristic** indicating when fractal behavior emerges (a·b ≥ 1)

---

## 📐 Mathematical Foundation

The 2D Weierstrass function is defined as:

$$
\Huge
W(x, y) = \sum_{n=0}^{N} a^n \cdot \cos(\pi b^n x) \cdot \cos(\pi b^n y)
$$

Where:

* $a \in (0,1)$ controls **amplitude decay**
* $b \in \{3, 5, 7, \dots\}$ (odd integers) controls **frequency growth**
* $N = 20$ is the number of terms used for approximation

---

## 🔍 Visualization Modes

### 1. Raw Function Values

![Raw View](raw_view.png)

* **X/Y Axes**: Spatial coordinates in [-1, 1] range
* **Color**: Normalized function value (blue = negative, red = positive)
* **Title**: "Normalized 2D Weierstrass Function"
* Shows actual output of the mathematical function

### 2. Density Approximation

![Density View](density_view.png)

* **X/Y Axes**: Spatial coordinates in [-1, 1] range
* **Color**: Probability density of values
* **Title**: "Value Probability Density"
* Reveals value distribution independent of location

### 3. FFT Spectrum

![FFT View](fft_view.png)

* **X/Y Axes**: Angular frequency (rad/sample)
* **Color**: Log-magnitude (dB scale)
* **Title**: "Frequency Spectrum"
* Shows dominant spatial frequencies and orientations

### 4. Box-Counting Dimension

![Fractal Dimension View](fractal_dimension.png)

* Calculates fractal dimension using box-counting method
* Requires `a·b ≥ 1` (fractal condition)
* Displayed in plot title when calculated

---

## ⚙️ Technical Implementation

### Optimized Computation

```python
@njit
def compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs):
    W = np.zeros_like(X)
    for n in range(len(a_powers)):
        W += a_powers[n] * np.cos(b_freqs[n] * X) * np.cos(b_freqs[n] * Y)
    return W
```

* Uses Numba JIT compilation for 100x speedup
* Precomputes power series for efficiency

### Box-Counting Algorithm

```python
@njit
def box_counting_dimension(Z, epsilons):
    # Normalize Z to [0,1]
    # Create 3D grid (x, y, value)
    # Count occupied boxes at different scales
    # Calculate dimension via log-log regression
```

* Operates in normalized value space
* Uses linear regression on log-scale data

### FFT Analysis

```python
def compute_fft(Z):
    fft_Z = np.fft.fft2(Z)
    fft_shifted = np.fft.fftshift(fft_Z)
    return np.log10(np.abs(fft_shifted) + 1e-10)
```

* Computes 2D Fourier transform
* Shifts zero-frequency to center
* Applies logarithmic scaling

---

## 📊 Visualization Legend

| Element          | Raw View        | Density View      | FFT View               |
|------------------|-----------------|-------------------|------------------------|
| **X-axis**       | X Coordinate    | X Coordinate      | ω_x (rad/sample)       |
| **Y-axis**       | Y Coordinate    | Y Coordinate      | ω_y (rad/sample)       |
| **Color**        | Function value  | Probability       | Log-magnitude (dB)     |
| **Range**        | [-1, 1]         | [-1, 1]           | [-π, π] rad/sample    |
| **Aspect Ratio** | 1:1             | 1:1               | 1:1                    |

---

## 🔑 Key Clarifications

1. **Two distinct "frequency" concepts:**
   * **Parameter b**: Controls term frequencies in *function definition*
   * **FFT analysis**: Measures spatial frequencies in *visual output*
   * These are related but separate concepts

2. **Density vs FFT:**
   * Density shows **value occurrence frequency**
   * FFT shows **pattern repetition frequency**

3. **Practical interpretation:**
   * Higher `b` → More fine details → More FFT energy at periphery
   * Higher `a` → Sharper contrasts → Wider value distribution
   * `a·b ≥ 1` → Fractal behavior → Valid dimension calculation

---

## 🧩 Parameter Effects

| Parameter Change | Raw View          | Density View       | FFT View               | Dimension   |
|------------------|-------------------|--------------------|------------------------|-------------|
| **a ↑**          | Sharper contrasts | Wider distribution | More high-frequency energy | ↑ (0.1-0.3) |
| **b ↑**          | Finer details     | More complex peaks | Energy shifts outward   | ↑ (0.1-0.4) |
| **a·b ≥ 1**      | Fractal patterns  | Heavy tails        | Power-law spectrum      | Valid result|

---

## ▶️ Getting Started

* Install requirements:

```bash
pip install numpy matplotlib numba
```

* Run the script:

```bash
python weierstrass_fractal_FFT_Box_Count.py
```

* Interact with controls:
  * Adjust `a` and `b` sliders
  * Toggle visualization modes
  * Click "Calculate Box-Counting Dimension" for fractal analysis

---

## 📚 References

1. Weierstrass, K. (1872). On continuous functions of a real argument that do not have a well-defined differential quotient.
2. Falconer, K. (2013). Fractal Geometry: Mathematical Foundations and Applications.
3. Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
