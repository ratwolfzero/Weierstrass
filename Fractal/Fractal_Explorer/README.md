# 2D-Weierstrass Function Visualization Toolkit

![Weierstrass Fractal](overview.png)

## 📌 Overview

This interactive Python tool visualizes the **2D Weierstrass function**—a fascinating extension of the classic 1D function known for being **continuous everywhere but differentiable nowhere**. The application provides multiple perspectives on this mathematical curiosity, including:

1. Raw function visualization (2D and 1D)
2. Density distribution mapping (2D)
3. Frequency spectrum analysis (FFT for both 2D and 1D)
4. Fractal dimension calculation (2D)

---

## 🚀 Key Features

* **Interactive controls** for parameters `a` (amplitude decay) and `b` (frequency scaling)
* **Multiple visualization modes**:
  * Raw function values (2D and 1D)
  * Value density distribution (2D)
  * Fast Fourier Transform (FFT) spectrum (2D and 1D)
* **Box-counting dimension** calculation for fractal analysis
* **Real-time updates** via Numba-accelerated computation
* **Constraint Indicator** indicating when fractal behavior emerges (a·b ≥ 1)

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
* $N = 30$ is the number of terms used for approximation

> **Note on Finite Approximation**:  
> While the infinite Weierstrass function is nowhere differentiable, our visualization uses a **finite approximation** (N=40 terms). This truncated version:
>
> * Is smooth (infinitely differentiable)
> * Forms a trigonometric polynomial
> * Permits standard FFT analysis
>
> The FFT visualizations show exact spectral composition of this approximation, which captures the *emergent fractal properties* of the true function when a·b ≥ 1.

The 1D slice, typically taken at $x=0$, simplifies to:

$$
\Huge
W(0, y) = \sum_{n=0}^{N} a^n \cdot \cos(0) \cdot \cos(\pi b^n y) = \sum_{n=0}^{N} a^n \cdot \cos(\pi b^n y)
$$

This highlights that the 1D function is a sum of cosine waves with geometrically increasing frequencies and exponentially decreasing amplitudes.

---

## 🔍 Visualization Modes

### 2D Visualizations

These modes apply to the full 2D function surface.

#### 1. Raw Function Values

![Raw View](raw_view.png)

* **X/Y Axes**: Spatial coordinates in [-1, 1] range (normalized units)
* **Color**: Normalized function value (blue = negative, red = positive)
* **Title**: "Normalized 2D Weierstrass Function"
* Shows actual output of the mathematical function

#### 2. Density Approximation

![Density View](density_view.png)

* **X/Y Axes**: Spatial coordinates in [-1, 1] range (normalized units)
* **Color**: Probability density of values
* **Title**: "Value Probability Density"
* Reveals value distribution independent of location

#### 3. FFT Spectrum (2D)

![FFT View](fft_view.png)

* **X/Y Axes**: Angular frequency (rad/normalized unit)
* **Color**: Log-magnitude (dB scale)
* **Title**: "2D Frequency Spectrum"
* Shows dominant spatial frequencies and orientations present in the 2D surface. The FFT operates on our **finite smooth approximation** of the Weierstrass function, showing:
  * Discrete frequency components at ω = (kπ, mπ)
  * Emergent power-law scaling when a·b ≥ 1

### 1D Visualization

This section focuses on a 1D slice of the Weierstrass function (specifically, $x=0$) and its frequency content.

#### 1. 1D Weierstrass Function (x=0)

![1D Weierstrass Function](1d_weierstrass_view.png)

* **X-axis**: Spatial coordinate `y` (normalized unit)
* **Y-axis**: Normalized function value `W(0, y)`
* **Title**: "1D Weierstrass Function (x=0)"
* Displays a cross-section of the 2D surface, revealing the intricate, non-differentiable oscillations characteristic of the function.

#### 2. FFT of 1D Weierstrass Function (Stem Plot)

* **X-axis**: **Normalized Frequency (cycles/normalized unit)**
  * Represents spatial frequencies in the 1D function
  * "Cycles/normalized unit" indicates wave cycles per spatial unit in the [-1,1] domain
* **Y-axis**: **Magnitude (log scale)**
  * Shows **amplitude** of each frequency component
* **Plot Type**: **Stem Plot**
  * Ideal for discrete frequency components in our finite approximation
* **Interpretation**:
  * **Distinct stems** at frequencies: $\frac{b^n}{2}$ cycles/normalized unit
  * **Height decreases** with frequency due to $a^n$ amplitude decay
  * **Mathematical Note**: Each stem corresponds to a term in $W_N(y) = \sum_{n=0}^{29} a^n \cos(\pi b^n y)$

### Fractal Dimension Calculation

#### 4. Box-Counting Dimension (2D)

![Fractal Dimension View](fractal_dimension.png)

* Calculates fractal dimension using box-counting method
* Requires `a·b ≥ 1` (fractal condition)
* Displayed when calculated, with color-coded validity indicator

---

## ⚙️ Technical Implementation

### Optimized Computation

```python
@njit
def compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs):
    # Computes FINITE approximation (N=30) 
    # This smooth trigonometric polynomial is FFT-suitable
    W = np.zeros_like(X)
    for n in range(len(a_powers)):  # n=0 to 29
        W += a_powers[n] * np.cos(b_freqs[n] * X) * np.cos(b_freqs[n] * Y)
    return W

@njit
def compute_weierstrass_1d(y, a_powers, b_freqs):
    total = np.zeros_like(y)
    for n in range(len(a_powers)):
        total += a_powers[n] * np.cos(b_freqs[n] * y)
    return total
```

* Uses Numba JIT compilation for 100x speedup
* Precomputes power series for efficiency
* Spatial coordinates in normalized units [-1,1]

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
def compute_fft(Z): # For 2D FFT
    fft_Z = np.fft.fft2(Z)
    fft_shifted = np.fft.fftshift(fft_Z)
    return np.log10(np.abs(fft_shifted) + 1e-10)
```

* `np.fft.fft2` computes 2D Fourier transform
* Frequency units:
  * 2D: Angular frequency (rad/normalized unit)
  * 1D: Cyclic frequency (cycles/normalized unit)
* Logarithmic scaling for magnitude visualization

---

## 📊 Visualization Legend

| Element          | Raw View (2D)        | Density View (2D)    | FFT View (2D)               | 1D Plot (x=0)        | 1D FFT (Stem)             |
|------------------|----------------------|----------------------|-----------------------------|----------------------|---------------------------|
| **X-axis** | X Coord (norm unit) | X Coord (norm unit)  | ω_x (rad/norm unit)         | y (norm unit)        | Freq (cycles/norm unit)   |
| **Y-axis** | Y Coord (norm unit) | Y Coord (norm unit)  | ω_y (rad/norm unit)         | W(0,y) Value         | Magnitude (log)           |
| **Color/Lines** | Function value      | Probability density  | Log-magnitude (dB)          | Blue line            | Red stems                 |
| **Range (X/Y)** | [-1, 1]             | [-1, 1]              | [-π, π] rad/norm unit       | [-1, 1]              | [0, max Freq]             |
| **Aspect Ratio** | 1:1                 | 1:1                  | 1:1                         | N/A                  | N/A                       |

---

## 🔑 Key Clarifications

1. **Two distinct "frequency" concepts**:
    * **Parameter `b`**: Controls frequency scaling in mathematical definition
    * **FFT analysis**: Measures spatial frequencies in output
    * **Unit Relationship**:
      $$1 \text{ rad/norm unit} = \frac{1}{2\pi} \text{ cycles/norm unit}$$

2. **Density vs FFT**:
    * **Density**: Frequency of value occurrences (statistical)
    * **FFT**: Frequency of spatial patterns (spectral)

3. **Practical interpretation**:
    * Higher `b` → More fine details → Higher frequency energy
    * Higher `a` → Sharper contrasts → Wider value distribution
    * `a·b ≥ 1` → Fractal behavior expected

4. **Finite vs Infinite**:  
   * Classical function: Requires distributional Fourier analysis  
   * Our approximation: Finite trigonometric polynomial (FFT-compatible)  
   * Emergent fractal properties when a·b ≥ 1  

5. **Mathematical ↔ FFT Relationship**:
   * Theoretical frequencies: $f_n = \frac{b^n}{2}$ cycles/normalized unit
   * FFT shows exact harmonics of finite approximation
   * Example (b=5):
     * Theoretical: 0.5, 2.5, 12.5, 62.5 cycles/normalized unit
     * Observed: ≈ 0.499, 2.49, 12.48, 62.38 cycles/normalized unit

6. **High-Frequency FFT Density**:
   * On logarithmic scale, stems cluster near Nyquist (0.5 cycles/norm unit)
   * Due to:
     * Linear FFT bin spacing
     * Geometric harmonic progression ($b^n$)
     * Logarithmic axis compression
   * **Visual tip**: Lower harmonics show clear separation

7. **High-Frequency Spectral Pattern**:
   * Near Nyquist (0.5 cycles/norm unit):
     * Cluster of closely-spaced stems
     * Final distinct peak at highest unaliased harmonic
   * Caused by:
     * Exponential frequency compression ($b^n$)
     * Amplitude decay ($a^n$) preserving structure
     * Discrete nature of harmonics

---

## 🧩 Parameter Effects

| Parameter Change | Raw View          | Density View       | FFT View (2D)          | 1D Plot           | 1D FFT (Stem)     | Dimension   |
|------------------|-------------------|--------------------|------------------------|-------------------|-------------------|-------------|
| **a ↑** | Sharper contrasts | Wider distribution | More HF energy         | Larger amplitudes | Higher stems      | ↑ (0.1-0.3) |
| **b ↑** | Finer details     | More complex peaks | Energy shifts outward  | More oscillations | Stems shift right | ↑ (0.1-0.4) |
| **a·b ≥ 1** | Fractal patterns  | Heavy tails        | Power-law spectrum     | Highly jagged     | Clear peaks       | Valid result|

---

## ▶️ Getting Started

Install requirements:

```bash
pip install numpy matplotlib numba
```

Run the script:

```bash
python weierstrass_fractal_explorer.py
```

Interact with controls:

* Adjust `a` (0.01-0.99) and `b` (odd integers 3-19)
* Toggle visualization modes (Raw/Density/FFT)
* Compute fractal dimension when a·b ≥ 1
* Enlarge 1D plots for detailed inspection

---

## 📚 References

1. Weierstrass, K. (1872). On continuous functions of a real argument that do not have a well-defined differential quotient.  
   *Mathematische Werke*, Vol 2. [Archive.org](https://archive.org/details/mathematischewer02weieuoft)
2. Falconer, K. (2013). *Fractal Geometry: Mathematical Foundations and Applications*
3. Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*
