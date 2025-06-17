# 🌀 2D-Weierstrass Function Visualization Toolkit

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
* **Scientific Accuracy**:
  * Correct dB scaling for FFT (20log₁₀)
  * Angular frequency units (rad/normalized unit)
  * Nyquist-limited spectral display

---

## 📐 Mathematical Foundation

The 2D Weierstrass function is defined as:

$$
\Huge
W(x, y) = \sum_{n=0}^{N} a^n \cdot \cos(\pi b^n x) \cdot \cos(\pi b^n y)
$$

Where:

* a in (0,1) controls **amplitude decay**.
* b in {3, 5, 7, ...} (odd integers) controls **exponential frequency growth**.
* N = 40 is the number of terms used for approximation.

> **Note on Finite Approximation**  
> While the infinite Weierstrass function is continuous everywhere but nowhere differentiable, our visualization uses a finite approximation (\(N = 40\) terms). This truncated version:  
>
> * Is a trigonometric polynomial (finite sum of continuous cosines).  
> * Permits standard FFT analysis (unlike the infinite sum).  
> * Captures emergent fractal properties when $(a \cdot b \geq 1)$.  
>  
> The FFT visualizations show the exact spectral composition of this approximation.

---

**Relationship to Classic 1D Weierstrass Function**  
The 1D slice at \(x = 0\) recovers the original 1D Weierstrass function:  

$$
\Huge
W(0, y) = \sum_{n=0}^{N} a^n \cdot \cos(\pi b^n y)
$$

> **Note on Why Slice at `x = 0`?**  
> We extract a vertical slice at \(x = 0\) for consistent 1D/2D comparison:  
>
> * Aligns with the vertical axis in 2D visualizations.  
> * Shows fractal variation along \(y\).  
> * The 1D FFT of \(W(0, y)\) gives the spectrum **only along \(y\) at \(x=0\)**.  
> * **No direct FFT correspondence**: The 2D FFT shows energy along **diagonals** $(k_x = \pm k_y)$ from $(\cos(\pi b^n x)\ \cos(\pi b^n y))$, with minimal energy at $(k_x = 0)$.  

---

> **Note on Spectral Leakage**  
> The Weierstrass function is continuous but nowhere differentiable, exhibiting **pseudo-periodicity** (local repeating patterns without global periodicity).  
>
> When sampled for FFT:  
>
> * **Spectral leakage** occurs intrinsically due to non-exact periodicity.  
> * Leakage spreads frequency peaks e.g., at $(b^n/2)$ into adjacent bins.  
> * This reflects the fractal’s dense frequency hierarchy and is **not mitigated by windowing/zero-padding**, as artificial periodicity would obscure true fractal properties.  
>  
> Leakage is thus a feature (not a bug) revealing the function’s spectral structure.

---

## 🔍 Visualization Perspectives

### 🖼️ 2D Surface Analysis

**1. Function Values (Spatial Domain)**  
![Raw View](raw_view.png)  
*Illustrates actual output of W(x,y)*  

* **X/Y Axes**: Spatial coordinates [-1, 1] (normalized units)  
* **Color**: Normalized function value (blue→red = min→max)  
* **Key Insight**: Reveals emergent fractal patterns when $a·b ≥ 1$  
* *Pattern Evolution*: Higher $a$ → sharper contrasts; Higher $b$ → finer details  

**2. Value Density (Statistical View)**  
![Density View](density_view.png)  
*Shows spatial distribution of values*  

* **X/Y Axes**: Spatial coordinates [-1, 1]  
* **Color**: Probability density (black→yellow = rare→common)  
* **Key Insight**: Heavy-tailed distributions indicate fractal behavior  
* *Mathematical Note*: Location-independent - captures global statistics  

**3. FFT Spectrum (Frequency Domain)**  
![FFT View](fft_view.png)  
*Visualizes spatial frequency composition*  

* **X/Y Axes**: Angular frequency ω (rad/normalized unit)  
* **Color**: Magnitude in dB (20log₁₀|FFT|)  
* **Signature Features**:  
  * Diagonal peaks at (±πbⁿ, ±πbⁿ)  
  * Four-fold symmetry from cosine products  
  * Power-law decay when fractal ($a·b ≥ 1$)  

### 📏 1D Cross-Section Analysis (x=0)

![1D Weierstrass Function](1d_weierstrass_view.png)  

**1. Function Profile**  
*Classical 1D Weierstrass behavior W(0,y)*  

* **X-axis**: Position along y-axis [-1, 1]  
* **Y-axis**: Function value (normalized)  
* **Characteristic**: Nowhere-differentiable oscillation  
* *Why x=0?*: Simplifies to $\sum a^n\cos(πb^ny)$ - the original 1D form  

**2. Frequency Spectrum**  
*Harmonic composition via stem plot*  

* **X-axis**: Angular frequency (rad/normalized unit, log scale)  
* **Y-axis**: Magnitude (log scale)  
* **Distinctive Signature**:  
  * Discrete stems at ωₙ = πbⁿ  
  * Exponential decay ∝ aⁿ  
  * Equal log-spacing Δlogω = log(b)  

### 📦 Fractal Quantification

![Fractal Dimension View](fractal_dimension.png)  
*Box-counting dimension measurement*  

* **Method**: Occupied boxes vs scale (log-log regression)  
* **Requirements**: $a·b ≥ 1$ (fractal regime)  
* **Interpretation Guidance**:  
  * Theoretical: $D = 3 + \frac{\log a}{\log b}$  
  * Computed typically 0.2-0.3 lower due to:  
    * Finite N=40 truncation  
    * 500×500 resolution limit  

---

## ⚙️ Technical Implementation

### Optimized Computation

```python
@njit
def compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs):
    # Computes FINITE approximation (N=40) 
    # This smooth trigonometric polynomial is FFT-suitable
    W = np.zeros_like(X)
    for n in range(len(a_powers)):  # n=0 to 39
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
    # Use set-based unique box identification
    # Count occupied boxes at different scales
    # Calculate dimension via log-log regression
```

* Optimized using unique identifier sets
* Maintains accuracy while improving performance

### FFT Analysis

```python
# 2D FFT with proper dB scaling
def compute_fft(Z): 
    fft_Z = np.fft.fft2(Z)
    fft_shifted = np.fft.fftshift(fft_Z)
    magnitude = np.abs(fft_shifted)
    # CORRECTED: Standard dB calculation
    return 20 * np.log10(magnitude + 1e-10)

# 1D Frequency processing
freqs_angular = 2 * np.pi * np.fft.fftfreq(n, d=d_spatial)
# Only positive frequencies below Nyquist
pos_mask = (freqs_angular > 0) & (freqs_angular < np.pi * size/2)
pos_angular_freqs = freqs_angular[pos_mask]
```

* **Consistent Angular Frequency Units**:
  * 2D FFT: Angular frequency (rad/normalized unit)
  * 1D FFT: Angular frequency (rad/normalized unit)
* **Nyquist Enforcement**:
  * Angular: $\omega_{\text{Nyquist}} = \pi \cdot \text{size}/2$ rad/normalized unit
  * Cyclic: $f_{\text{Nyquist}} = \text{size}/4$ cycles/normalized unit

---

## 📊 Visualization Legend

| Element          | Raw View (2D)        | Density View (2D)    | FFT View (2D)               | 1D Plot (x=0)        | 1D FFT (Stem)             |
|------------------|----------------------|----------------------|-----------------------------|----------------------|---------------------------|
| **X-axis** | X Coord (norm unit) | X Coord (norm unit)  | ω_x (rad/norm unit)         | Position along y-axis (norm unit)        | Angular Freq (rad/norm unit) |
| **Y-axis** | Y Coord (norm unit) | Y Coord (norm unit)  | ω_y (rad/norm unit)         | W(0,y) Value         | Magnitude (log)           |
| **Color/Lines** | Function value      | Probability density  | **Magnitude (dB)**          | Blue line            | Red stems                 |
| **Range (X/Y)** | [-1, 1]             | [-1, 1]              | [-ω_max, ω_max]             | [-1, 1]              | [0, ω_nyquist]            |
| **Aspect Ratio** | 1:1                 | 1:1                  | 1:1                         | N/A                  | N/A                       |

---

## 🔑 Key Interpretations & Limitations

### 1. Core Mathematical Representation

* **Finite Approximation**: Our implementation uses $N=40$ terms, creating a trigonometric polynomial
* **Fractal Emergence**: True fractal behavior appears when $a\cdot b \geq 1$
* **Mathematical vs Computational**:
  * Infinite series: Requires distributional Fourier analysis
  * Our approximation: FFT-compatible trigonometric polynomial

### 2. Frequency Domain Consistency

* **Angular Frequency Standard**: All FFT visualizations use rad/normalized unit
  * Conversion: 1 rad/norm unit = $\frac{1}{2\pi}$ cycles/norm unit
* **Parameter Mapping**:
  * $b$ controls exponential frequency growth ($\omega_n = \pi b^n$)
  * FFT peaks appear at theoretical frequencies $\pm\pi b^n$

### 3. Spectral Analysis Insights

* **FFT ↔ Theory Relationship**:
  * Theoretical frequencies: $\omega_n = \pi b^n$ rad/norm unit
  * Resolution limits: $\Delta\omega = \pi$ rad/norm unit
  * Max frequency error: $\pm\pi/2$ rad/norm unit
* **Spectral Leakage**:
  * Intrinsic to fractal pseudo-periodicity
  * Not an artifact: Reveals hierarchical frequency structure
  * Deliberately not windowed to preserve fractal properties

### 4. Critical Visualization Considerations

* **Logarithmic Scaling Effects**:
  * Equal stem spacing: $\Delta_{\log} = \log b$
  * Exponential amplitude decay: $a^n$ scaling
* **Nyquist Enforcement**:
  * $\omega_{\text{nyq}} = \pi \cdot \frac{\text{size}}{2}$
  * Highest visible harmonic: $\max(\omega_n < \omega_{\text{nyq}})$
  * Example (b=5, size=500): Visible n=0-3 ($\pi$ to $125\pi$)

### 5. Fractal Dimension Context

* **Theoretical Basis**: $D_{\text{theory}} = 3 + \frac{\log a}{\log b}$
* **Computation Limitations**:
  * Finite terms (N=40): Missing high-frequency details
  * Resolution limit (500×500 grid): Sub-pixel structures unresolved
  * Box-size range (ε=0.02-0.20): Limited scaling regime
* **Interpretation Guidance**:
  > "The measured dimension characterizes our finite approximation, not the infinite mathematical ideal. It remains a valid indicator of emergent fractal behavior."

---

## 🧩 Parameter Effects

| Parameter Change | Raw View          | Density View       | FFT View (2D)          | 1D Plot           | 1D FFT (Stem)     | Dimension   |
|------------------|-------------------|--------------------|------------------------|-------------------|-------------------|-------------|
| **a ↑** | Sharper contrasts | Wider distribution | More HF energy         | Larger amplitudes | Higher stems      | ↑ (0.1-0.3) |
| **b ↑** | Finer details     | More complex peaks | Energy shifts right    | More oscillations | Stems shift right + compress | ↑ (0.1-0.4) |
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

1. Weierstrass, K. (1872). Über continuierliche Funktionen eines reellen Arguments, die für keinen Werth desselben einen bestimmten Differentialquotienten besitzen.  
   *Mathematische Werke*, Vol 2. [Archive.org](https://archive.org/details/mathematischewer02weieuoft)
2. Falconer, K. (2013). *Fractal Geometry: Mathematical Foundations and Applications*
3. Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*
4. Oppenheim, A. V. (1999). *Discrete-Time Signal Processing* (FFT scaling conventions)
