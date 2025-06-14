import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from numba import njit
import matplotlib.gridspec as gridspec


class WeierstrassVisualizer:
    """Interactive visualization of the Weierstrass function with various views and analysis tools.

    Args:
        size: Size of the grid for the 2D visualization
        N: Number of terms in the Weierstrass series
        bins: Number of bins for density calculation
    """

    def __init__(self, size: int = 500, N: int = 30, bins: int = 500) -> None:
        """Initialize the Weierstrass visualizer with default parameters."""
        self.size = size
        self.N = N
        self.bins = bins
        self.odd_b_values = np.arange(3, 21, 2)

        # Initialize coordinate grids
        self._setup_coordinates()

        # Initialize parameters
        self.init_a = 0.22
        self.init_b = 5

        # Initialize GUI elements
        self._setup_gui()

        # Initialize plot
        self.update_plot(None)

    # Numba-accelerated functions
    @staticmethod
    @njit
    def compute_weierstrass_1d(y, a_powers, b_freqs):
        total = np.zeros_like(y)
        for n in range(len(a_powers)):
            total += a_powers[n] * np.cos(b_freqs[n] * y)
        return total

    @staticmethod
    @njit
    def compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs):
        W = np.zeros_like(X)
        for n in range(len(a_powers)):
            W += a_powers[n] * np.cos(b_freqs[n] * X) * np.cos(b_freqs[n] * Y)
        return W

    @staticmethod
    def compute_density_approx(values, bins=500):
        hist, bin_edges = np.histogram(values, bins=bins, density=True)
        bin_indices = np.digitize(values, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
        return hist[bin_indices]

    @staticmethod
    def compute_fft(Z):
        fft_Z = np.fft.fft2(Z)
        fft_shifted = np.fft.fftshift(fft_Z)
        magnitude = np.abs(fft_shifted)
        return np.log10(magnitude + 1e-10)

    @staticmethod
    @njit
    def box_counting_dimension(Z, epsilons):
        size = Z.shape[0]
        counts = np.zeros(len(epsilons))

        # Normalize Z to [0,1] range
        z_min = np.min(Z)
        z_max = np.max(Z)
        z_range = z_max - z_min + 1e-9
        Z_norm = (Z - z_min) / z_range

        for i in range(len(epsilons)):
            eps = epsilons[i]
            box_size_xy = max(1, int(np.ceil(eps * size)))
            box_size_z = eps

            grid_x = (size + box_size_xy - 1) // box_size_xy
            grid_y = (size + box_size_xy - 1) // box_size_xy
            grid_z = max(1, int(np.ceil(1.0 / box_size_z)))

            occupied = np.zeros((grid_x, grid_y, grid_z), dtype=np.bool_)

            for x in range(size):
                for y in range(size):
                    bx = x // box_size_xy
                    by = y // box_size_xy
                    bz = int(Z_norm[x, y] / box_size_z)
                    bz = min(bz, grid_z - 1)

                    if not occupied[bx, by, bz]:
                        occupied[bx, by, bz] = True

            counts[i] = np.sum(occupied)

        sum_x = sum_y = sum_xy = sum_x2 = 0.0
        n = len(epsilons)
        for i in range(n):
            log_eps = np.log(epsilons[i])
            log_count = np.log(counts[i])
            sum_x += log_eps
            sum_y += log_count
            sum_xy += log_eps * log_count
            sum_x2 += log_eps * log_eps

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return -slope
       
    def _setup_coordinates(self) -> None:
        """Initialize coordinate grids and FFT frequency coordinates."""
        # Initialize coordinate grids
        self.x = np.linspace(-1, 1, self.size)
        self.y = np.linspace(-1, 1, self.size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # For density view using pcolormesh
        self.x_edges = np.linspace(-1, 1, self.size+1)
        self.y_edges = np.linspace(-1, 1, self.size+1)

        # Initialize FFT frequency coordinates
        # Compute frequencies in cycles/sample
        freq_cycles_x = np.fft.fftshift(np.fft.fftfreq(self.size, d=2/self.size))
        freq_cycles_y = np.fft.fftshift(np.fft.fftfreq(self.size, d=2/self.size))
    
        # Convert to angular frequency (rad/sample) = 2π × cycles/sample
        self.freq_x = freq_cycles_x * 2 * np.pi
        self.freq_y = freq_cycles_y * 2 * np.pi
    
        self.extent_freq = [self.freq_x[0],
                        self.freq_x[-1], self.freq_y[0], self.freq_y[-1]]

    def _setup_gui(self) -> None:
        """Set up the graphical user interface components."""
        # Create figure with improved layout
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.4, 0.6], height_ratios=[1, 1],
                               top=0.95, bottom=0.35, hspace=0.3, wspace=0.3)

        # Set up plot axes
        self._setup_plot_axes(gs)

        # Set up controls
        self._setup_controls()

        # Initialize state
        self.current_Z_norm = None
        self.current_1d_data = None
        self.current_1d_fft_freqs = None
        self.current_1d_fft_mag = None
        self.current_dimension = None
        self.current_plot = None
        self.dimension_calculated = False
        self.last_a = self.init_a
        self.last_b = self.init_b
        self.fft_stem = None

        # Bind events
        self.slider_a.on_changed(self.update_plot)
        self.slider_b.on_changed(self.update_plot)
        self.radio_buttons.on_clicked(self.update_plot)
        self.button.on_clicked(self.calculate_dimension)
        self.enlarge_button.on_clicked(self.enlarge_1d_plots)

    def _setup_plot_axes(self, gs: gridspec.GridSpec) -> None:
        """Set up all plot axes in the figure.

        Args:
            gs: GridSpec object for layout
        """
        # Main 2D plot
        self.ax0 = plt.subplot(gs[:, 0])
        self.im = self.ax0.imshow(
            np.zeros((self.size, self.size)), extent=(-1, 1, -1, 1))
        self.cbar = plt.colorbar(self.im, ax=self.ax0)
        self.title_2d = self.ax0.set_title('')

        # 1D Weierstrass plot
        self.ax1 = plt.subplot(gs[0, 1])
        self.line_1d, = self.ax1.plot(self.y, np.zeros_like(self.y), 'b-')
        self.ax1.set_title('1D Weierstrass Function (x=0)')
        self.ax1.set_xlabel('y')
        self.ax1.set_ylabel('W(y)')
        self.ax1.grid(True)

        # FFT plot
        self.ax2 = plt.subplot(gs[1, 1])
        self.ax2.set_title('FFT of 1D Weierstrass (Stem Plot)')
        self.ax2.set_xlabel('Frequency (cycles/sample)')
        self.ax2.set_ylabel('Magnitude (log scale)')
        self.ax2.set_xscale('log')
        self.ax2.set_yscale('log')
        self.ax2.grid(True)

    def _setup_controls(self) -> None:
        """Set up all interactive controls (sliders, buttons, etc.)."""
        # Sliders
        ax_a_pos = [0.1, 0.25, 0.65, 0.03]
        ax_b_pos = [0.1, 0.18, 0.65, 0.03]

        self.ax_a = plt.axes(ax_a_pos)
        self.slider_a = Slider(self.ax_a, '', 0.01, 0.99,
                               valinit=self.init_a, valstep=0.01)

        self.ax_b = plt.axes(ax_b_pos)
        self.slider_b = Slider(self.ax_b, '', self.odd_b_values[0], self.odd_b_values[-1],
                               valinit=self.init_b, valstep=2)

        # Labels above sliders
        label_a_y = ax_a_pos[1] + ax_a_pos[3] + 0.01
        label_b_y = ax_b_pos[1] + ax_b_pos[3] + 0.01

        self.fig.text(ax_a_pos[0], label_a_y, 'Amplitude Decay (a)',
                      verticalalignment='bottom', horizontalalignment='left',
                      fontsize=10, weight='bold')

        self.fig.text(ax_b_pos[0], label_b_y, 'Frequency Growth (b)',
                      verticalalignment='bottom', horizontalalignment='left',
                      fontsize=10, weight='bold')

        # Radio buttons
        self.ax_radio = plt.axes([0.1, 0.12, 0.8, 0.05])
        self.radio_buttons = RadioButtons(
            self.ax_radio, ['Raw Values', 'Show Density', 'Show FFT'], active=1)
        self.ax_radio.set_frame_on(False)
        self.ax_radio.set_facecolor('none')

        # Validity indicator
        self.indicator_ax = plt.axes([0.8, 0.20, 0.1, 0.03])
        self.indicator_text = self.indicator_ax.text(0.5, 0.5, '', fontsize=10,
                                                     ha='center', va='center',
                                                     transform=self.indicator_ax.transAxes)
        self.indicator_ax.set_xticks([])
        self.indicator_ax.set_yticks([])
        self.indicator_ax.set_frame_on(True)
        self.indicator_ax.set_facecolor('white')

        self.fig.text(0.8, 0.25, 'Constraint ab ≥ 1', fontsize=8)

        # Box-counting button
        self.ax_button = plt.axes([0.1, 0.05, 0.3, 0.05])
        self.button = Button(self.ax_button, 'Display Box-Counting (Fractal) Dimension',
                             color='lightgoldenrodyellow')

        # Dimension display
        ax_dim = plt.axes([0.42, 0.05, 0.18, 0.05])
        self.dim_text = ax_dim.text(0.5, 0.5, 'Fractal Dimension: --', fontsize=10,
                                    ha='center', va='center', transform=ax_dim.transAxes)
        ax_dim.set_xticks([])
        ax_dim.set_yticks([])
        ax_dim.set_frame_on(True)
        ax_dim.set_facecolor('lightblue')

        # Enlarge 1D plots button
        self.ax_enlarge = plt.axes([0.62, 0.05, 0.3, 0.05])
        self.enlarge_button = Button(
            self.ax_enlarge, 'Enlarge 1D Plots', color='lightgreen')
        self.enlarge_button.hovercolor = 'palegreen'

    def _update_parameters(self, a: float, b: float) -> None:
        """Update parameters and check for changes.

        Args:
            a: Amplitude decay parameter
            b: Frequency growth parameter
        """
        ab = a * b
        params_changed = (a != self.last_a) or (b != self.last_b)

        if params_changed:
            self.dimension_calculated = False
            self.last_a = a
            self.last_b = b
            self.button.color = 'lightgoldenrodyellow'
            self.button.hovercolor = 'lightgoldenrodyellow'
            if ab >= 1:
                self.dim_text.set_text('Fractal Dimension: --')
            else:
                self.dim_text.set_text('ab < 1: Not fractal')

        # Update validity indicator
        self.indicator_text.set_text(f"a·b = {ab:.2f}")
        if ab >= 1:
            self.indicator_ax.set_facecolor('lightgreen')
            if not self.dimension_calculated:
                self.button.color = 'lightgoldenrodyellow'
                self.button.hovercolor = 'lightgoldenrodyellow'
        else:
            self.indicator_ax.set_facecolor('lightcoral')
            self.button.color = 'lightgray'
            self.button.hovercolor = 'lightgray'

    def _compute_data(self, a: float, b: float) -> tuple:
        """Compute all necessary data for the plots.

        Args:
            a: Amplitude decay parameter
            b: Frequency growth parameter

        Returns:
            tuple: (Z_norm, W_1d_norm, pos_freqs, pos_fft)
        """
        # Precompute terms
        a_powers = np.array([a ** n for n in range(self.N)], dtype=np.float64)
        b_freqs = np.array(
            [(b ** n) * np.pi for n in range(self.N)], dtype=np.float64)

        # Compute 2D surface
        Z = self.compute_weierstrass_2d_precomputed(
            self.X, self.Y, a_powers, b_freqs)
        Z_norm = Z / np.max(np.abs(Z))
        self.current_Z_norm = Z_norm

        # Compute 1D slice at x=0
        center_idx = self.size // 2
        y_vals = self.y
        W_1d = self.compute_weierstrass_1d(y_vals, a_powers, b_freqs)
        W_1d_norm = W_1d / np.max(np.abs(W_1d))
        self.current_1d_data = W_1d_norm

        # Compute FFT of 1D function
        fft_vals = np.fft.fft(W_1d_norm)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.fftfreq(len(W_1d_norm), d=2/self.size)

        # Only show positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_mag[:len(fft_mag)//2]

        # Note: Angular frequency (rad/sample) = 2π × cycles/sample
        # For 2D FFT we use angular frequency, for 1D FFT we keep cycles/sample

        # Store for enlarge button
        self.current_1d_fft_freqs = pos_freqs
        self.current_1d_fft_mag = pos_fft

        return Z_norm, W_1d_norm, pos_freqs, pos_fft

    def _update_plots(self, Z_norm: np.ndarray, W_1d_norm: np.ndarray,
                      pos_freqs: np.ndarray, pos_fft: np.ndarray,
                      view_mode: str, a: float, b: int) -> None:
        """Update all plots with the computed data.

        Args:
            Z_norm: Normalized 2D Weierstrass function
            W_1d_norm: Normalized 1D Weierstrass function
            pos_freqs: Positive frequencies for FFT
            pos_fft: FFT magnitudes
            view_mode: Current view mode ('Raw Values', 'Show Density', 'Show FFT')
            a: Amplitude decay parameter
            b: Frequency growth parameter
        """
        # Update 1D plot
        self.line_1d.set_data(self.y, W_1d_norm)
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Update FFT plot with stem plot
        if self.fft_stem is not None:
            for artist in self.fft_stem:
                try:
                    artist.remove()
                except:
                    pass

        if len(pos_freqs) > 0:
            self.fft_stem = self.ax2.stem(
                pos_freqs,
                pos_fft,
                linefmt='r-',
                markerfmt='ro',
                basefmt='k-'
            )
            plt.setp(self.fft_stem[0], markersize=3,
                     markerfacecolor='r', markeredgecolor='r')
            plt.setp(self.fft_stem[1], linewidth=0.7)
            plt.setp(self.fft_stem[2], linewidth=0.7)

            if len(pos_freqs) > 0:
                min_freq = np.min(pos_freqs[pos_freqs > 0])
                max_freq = np.max(pos_freqs)
                self.ax2.set_xlim(min_freq * 0.9, max_freq * 1.1)
                self.ax2.set_ylim(
                    np.min(pos_fft[pos_fft > 0]) * 0.9, np.max(pos_fft) * 1.1)

        # Clear previous 2D plot if it exists
        if self.current_plot:
            self.current_plot.remove()
            self.current_plot = None

        self._update_2d_plot(Z_norm, view_mode, a, b)

    def _update_2d_plot(self, Z_norm: np.ndarray, view_mode: str,
                        a: float, b: int) -> None:
        """Update the 2D plot based on the current view mode.

        Args:
            Z_norm: Normalized 2D Weierstrass function
            view_mode: Current view mode ('Raw Values', 'Show Density', 'Show FFT')
            a: Amplitude decay parameter
            b: Frequency growth parameter
        """
        if view_mode == 'Raw Values':
            data = Z_norm
            cmap = 'coolwarm'
            clim = (-1, 1)
            label = 'Normalized Value'
            self.current_plot = self.ax0.imshow(
                data, cmap=cmap, extent=(-1, 1, -1, 1))
            self.ax0.set_xlim(-1, 1)
            self.ax0.set_ylim(-1, 1)
            self.ax0.set_xlabel('X Coordinate')
            self.ax0.set_ylabel('Y Coordinate')
            self.ax0.set_aspect('equal')
            current_title = f'Normalized 2D Weierstrass Function (a={a:.2f}, b={int(b)})'
        elif view_mode == 'Show Density':
            data = self.compute_density_approx(
                Z_norm.flatten(), self.bins).reshape(Z_norm.shape)
            cmap = 'inferno'
            clim = (0, np.max(data) + 1e-9 if np.max(data)
                    == 0 else np.max(data))
            label = 'Probability Density'
            self.current_plot = self.ax0.pcolormesh(
                self.x_edges, self.y_edges, data, cmap=cmap, shading='auto')
            self.ax0.set_xlim(-1, 1)
            self.ax0.set_ylim(-1, 1)
            self.ax0.set_xlabel('X Coordinate')
            self.ax0.set_ylabel('Y Coordinate')
            self.ax0.set_aspect('equal')
            current_title = f'Value Probability Density (a={a:.2f}, b={int(b)})'
        else:  # Show FFT
            data = self.compute_fft(Z_norm)
            cmap = 'inferno'
            clim = (np.min(data), np.max(data))
            label = 'Log-Magnitude (dB)'
            self.current_plot = self.ax0.imshow(
                data, cmap='viridis', extent=self.extent_freq)
            self.ax0.set_xlim(self.extent_freq[0], self.extent_freq[1])
            self.ax0.set_ylim(self.extent_freq[2], self.extent_freq[3])
            self.ax0.set_xlabel('Angular Frequency ω_x (rad/sample)')
            self.ax0.set_ylabel('Angular Frequency ω_y (rad/sample)')
            self.ax0.set_aspect('equal')
            current_title = f'Frequency Spectrum (a={a:.2f}, b={int(b)})'

        self.current_plot.set_clim(*clim)
        self.cbar.update_normal(self.current_plot)
        self.cbar.set_label(label)
        self.title_2d.set_text(current_title)
        self.ax1.set_title(f'1D Weierstrass (x=0, a={a:.2f}, b={int(b)})')

    def update_plot(self, val) -> None:
        """Main plot update function called when parameters change.

        Args:
            val: Not used (required by matplotlib callback)
        """
        a = self.slider_a.val
        b = self.slider_b.val
        view_mode = self.radio_buttons.value_selected

        self._update_parameters(a, b)
        Z_norm, W_1d_norm, pos_freqs, pos_fft = self._compute_data(a, b)
        self._update_plots(Z_norm, W_1d_norm, pos_freqs,
                           pos_fft, view_mode, a, b)
        self.fig.canvas.draw_idle()

    def calculate_dimension(self, event):
        if self.current_Z_norm is None or self.dimension_calculated:
            return

        a = self.slider_a.val
        b = self.slider_b.val
        ab = a * b

        if ab < 1:
            self.dim_text.set_text('ab < 1: Not fractal')
            return

        self.radio_buttons.set_active(0)
        epsilons = np.linspace(0.02, 0.2, 10)
        self.current_dimension = self.box_counting_dimension(
            self.current_Z_norm, epsilons)
        self.dim_text.set_text(
            f'Fractal Dimension: {self.current_dimension:.3f}')
        self.dimension_calculated = True
        self.button.color = 'lightgray'
        self.button.hovercolor = 'lightgray'

    def enlarge_1d_plots(self, event):
        if self.current_1d_data is None or self.current_1d_fft_freqs is None or self.current_1d_fft_mag is None:
            return

        fig1d, (ax1d, axfft) = plt.subplots(2, 1, figsize=(12, 10))

        a = self.slider_a.val
        b = self.slider_b.val

        ax1d.plot(self.y, self.current_1d_data, 'b-', linewidth=1.5)
        ax1d.set_title(
            f'1D Weierstrass Function (x=0, a={a:.2f}, b={int(b)})', fontsize=14)
        ax1d.set_xlabel('y', fontsize=12)
        ax1d.set_ylabel('W(y)', fontsize=12)
        ax1d.grid(True)
        ax1d.tick_params(axis='both', which='major', labelsize=10)

        markerline, stemlines, baseline = axfft.stem(
            self.current_1d_fft_freqs,
            self.current_1d_fft_mag,
            linefmt='r-',
            markerfmt='ro',
            basefmt='k-'
        )

        plt.setp(markerline, markersize=4,
                 markerfacecolor='r', markeredgecolor='r')
        plt.setp(stemlines, linewidth=1.0)
        plt.setp(baseline, linewidth=1.0)

        axfft.set_xscale('log')
        axfft.set_yscale('log')
        axfft.set_title(
            'FFT of 1D Weierstrass Function (Stem Plot)', fontsize=14)
        axfft.set_xlabel('Frequency (cycles/sample)', fontsize=12)
        axfft.set_ylabel('Magnitude (log scale)', fontsize=12)
        axfft.grid(True, which='both', linestyle='--', alpha=0.7)
        axfft.tick_params(axis='both', which='major', labelsize=10)

        if len(self.current_1d_fft_freqs) > 0:
            min_freq = np.min(
                self.current_1d_fft_freqs[self.current_1d_fft_freqs > 0])
            max_freq = np.max(self.current_1d_fft_freqs)
            axfft.set_xlim(min_freq * 0.9, max_freq * 1.1)

            min_mag = np.min(
                self.current_1d_fft_mag[self.current_1d_fft_mag > 0])
            max_mag = np.max(self.current_1d_fft_mag)
            axfft.set_ylim(min_mag * 0.9, max_mag * 1.1)

        fig1d.suptitle(
            f'Weierstrass Function Analysis (a={a:.2f}, b={int(b)})', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()


# Create and show the visualization
if __name__ == "__main__":
    visualizer = WeierstrassVisualizer()
    plt.show()
