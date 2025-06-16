import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from numba import njit
import matplotlib.gridspec as gridspec


class WeierstrassVisualizer:
    """Interactive visualization of the Weierstrass function with various views and analysis tools."""

    def __init__(self, size: int = 500, N: int = 40, bins: int = 500) -> None:
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

        # Initial plot
        self.update_plot(None)

    @staticmethod
    @njit
    def compute_weierstrass_1d(y, a_powers, b_freqs):
        """Compute the 1D Weierstrass function along the y-axis."""
        total = np.zeros_like(y)
        for n in range(len(a_powers)):
            total += a_powers[n] * np.cos(b_freqs[n] * y)
        return total

    @staticmethod
    @njit
    def compute_weierstrass_2d_precomputed(X, Y, a_powers, b_freqs):
        """Compute the 2D Weierstrass function."""
        W = np.zeros_like(X)
        for n in range(len(a_powers)):
            W += a_powers[n] * np.cos(b_freqs[n] * X) * np.cos(b_freqs[n] * Y)
        return W

    @staticmethod
    def compute_density_approx(values, bins=500):
        """Compute density approximation for the density view."""
        hist, bin_edges = np.histogram(values, bins=bins, density=True)
        bin_indices = np.searchsorted(bin_edges, values, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
        return hist[bin_indices]

    @staticmethod
    def compute_fft(Z):
        """Compute 2D FFT magnitude in logarithmic scale (dB)."""
        fft_Z = np.fft.fft2(Z)
        fft_shifted = np.fft.fftshift(fft_Z)
        magnitude = np.abs(fft_shifted)
        # CORRECTED: Use proper dB calculation (20*log10)
        return 20 * np.log10(magnitude + 1e-10)

    @staticmethod
    @njit
    def box_counting_dimension(Z, epsilons):
        """Calculate box-counting dimension using multiple scales (optimized)."""
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

            # Create a flat set for occupied boxes
            flat_indices = set()

            for x in range(size):
                for y in range(size):
                    # Spatial box indices
                    bx = x // box_size_xy
                    by = y // box_size_xy

                    # Value box index
                    z_val = Z_norm[x, y]
                    bz = int(z_val / box_size_z)
                    bz = min(bz, grid_z - 1)

                    # Create unique box identifier
                    box_id = bx + by * grid_x + bz * (grid_x * grid_y)
                    flat_indices.add(box_id)

            counts[i] = len(flat_indices)

        # Linear regression in log-log space
        log_eps = np.log(epsilons)
        log_counts = np.log(counts)
        n = len(epsilons)
        sum_x = np.sum(log_eps)
        sum_y = np.sum(log_counts)
        sum_xy = np.sum(log_eps * log_counts)
        sum_x2 = np.sum(log_eps**2)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        return -slope

    # Dummy call to ensure the function is pre-compiled by the JIT compiler before it's called by the interpreter.
    dummy_Z = np.random.rand(4, 4)
    dummy_eps = np.array([0.1, 0.2, 0.3])
    _ = box_counting_dimension(dummy_Z, dummy_eps)

    def _setup_coordinates(self) -> None:
        """Initialize coordinate grids and FFT frequency coordinates."""
        # Spatial coordinates
        self.x = np.linspace(-1, 1, self.size)
        self.y = np.linspace(-1, 1, self.size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # For density view
        self.x_edges = np.linspace(-1, 1, self.size + 1)
        self.y_edges = np.linspace(-1, 1, self.size + 1)

        # FFT frequency coordinates (2D)
        dx = 2.0 / (self.size - 1)  # Spatial step
        freq_cycles_x = np.fft.fftshift(np.fft.fftfreq(self.size, d=dx))
        freq_cycles_y = np.fft.fftshift(np.fft.fftfreq(self.size, d=dx))

        # Convert to angular frequency (rad/normalized unit)
        self.freq_x = freq_cycles_x * 2 * np.pi
        self.freq_y = freq_cycles_y * 2 * np.pi
        self.extent_freq = [self.freq_x[0], self.freq_x[-1],
                            self.freq_y[0], self.freq_y[-1]]

    def _setup_gui(self) -> None:
        """Set up the graphical user interface components."""
        # Create figure with improved layout
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.4, 0.6], height_ratios=[1, 1],
                               top=0.95, bottom=0.35, hspace=0.3, wspace=0.3)

        # Set up plot axes
        self.ax0 = plt.subplot(gs[:, 0])
        self.im = self.ax0.imshow(
            np.zeros((self.size, self.size)), extent=(-1, 1, -1, 1))
        self.cbar = plt.colorbar(self.im, ax=self.ax0)
        self.title_2d = self.ax0.set_title('')

        # 1D Weierstrass plot
        self.ax1 = plt.subplot(gs[0, 1])
        self.line_1d, = self.ax1.plot(self.y, np.zeros_like(self.y), 'b-')
        self.ax1.set_title('1D Weierstrass Function (x=0)')
        self.ax1.set_xlabel('Position along y-axis (normalized unit)')
        self.ax1.set_ylabel('W(y)')
        self.ax1.grid(True)

        # FFT plot (1D) - Corrected to angular frequency
        self.ax2 = plt.subplot(gs[1, 1])
        self.ax2.set_title('FFT of 1D Weierstrass Function')
        self.ax2.set_xlabel(
            'Angular Frequency (rad/normalized unit)')  # Correct unit
        self.ax2.set_ylabel('Magnitude (log scale)')
        self.ax2.set_xscale('log')
        self.ax2.set_yscale('log')
        self.ax2.grid(True)

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
        """Update parameters and check for changes."""
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
        """Compute all necessary data for the plots."""
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

        # Compute FFT of 1D function - CORRECTED TO ANGULAR FREQUENCY
        d_spatial = 2.0 / (self.size - 1)  # Spatial step
        fft_vals = np.fft.fft(W_1d_norm)
        fft_mag = np.abs(fft_vals)

        # Get cyclic frequencies and convert to angular
        freqs_cycles = np.fft.fftfreq(
            len(W_1d_norm), d=d_spatial)  # cycles/unit
        freqs_angular = freqs_cycles * 2 * np.pi  # rad/unit

        # CORRECTED: Use mask for positive frequencies
        pos_mask = freqs_angular > 0
        pos_angular_freqs = freqs_angular[pos_mask]
        pos_fft = fft_mag[pos_mask]

        # Store for enlarge button
        self.current_1d_fft_freqs = pos_angular_freqs
        self.current_1d_fft_mag = pos_fft

        return Z_norm, W_1d_norm, pos_angular_freqs, pos_fft

    def _update_plots(self, Z_norm: np.ndarray, W_1d_norm: np.ndarray,
                      pos_angular_freqs: np.ndarray, pos_fft: np.ndarray,
                      view_mode: str, a: float, b: int) -> None:
        """Update all plots with the computed data."""
        # Update 1D plot
        self.line_1d.set_data(self.y, W_1d_norm)
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Update FFT plot with stem plot (using angular frequencies)
        if self.fft_stem is not None:
            for artist in self.fft_stem:
                try:
                    artist.remove()
                except:
                    pass

        if len(pos_angular_freqs) > 0:
            self.fft_stem = self.ax2.stem(
                pos_angular_freqs,
                pos_fft,
                linefmt='r-',
                markerfmt='ro',
                basefmt='k-'
            )
            plt.setp(self.fft_stem[0], markersize=3,
                     markerfacecolor='r', markeredgecolor='r')
            plt.setp(self.fft_stem[1], linewidth=0.7)
            plt.setp(self.fft_stem[2], linewidth=0.7)

            if len(pos_angular_freqs) > 0:
                min_freq = np.min(pos_angular_freqs[pos_angular_freqs > 0])
                max_freq = np.max(pos_angular_freqs)
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
        """Update the 2D plot based on the current view mode."""
        if view_mode == 'Raw Values':
            data = Z_norm
            cmap = 'coolwarm'
            clim = (-1, 1)
            # CORRECTED: Remove redundant "normalized"
            label = 'Value'
            self.current_plot = self.ax0.imshow(
                data, cmap=cmap, extent=(-1, 1, -1, 1))
            self.ax0.set_xlim(-1, 1)
            self.ax0.set_ylim(-1, 1)
            self.ax0.set_xlabel('X Coordinate (normalized unit)')
            self.ax0.set_ylabel('Y Coordinate (normalized unit)')
            self.ax0.set_aspect('equal')
            current_title = f'2D Weierstrass Function (a={a:.2f}, b={int(b)})'
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
            self.ax0.set_xlabel('X Coordinate (normalized unit)')
            self.ax0.set_ylabel('Y Coordinate (normalized unit)')
            self.ax0.set_aspect('equal')
            current_title = f'Value Probability Density (a={a:.2f}, b={int(b)})'
        else:  # Show FFT
            data = self.compute_fft(Z_norm)
            cmap = 'inferno'
            clim = (np.min(data), np.max(data))
            # CORRECTED: Standard dB labeling
            label = 'Magnitude (dB)'
            self.current_plot = self.ax0.imshow(
                data, cmap='viridis', extent=self.extent_freq)
            self.ax0.set_xlim(self.extent_freq[0], self.extent_freq[1])
            self.ax0.set_ylim(self.extent_freq[2], self.extent_freq[3])
            self.ax0.set_xlabel('Angular Frequency ω_x (rad/normalized unit)')
            self.ax0.set_ylabel('Angular Frequency ω_y (rad/normalized unit)')
            self.ax0.set_aspect('equal')
            current_title = f'2D Frequency Spectrum (a={a:.2f}, b={int(b)})'

        self.current_plot.set_clim(*clim)
        self.cbar.update_normal(self.current_plot)
        self.cbar.set_label(label)
        self.title_2d.set_text(current_title)
        self.ax1.set_title(f'1D Weierstrass (x=0, a={a:.2f}, b={int(b)})')

    def update_plot(self, val) -> None:
        """Main plot update function called when parameters change."""
        a = self.slider_a.val
        b = self.slider_b.val
        view_mode = self.radio_buttons.value_selected

        self._update_parameters(a, b)
        Z_norm, W_1d_norm, pos_angular_freqs, pos_fft = self._compute_data(
            a, b)
        self._update_plots(Z_norm, W_1d_norm, pos_angular_freqs,
                           pos_fft, view_mode, a, b)
        self.fig.canvas.draw_idle()

    def calculate_dimension(self, event):
        """Calculate and display the fractal dimension."""
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

        # CORRECTED theoretical dimension for 2D surface
        theoretical_dim = 3 + np.log(a)/np.log(b)

        # Ensure dimension is physically plausible (2 ≤ D ≤ 3)
        theoretical_dim = max(2, min(3, theoretical_dim))

        self.dim_text.set_text(
            f'Fractal Dim: {self.current_dimension:.3f} (Theory: {theoretical_dim:.3f})')

        self.dimension_calculated = True
        self.button.color = 'lightgray'
        self.button.hovercolor = 'lightgray'

    def enlarge_1d_plots(self, event):
        """Create enlarged view of 1D plots in a separate figure."""
        if (self.current_1d_data is None or
            self.current_1d_fft_freqs is None or
                self.current_1d_fft_mag is None):
            return

        fig1d, (ax1d, axfft) = plt.subplots(2, 1, figsize=(12, 10))

        a = self.slider_a.val
        b = self.slider_b.val

        ax1d.plot(self.y, self.current_1d_data, 'b-', linewidth=1.5)
        ax1d.set_title(
            f'1D Weierstrass Function (x=0, a={a:.2f}, b={int(b)})', fontsize=14)
        ax1d.set_xlabel(
            'Position along y-axis (normalized unit))', fontsize=12)
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
            'FFT of 1D Weierstrass Function', fontsize=14)
        axfft.set_xlabel(
            'Angular Frequency (rad/normalized unit)', fontsize=12)
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
        plt.close(fig1d)  # Memory management


# Create and show the visualization
if __name__ == "__main__":

    new_size = 500
    new_N = 40
    visualizer = WeierstrassVisualizer(size=new_size, N=new_N, bins=new_size)
    plt.show()
