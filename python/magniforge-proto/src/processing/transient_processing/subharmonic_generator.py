import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.signal.windows import gaussian
from collections import deque


class SubharmonicGenerator:
    def __init__(self,
                 sample_rate: int = 44100,
                 fft_size: int = 2048,
                 peak_threshold: float = 0.1,
                 subharmonic_octave: int = 1,
                 min_subharmonic_freq: float = 50.0,
                 ma_window_size: int = 20,
                 subharmonic_gain: float = 0.5,
                 bell_width: int = 5,
                 hysteresis_threshold_factor: float = 0.05):
        """
        Spectral subharmonic generator with peak tracking and smooth subharmonic transitions

        Args:
            sample_rate: Signal sample rate
            fft_size: Number of bins in FFT frame
            peak_threshold: Normalised threshold for lowest peak identification (0-1)
            subharmonic_octave: Number of octaves below lowest peak that the subharmonic is produced
            min_subharmonic_freq: The lowest frequency subharmonic that can be produced
        """

        if not 1 <= subharmonic_octave <= 3:
            raise ValueError("Cannot have subharmonic octave under 3 octaves (must be between 1 and 3")

        if not 0 < hysteresis_threshold_factor <= 1:
            raise ValueError("hysteresis_threshold_factor must be between 0 and 1")

        # Ensure bell width is odd, so we have a centre value of the window
        self.bell_width = bell_width + (1 - bell_width % 2)
        # Create Gaussian window for the bell-curve
        self.bell_window = gaussian(self.bell_width, std=self.bell_width/6)

        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.peak_threshold = peak_threshold
        self.subharmonic_octave = subharmonic_octave
        self.min_subharmonic_freq = min_subharmonic_freq
        self.min_peak_bin = None
        self.subharmonic_gain = subharmonic_gain

        # Initialise deques for moving average buffers
        self.peak_buffer = deque(maxlen=ma_window_size)
        self.energy_buffer = deque([0] * ma_window_size, maxlen=ma_window_size)

        # Hysteresis member variables
        self.previous_peak_bin = None
        self.hysteresis_threshold_factor = hysteresis_threshold_factor
        self.hysteresis_range = 1

    def detect_lowest_peak(self, spectrum: NDArray):
            """
            Looks for the lowest peak in normalised STFT frame over a certain threshold and above min_subharmonic_freq

            Args:
                spectrum: Complex FFT spectrum
            Returns:
                Bin index of lowest peak over a threshold or 0 if none found
            """
            # Calculate minimum peak frequency and bin dynamically
            min_peak_freq = self.min_subharmonic_freq * (2 ** self.subharmonic_octave)
            self.min_peak_bin = int(min_peak_freq * self.fft_size / self.sample_rate)

            # Get positive frequency bins only
            positive_spectrum = spectrum[:self.fft_size // 2]

            # Get FFT bin magnitudes and normalise
            magnitude_spectrum = np.abs(positive_spectrum)
            spectrum_norm = magnitude_spectrum / np.max(magnitude_spectrum)

            # First find strong peaks using our normal threshold
            peaks, _ = find_peaks(spectrum_norm, height=self.peak_threshold)

            # Hysteresis processing
            if self.previous_peak_bin is not None:
                # Define our Hysteresis region of interest around the previous peak
                search_start = max(self.min_peak_bin, self.previous_peak_bin - self.hysteresis_range)
                search_end = min(len(spectrum_norm), self.previous_peak_bin + self.hysteresis_range)

                # Search for peaks in this region with a lower threshold
                hysteresis_threshold = self.peak_threshold * self.hysteresis_threshold_factor  # More lenient threshold
                nearby_peaks, _ = find_peaks(spectrum_norm[search_start:search_end],
                                             height=hysteresis_threshold)

                # If there are any peaks in the region of interest
                if len(nearby_peaks) > 0:
                    # Convert local indices to global spectrum indices
                    nearby_peaks += search_start

                    # Pick strongest nearby peak
                    peak_idx = nearby_peaks[np.argmax(spectrum_norm[nearby_peaks])]
                    self.previous_peak_bin = peak_idx
                    return peak_idx, spectrum_norm[peak_idx]

            # If we did not find any peaks with the hysterisis, do normal peak detection
            valid_peaks = peaks[peaks >= self.min_peak_bin]
            if len(valid_peaks) > 0:
                peak_idx = valid_peaks[0]
                self.previous_peak_bin = peak_idx
                return peak_idx, spectrum_norm[peak_idx]

            return 0, 0.0  # If no peaks are found

    def get_smoothed_peaks(self, spectrum: NDArray):
        """
        Calculates smoothed peak frequency and energy using moving average filters.

        Args:
            spectrum: Complex FFT spectrum
        Returns:
            tuple of smoothed peak bin and smoothed energy
        """
        peak_bin, energy = self.detect_lowest_peak(spectrum)

        # Append to back of buffers (will automatically pop the left most value (FIFO))
        if peak_bin != 0:
            self.peak_buffer.append(peak_bin)
        self.energy_buffer.append(energy)

        # Calculate moving averages
        smoothed_peak = sum(self.peak_buffer) / len(self.peak_buffer)
        smoothed_energy = sum(self.energy_buffer) / len(self.energy_buffer)

        # Round to nearest integer bin
        rounded_peak_bin = int(round(smoothed_peak))

        return rounded_peak_bin, smoothed_energy

    def apply_bell_curve(self, spectrum: NDArray, centre_bin: int, magnitude: float) -> NDArray:
        """
        Applies Gaussian based bell-curved modification around centre frequency bin to allow a more natural and
        artefact free subharmonic

        Args:
            spectrum: FFT frame to modify
            centre_bin: The central bin to modify
            magnitude: The magnitude to apply at the centre
        Returns:
            Modified spectrum
        """

        # Calculate which bins are to be modified
        half_width = self.bell_width // 2
        start_bin = max(0, centre_bin - half_width)
        end_bin = min(self.fft_size//2, centre_bin + half_width + 1)

        # Calculate how much of the window can actually be used
        window_start = max(0, half_width - centre_bin)

        modified_spectrum = spectrum.copy()

        # Lop through bins to be modified
        for i, bin_idx in enumerate(range(start_bin, end_bin)):
            window_idx = window_start + i
            # Calculate magnitude to be added on from bell curve
            additional_magnitude = magnitude * self.bell_window[window_idx]

            # Get complex value of current bin
            current_bin = spectrum[bin_idx]
            current_magnitude = np.abs(current_bin)
            current_phase = np.angle(current_bin)

            # Add magnitudes together while preserving phase
            new_magnitude = current_magnitude + additional_magnitude
            modified_spectrum[bin_idx] = new_magnitude * np.exp(1j * current_phase)

            # Mirror spectrum for negative frequencies
            if bin_idx > 0:
                modified_spectrum[-bin_idx] = np.conj(modified_spectrum[bin_idx])

        return modified_spectrum

    def generate_subharmonics(self, spectrum: NDArray) -> NDArray:
        """
        Generates subharmonics within the FFT frame, with smooth transitions between peaks
        Args:
            spectrum: Complex FFT spectrum
        Returns:
            Modified FFT frame with added subharmonics
        """
        # Get smoothed peak and energy
        peak_bin, smoothed_energy = self.get_smoothed_peaks(spectrum)

        # Calculate subharmonic bin by dividing peak bin by 2^octave
        subharmonic_bin = peak_bin // (2 ** self.subharmonic_octave)

        # Get magnitude of peak
        peak_magnitude = np.abs(spectrum[peak_bin])

        # Calculate subharmonic magnitude to be applied based on original peak, energy and user gain
        subharmonic_magnitude = peak_magnitude * self.subharmonic_gain * (smoothed_energy / np.max(np.abs(spectrum)))

        return self.apply_bell_curve(spectrum, subharmonic_bin, subharmonic_magnitude)












