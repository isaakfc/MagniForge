import numpy as np
from typing import Optional, Tuple


class TransientDetector:
    def __init__(self,
                 initial_normalization: float = 2000.0,
                 forgetting_factor: float = 0.99,
                 graded_response_slope: float = 1.7,
                 graded_response_threshold: float = 0.5,
                 graded_response_range: float = 1.5,
                 ma_window_size: int = 15,
                 mf_window_size: int = 10,
                 alpha_smoothing: float = 0.1):
        """
        Default parameters are taken from the Graded and Amplify examples from paper: Frequency-Domain Algorithms for
        Audio Signal Enhancement Based on Transient Modification

        Args:
            initial_normalization: Initial β₀ for flux normalisation
            forgetting_factor: Decay constant γ for the peak following (Should be set close to 1)
            graded_response_slope: λ Slope parameter for the graded response (Higher values make the transient detection
            score more binary)
            graded_response_threshold: Φ₀ Phi sets which input value yields the midpoint of the possible output values
            graded_response_range: α₀ Alpha_0 sets minimum value of alpha w
            ma_window_size: Size of the moving average filter buffer
            mf_window_size: Size of the median filter buffer
            alpha_smoothing: Amount of smoothing for exponential filter (lower values mean more smoothing)
        """
        self.beta = initial_normalization
        self.gamma = forgetting_factor
        self.lambd = graded_response_slope
        self.phi_0 = graded_response_threshold
        self.alpha_0 = graded_response_range

        # Internal state
        self.prev_magnitudes: Optional[np.ndarray] = None

        # Initialize smoothing buffers and state
        self.ma_buffer = np.zeros(ma_window_size)
        self.mf_buffer = np.zeros(mf_window_size)
        self.alpha_smooth = alpha_smoothing
        self.prev_alpha = 0.0

    def calculate_spectral_flux(self, magnitudes: np.ndarray) -> float:
        """
        Calculates the unnornmalised spectral flux between two consecutive frames. Uses the formula
        ρ[n] = Σ|Δ[n, k]|^(1/2) from the paper where ρ[n] is the unnormalised spectral flux

        Args:
            magnitudes: Magnitude spectrum of the current frame
        Returns:
            Unnormalised spectral flux
        """

        if self.prev_magnitudes is None:
            self.prev_magnitudes = magnitudes
            return 0.0

        # Compute first order difference of consecutive magnitude arrays
        delta = magnitudes - self.prev_magnitudes
        self.prev_magnitudes = magnitudes

        # Compute spectral flux as formulated in the paper
        flux = np.sum(np.abs(delta) ** 0.5)

        return flux

    def calculate_normalisation(self, flux: float) -> float:
        """
        Calculates normalisation factor β using the peak follower formulation:
        βₙ = ρ[n]     if ρ[n] > βₙ₋₁
             γβₙ₋₁    otherwise

        Args:
            flux: Current unnormalised spectral flux calculation ρ[n]
        Returns:
            Spectral flux normalisation factor β
        """

        if flux > self.beta:
            self.beta = flux
        else:
            self.beta = self.beta * self.gamma

        return self.beta

    def calculate_graded_response(self, phi: float) -> float:
        """
        Calculates the graded response of the normalised spectral flux phi Φ[n] via the hyperbolic tangent function:
        α[n] = ((α₀ + 1)/2) + ((α₀ - 1)/2) * tanh[πλ(Φ[n] - Φ₀)]

        Args:
            phi: The normalised spectral flux Φ[n]
        Returns:
            The calculated graded response value α[n]
        """

        alpha_plus = (self.alpha_0 + 1) / 2
        alpha_minus = (self.alpha_0 - 1) / 2

        tanh_component = np.tanh((np.pi * self.lambd) * (phi - self.phi_0))

        alpha = alpha_plus + (alpha_minus * tanh_component)

        return alpha

    def smooth_normalized_flux(self, flux: float) -> float:
        """
        Smooths normalised flux values by using a moving average filter
        """
        self.ma_buffer = np.roll(self.ma_buffer, 1)
        self.ma_buffer[0] = flux
        return np.mean(self.ma_buffer)

    def smooth_graded_response(self, alpha: float) -> float:
        """
        Uses exponential smoothgin via the formula: y[n] = αx[n] + (1-α)y[n-1]
        """
        smoothed = (self.alpha_smooth * alpha +
                   (1 - self.alpha_smooth) * self.prev_alpha)
        self.prev_alpha = smoothed
        return smoothed

    def calculate_adaptive_threshold_no_lookahead(self, detection_function: float) -> float:
        """
        Uses a median filter with a buffer of previous values for smoothing:
        """

        self.mf_buffer = np.roll(self.mf_buffer, -1)
        self.mf_buffer[-1] = detection_function

        # Can edit this scaling factor to vary results
        scaling_factor = 1
        threshold = scaling_factor * np.median(self.mf_buffer)
        return threshold

    def frame_transient_calculation(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Process a single FFT frame and calculate its transient value

        Args:
            frame: Complex STFT frame

        Returns:
            transient_value: The calculated transient value (graded response value α[n])
            normalised_flux: The normalised flux Φ[n]
        """

        # Obtain positive frequency magnitudes
        magnitudes = np.abs(frame[:len(frame) // 2])

        # Calculate flux for current frame ρ[n]
        flux = self.calculate_spectral_flux(magnitudes)
        # Calulate normalisation factor for the flux β[n]
        beta = self.calculate_normalisation(flux)
        # Compute normalised flux Φ[n]
        normalized_flux = flux / beta if beta > 0 else 0

        smoothed_flux = self.smooth_normalized_flux(normalized_flux)
        # smoothed_flux = self.calculate_adaptive_threshold_no_lookahead(normalized_flux)
        # Compute graded response α[n]
        transient_value = self.calculate_graded_response(smoothed_flux)

        transient_value = self.calculate_adaptive_threshold_no_lookahead(transient_value)
        # transient_value = self.smooth_graded_response(transient_value)

        return transient_value, smoothed_flux





if __name__ == "__main__":
    transient_analyser = TransientDetector(initial_normalization=100, graded_response_slope=1.7)

    print(transient_analyser.calculate_graded_response(0.5))






