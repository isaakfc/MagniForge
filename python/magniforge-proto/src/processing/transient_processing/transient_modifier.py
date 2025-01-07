import numpy as np


class TransientModifier:
    def __init__(self, use_nonlinear: bool = True):
        """
        Modifies the magnitude of the transient bins with options for linear or non-linear modification

        Args:
            use_nonlinear: If true, the magnitude modification is changed from linear to non-linear
        """
        self.use_nonlinear = use_nonlinear

    def process_frame(self, spectrum: np.ndarray, transient_value: float) -> np.ndarray:
        """
        Modifies the magnitudes of a single FFT frame based on the transient value

        Args:
            spectrum: Complex STFT frame
            transient_value: α[n] from transient detector (graded response)

        Returns:
            Modified complex spectrum
        """
        magnitudes = np.abs(spectrum)
        phases = np.angle(spectrum)

        if self.use_nonlinear:
            # Non-linear modification (Eq 10 from paper)
            modified_magnitudes = (magnitudes + 1) ** transient_value - 1
        else:
            # Linear modification (Eq 8 from paper)
            modified_magnitudes = magnitudes * transient_value

        # Multiply phase back in
        return modified_magnitudes * np.exp(1j * phases)