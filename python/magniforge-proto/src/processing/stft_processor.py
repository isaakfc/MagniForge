import numpy as np
import scipy.signal
import librosa
import soundfile as sf


class SpectralProcessor:
    def __init__(self, fft_size=1024, window_size=1024, hop_length=512, sample_rate=44100):
        self.fft_size = fft_size
        self.window_size = window_size
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.window = scipy.signal.windows.hann(window_size, sym=False)

    def stft(self, signal):
        """
        Performs Short-Time Fourier Transform (STFT) on the input signal.
        Simplified version without log conversion and unnecessary operations.
        """
        # Ensure signal is 1D
        signal = np.asarray(signal)

        # Add half-window padding at the beginning
        half_window = self.window_size // 2
        signal = np.pad(signal, (half_window, 0), mode='constant')

        # Calculate number of frames
        n_samples = len(signal)
        n_windows = int(np.ceil((n_samples - self.window_size) / self.hop_length)) + 1

        # Pad signal if needed
        total_samples = (n_windows - 1) * self.hop_length + self.window_size
        if total_samples > n_samples:
            signal = np.pad(signal, (0, total_samples - n_samples), mode='constant')

        # Initialize STFT matrix
        stft_matrix = np.empty((self.fft_size // 2 + 1, n_windows), dtype=np.complex64)

        # Compute STFT
        for i in range(n_windows):
            start = i * self.hop_length
            end = start + self.window_size

            # Window the signal and compute FFT
            frame = signal[start:end] * self.window
            stft_matrix[:, i] = scipy.fft.rfft(frame, n=self.fft_size)

        return stft_matrix

    def istft(self, stft_matrix):
        """
        Performs Inverse Short-Time Fourier Transform (ISTFT) with simplified normalization
        for 50% overlap condition.
        """
        n_frames = stft_matrix.shape[1]
        expected_signal_len = self.fft_size + self.hop_length * (n_frames - 1)

        # Initialize output signal
        y = np.zeros(expected_signal_len)

        # Perform ISTFT
        for i in range(n_frames):
            # Inverse FFT
            frame = np.fft.irfft(stft_matrix[:, i], n=self.fft_size)

            # Apply window
            frame = frame * self.window

            # Overlap-add
            start = i * self.hop_length
            end = start + self.fft_size
            y[start:end] += frame

        win_sum = np.zeros(y.shape)
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.fft_size
            win_sum[start:end] += self.window ** 2

        # Avoid division by zero
        threshold = 1e-10
        # Replace values below 1e-10 in win_sum with 1.0
        win_sum[win_sum < threshold] = 1.0

        # Normalise the signal by dividing by the accumulated squared window function
        y /= win_sum

        # Remove padding introduced by centering within the STFT (Half a window length at beginning)
        y = y[self.fft_size // 2:]

        return y


class COLASpectralProcessor:
    def __init__(self, frame_size, fft_size=None, sample_rate=44100):
        """
        Class which can perform STFT and ISTFT using double windowing with square root Hanning windows to avoid having
        to perform any normalisation. Uses fixed 50% overlap, determined from the frame size to achieve perfect COLA
        conditions. The FFT size can be chosen to be larger than the frame size for frequency bin interpolation.

        Args:
            frame_size: Frame size of samples taken from the signal
            fft_size: Size of FFT for frequency interpolation. If None, defaults to frame_size
            sample_rate: Samples per second
        """
        self.frame_size = frame_size
        # Fix hop at 50% overlap for COLA conditions
        self.hop_size = frame_size // 2
        self.sample_rate = sample_rate

        # Ensure FFT size is not smaller than the frame size
        if fft_size is None:
            self.fft_size = frame_size
        elif fft_size < frame_size:
            raise ValueError("FFT size must be greater than or equal to frame size")
        else:
            self.fft_size = fft_size

        # Create square root Hanning windows for analysis and synthesis
        full_window = np.hanning(self.frame_size)
        self.analysis_window = np.sqrt(full_window)
        self.synthesis_window = np.sqrt(full_window)

    def stft(self, signal):
        """
        Calculate the Short-Time Fourier Transform of a signal.

        Args:
            signal: Input signal
        Returns:
            Complex STFT matrix
        """
        # Determine number of frames (no padding used)
        num_frames = (len(signal) - self.frame_size) // self.hop_size + 1

        # Pre-allocate numpy array for STFT output
        X = np.zeros((num_frames, self.fft_size), dtype=complex)

        # Process frame by frame
        for frame_idx in range(num_frames):
            # Get start index
            start_idx = frame_idx * self.hop_size
            # Extract frame and apply analysis window
            frame = signal[start_idx:start_idx + self.frame_size]
            windowed_frame = self.analysis_window * frame

            # Zero-pad if FFT size is bigger than frame size
            if self.fft_size > self.frame_size:
                padded_frame = np.zeros(self.fft_size)
                padded_frame[:self.frame_size] = windowed_frame
                X[frame_idx] = np.fft.fft(padded_frame)
            else:
                X[frame_idx] = np.fft.fft(windowed_frame)

        return X

    def istft(self, X, signal_length):
        """
        Calculate the Inverse Short-Time Fourier Transform.

        Args:
            X: Complex STFT matrix (each row contains a bin)
            signal_length: Number of samples in the original signal

        Returns:
            Reconstructed time-domain signal
        """
        # Initialize output array
        x = np.zeros(signal_length)

        # Process frame by frame
        for frame_idx in range(len(X)):
            # Get start index
            start_idx = frame_idx * self.hop_size
            # Compute inverse FFT
            ifft_frame = np.real(np.fft.ifft(X[frame_idx]))
            # Apply synthesis window and overlap-add
            x[start_idx:start_idx + self.frame_size] += (
                self.synthesis_window * ifft_frame[:self.frame_size]
            )

        return x

