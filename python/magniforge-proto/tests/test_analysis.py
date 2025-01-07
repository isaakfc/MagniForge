import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from src.processing import COLASpectralProcessor
from src.analysis import TransientDetector


if __name__ == "__main__":
    # Initialize processor with your parameters
    FFT_SIZE = 2048
    WINDOW_SIZE = 1024
    HOP_LENGTH = 512
    SAMPLE_RATE = 44100

    processor = COLASpectralProcessor(frame_size=WINDOW_SIZE, fft_size=FFT_SIZE, sample_rate=SAMPLE_RATE)

    transient_analyser = TransientDetector(initial_normalization=100, graded_response_slope=1.7,
                                           graded_response_threshold=0.7)

    signal = librosa.load('../test_files/drums.wav', sr=SAMPLE_RATE, mono=True)[0]

    # Process through STFT and ISTFT
    stft_matrix = processor.stft(signal)
    reconstructed_signal = processor.istft(stft_matrix, len(signal))

    print(stft_matrix.shape)
    n_frames = stft_matrix.shape[0]  # Number of frames is first dimension now

    transient_values = np.zeros(n_frames)
    normalised_fluxes = np.zeros(n_frames)

    for i in range(n_frames):
        # Compute transient value and spectral flux for each frame
        transient_values[i], normalised_fluxes[i] = transient_analyser.frame_transient_calculation(stft_matrix[i])

    # Create time arrays for plotting
    signal_time = np.arange(len(signal)) / SAMPLE_RATE
    frame_time = np.arange(n_frames) * HOP_LENGTH / SAMPLE_RATE

    # Create the visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # Plot the audio waveform
    ax1.plot(signal_time, signal)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Audio Signal')

    # Plot the normalized flux
    ax2.plot(frame_time, normalised_fluxes)
    ax2.set_ylabel('Smoothed Normalized\nFlux')
    ax2.set_title('Smoothed Normalized Spectral Flux')

    # Plot the transient values
    ax3.plot(frame_time, transient_values)
    ax3.set_ylim(1, 1.5)
    ax3.set_ylabel('Smoothed Transient\nValue')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Smoothed Transient Detection Values')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig('../test_files/transient_analysis.png')
    plt.close()

    # Save the result
    sf.write('../test_files/output.wav', reconstructed_signal, SAMPLE_RATE)
