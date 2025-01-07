import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from src.processing import COLASpectralProcessor
from src.analysis import TransientDetector
from src.processing.transient_processing.transient_modifier import TransientModifier
from src.processing.transient_processing.bass_enhancer import BassEnhancement

if __name__ == "__main__":
    # Initialize processor with your parameters
    FFT_SIZE = 2048
    WINDOW_SIZE = 1024
    HOP_LENGTH = 512
    SAMPLE_RATE = 44100

    processor = COLASpectralProcessor(frame_size=WINDOW_SIZE, fft_size=FFT_SIZE, sample_rate=SAMPLE_RATE)

    transient_analyser = TransientDetector(initial_normalization=100, graded_response_slope=1.7,
                                           graded_response_threshold=0.7)

    transient_modifier = TransientModifier(use_nonlinear=True)

    signal = librosa.load('../test_files/drums.wav', sr=SAMPLE_RATE, mono=True)[0]

    # Extract STFT matrix
    stft_matrix = processor.stft(signal)

    # Set up arrays and get number of frames
    n_frames = stft_matrix.shape[0]
    modified_stft = np.zeros_like(stft_matrix)
    transient_values = np.zeros(n_frames)

    # Loop through STFT frames
    for i in range(n_frames):
        # Extract frame from array
        frame = stft_matrix[i]
        # Detect transients
        transient_values[i], _ = transient_analyser.frame_transient_calculation(frame)
        # Modify the frame based on transient value
        modified_stft[i] = transient_modifier.process_frame(frame, transient_values[i])

    # Reconstruct modified signal
    processed_signal = processor.istft(modified_stft, len(signal))

    # Create time array for plotting
    signal_time = np.arange(len(signal)) / SAMPLE_RATE

    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Plot the original audio waveform
    ax1.plot(signal_time, signal)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original Signal')

    # Plot the processed audio waveform
    ax2.plot(signal_time, processed_signal)
    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('Processed Signal')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig('../test_files/transient_processing.png')
    plt.close()

    # Save the processed audio
    sf.write('../test_files/processed_output.wav', processed_signal, SAMPLE_RATE)