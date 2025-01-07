import numpy as np
from scipy import signal


class BassEnhancement:
    """
    Implementation of a NLD system for bass enhancement. The NLD path goes through:
    LPF -> NLD-> DC blocker -> BPS and is then mixed with a delayed version of the original signal. The original
    signal is delayed based on the total system latency created from the filters. The NLD is based on a taylor series
    approximation of the exponential function which goes up to the 6th polynomial.

    Args:
        sample_rate: Sample rate for filter coefficient calculations
        nld_drive: The base of the NLD function (b^x)
    """
    def __init__(self, sample_rate=44100, nld_drive=5.5, nld_even_reduction=1.0):
        # These values were pre-determined looking at the group delay of the filters
        self.lpf_delay_ms = 4.91
        self.dc_delay_ms = 1.43
        self.bpf_delay_ms = 1.00
        self.total_delay_ms = self.lpf_delay_ms + self.dc_delay_ms + self.bpf_delay_ms

        # Filter parameters, allowing up to 6 harmonics from the lpf cutoff
        self.lpf_order = 4
        self.bpf_order = 2
        self.lpf_cutoff = 100  # Hz
        self.bpf_low = 100  # Hz
        self.bpf_high = 600  # Hz

        # NLD parameters
        if nld_drive < 1.0:
            raise ValueError("NLD drive should be greater than 1")
        self.nld_drive = nld_drive
        self.nld_even_reduction = nld_even_reduction

        # Amount of DC blocking
        self.dc_alpha = 0.995
        # Output gain which is applied after the BPS
        self.output_gain = 5.0
        # Initialise with input sample rate
        self.initialize_system(sample_rate)

    def initialize_system(self, sample_rate):
        """
        Initialises all the filters states
        """
        self.sample_rate = sample_rate
        nyq = sample_rate / 2

        # Initialise delay buffer based on total system latency
        self.delay_samples = int(self.total_delay_ms * sample_rate / 1000)
        self.delay_buffer = np.zeros(self.delay_samples)
        self.delay_index = 0

        # Create filter (ignore warnings in pycharm)
        self.lpf_b, self.lpf_a = signal.bessel(self.lpf_order, self.lpf_cutoff / nyq, 'lowpass')
        self.bpf_b, self.bpf_a = signal.bessel(self.bpf_order,
                                               [self.bpf_low / nyq, self.bpf_high / nyq],
                                               'bandpass')

        # Initialise filter states
        self.lpf_state = signal.lfilter_zi(self.lpf_b, self.lpf_a)
        self.bpf_state = signal.lfilter_zi(self.bpf_b, self.bpf_a)
        self.dc_state = 0.0

    def process_nld(self, x):
        """
        Applies Taylor series approximation of the exponential function for our NLD with automatic gain compensation
        """
        ln_b = np.log(self.nld_drive)
        result = 1.0
        result += ln_b * x
        result += (ln_b ** 2 / 2) * x ** 2 * self.nld_even_reduction
        result += (ln_b ** 3 / 6) * x ** 3
        result += (ln_b ** 4 / 24) * x ** 4 * self.nld_even_reduction
        result += (ln_b ** 5 / 120) * x ** 5
        result += (ln_b ** 6 / 720) * x ** 6 * self.nld_even_reduction

        # Reduce gain based on drive parameter
        output_gain = 1.0 / (1 + np.log(self.nld_drive))
        return result * output_gain

    def process_dc_blocker(self, x):
        """
        Process sample through DC blocking filter
        """
        y = x - self.dc_state
        self.dc_state = self.dc_alpha * self.dc_state + (1 - self.dc_alpha) * x
        return y

    def update_delay_line(self, x):
        """
        Update the delay line circular buffer and return delayed sample
        """
        # Get delayed sample (delay index is calculated from total system latency)
        delayed = self.delay_buffer[self.delay_index]

        # Update circular buffer
        self.delay_buffer[self.delay_index] = x
        self.delay_index = (self.delay_index + 1) % self.delay_samples

        return delayed

    def process_sample(self, x):
        """
        Process a sample through the full NLD chain LPF -> NLD-> DC blocker -> BPS and combine with delayed original
        signal.
        """
        # Store input in delay line and get delayed version
        delayed = self.update_delay_line(x)

        # Process sample through LPF
        x_lpf, self.lpf_state = signal.lfilter(self.lpf_b, self.lpf_a,
                                               [x], zi=self.lpf_state)

        # Apply exponential NLD
        x_nld = self.process_nld(x_lpf[0])

        # Remove DC offset
        x_dc = self.process_dc_blocker(x_nld)

        # Process sample through BPF
        x_bpf, self.bpf_state = signal.lfilter(self.bpf_b, self.bpf_a,
                                               [x_dc], zi=self.bpf_state)

        # Apply amke up gain and combine with original signal
        processed = x_bpf[0] * self.output_gain

        return delayed + processed

    def process_block(self, input_block):
        """
        Loop through block and process
        """""
        output_block = np.zeros_like(input_block)

        for i in range(len(input_block)):
            output_block[i] = self.process_sample(input_block[i])

        return output_block



