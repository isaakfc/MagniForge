# MagniForge

- [Overview](#overview)
- [Technical Implementation](#technical-implementation)
  - [Transient Detection](#transient-detection)
  - [Bass Enhancement](#bass-enhancement)
  - [Transient Modification](#transient-modification)
  - [Subharmonic Generation](#subharmonic-generation)
- [Development Status](#development-status)
- [Future Work](#future-work)
- [References](#references)

A low drum enhancement audio plugin that uses intelligent transient detection and multiple processing stages to deliver powerful, clean drum enhancement.

## Overview
MagniForge uses advanced DSP techniques including:
- Continuous transient detection and scoring
- Nonlinear transient amplitude modification 
- Subharmonic synthesis
- Intelligent bass enhancement

The system adapts its processing based on the detected transient content, allowing for impactful drum hits while maintaining clarity in the resonant portions of the sound.

## Technical Implementation

### Transient Detection
The transient detection system is based on research by Goodwin and Avendano [1], implementing a continuous transient scoring approach rather than binary detection. This provides nuanced control over the processing chain:

<img src="https://github.com/user-attachments/assets/3edad215-1bfa-4858-ace4-f7e0f3a82080" width="600" alt="Transient analysis showing input signal (top), smoothed normalized flux (middle), and smoothed graded response (bottom)">

*Transient analysis showing input signal (top), smoothed normalized flux (middle), and smoothed graded response (bottom)*

The implementation includes:
- Spectral flux calculation
- Moving average smoothing
- Median filtering for the graded response
- Adaptive thresholding

### Bass Enhancement
The bass enhancement chain follows the architecture proposed by Hill and Hawksford [2]. The signal flow is:

```
Input -> LPF -> NLD -> DC Blocker -> BPF -> Gain -> Sum with delayed input -> Output
```

The implementation includes:
- 4th order Bessel LPF for clean low frequency isolation
- Exponential NLD for natural harmonic generation
- DC blocking filter to remove offset introduced by NLD
- 2nd order Bessel BPF for harmonic shaping
- Delay-matched direct signal mixing calculated via total system latency
- Automatic NLD gain compensation

<img src="https://github.com/user-attachments/assets/5c6e733d-4dc6-4e23-8d27-740f2918b631" width="600" alt="Frequency, phase and group delay responses of the Bessel filters">

*Frequency, phase and group delay responses of the Bessel filters used in bass enhancement*

The NLD implementation uses an exponential function approximation via a Taylor Series with controllable harmonic content as described by Oo and Gan [3]. Even harmonic reduction is available to minimize intermodulation distortion.

### Transient Modification
The transient modification system implements a nonlinear magnitude modification in the frequency domain, adapting its processing based on the transient detection score. This approach provides:
- Natural enhancement of transient events
- Preservation of phase information
- Smooth transitions between processing states

### Subharmonic Generation
The subharmonic generator uses:
- Peak detection with moving average
- Hysteresis for stable frequency tracking
- Gaussian windowing for artifact-free synthesis
- Energy-dependent subharmonic mixing

## Development Status
Current implementation is in Python with plans to port to C++/JUCE for plugin deployment. Core DSP modules are completed and tested individually. Integration of the transient detection system with adaptive parameter control for other processing classes is in development.

## Future Work
- Complete integration of transient-adaptive processing
- Implement C++/JUCE plugin framework
- Add user interface with real-time visualization
- Optimise DSP for real-time performance

## References
[1] Goodwin, M., & Avendano, C. (2006). "Frequency-Domain Algorithms for Audio Signal Enhancement Based on Transient Modification"  
[2] Hill, A. J., & Hawksford, M. O. J. (2010). "A hybrid virtual bass system for optimized steady-state and transient performance"  
[3] Oo, N., & Gan, W. S. (2008). "Harmonic Analysis of Nonlinear Devices for Virtual Bass System"  
