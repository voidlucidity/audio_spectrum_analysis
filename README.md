# Analyzing the Spectrum of Audio Signals

This tool allows for the visualization and precise analysis of frequencies and
phases in audio signals. I developed it to identify the exact types and
frequencies of brainwave entrainment signals used in various audio programs. It
can detect and analyze the following signals:
- Binaural beat frequencies
- Tone spatial angle modulation type and frequency
- Tone panning
- Monaural beats
- Isochronic tones
- Certain background noise effects

Use [spectrum_analysis.ipynb](spectrum_analysis.ipynb) as a starting setup.

## Notes:

- Analysis should be performed on intervals where the signal is stationary to
obtain an accurate composition of frequencies.
- The longer the interval, the higher the frequency resolution and the lower the
noise floor.
- If a tone is masked by noise, it can often still be measured by using a
sufficiently long interval.
- The code is quite slow for long intervals and consumes a lot of RAM, but it is
capable of analyzing audio files up to two hours long. This allows to achive an
extremely low noise floor and highly precise frequency measurements.