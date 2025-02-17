import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
import soundfile
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift, ifftshift
import gc


def plot_spectrogram(
    left,  # Left signal channel
    right,  # Right signal channel
    fs,  # Signal sampling frequency
    start_min,  # Beginning of interval in minutes
    end_min,  # End of interval in minutes
    window_size=None,  # Tradeoff between spatial and temroral resolution
    vmax=-20,  # Max signal level for color visualization
):
    """Plot spectrogram of the signal."""
    if not window_size:
        window_size = fs
    start = int(fs * 60 * start_min)
    end = int(fs * 60 * end_min)

    fig, ax = plt.subplots(2, figsize=(16, 10), sharey=True, sharex=True)
    fig.suptitle("Spectrogram")
    fig.supylabel("Frequency [Hz]")
    fig.supxlabel("Time [minutes]")

    for i, channel_name, samples in zip(
        (0, 1), ("Left channel", "Right channel"), (left, right)
    ):
        samples = samples[start:end]
        f, t, Zxx = scipy.signal.spectrogram(
            samples,
            fs,
            scaling="spectrum",
            mode="magnitude",
            nperseg=window_size,
            noverlap=window_size // 2,
        )
        magnitude = 10 * np.log10(Zxx)

        ax[i].pcolormesh(
            t / 60 + start_min, f, magnitude, cmap="inferno", vmin=-60, vmax=vmax
        )
        ax[i].set_title(channel_name)
        ax[i].set_yscale("symlog")
        ax[i].set_ylim(20, 20000)
        ax[i].xaxis.set_major_locator(plt.MultipleLocator(1))
        ax[i].xaxis.set_minor_locator(plt.MultipleLocator(1 / 6))
    plt.tight_layout()
    plt.show()


def plot_fft(
    left,  # Left signal channel
    right,  # Right signal channel
    fs,  # Signal sampling frequency
    start_min,  # Beginnig of interval in minutes
    end_min,  # End of interval in minutes
    min_f=20,  # Minimum frequency range
    max_f=1000,  # Maximum frequency range
    log_y=False,  # Log magnitude on log scale
    vmax=-10,  # Maximum signal level, only applicable when used with log scale
    min_peak_dist=-1,  # Minimum distance between detected peaks
):
    """Plot precise spectrum of the signal."""
    start = int(fs * 60 * start_min)
    end = int(fs * 60 * end_min)

    fig, ax = plt.subplots(2, figsize=(16, 8), sharey=True, sharex=True)
    fig.suptitle("FFT")
    fig.supylabel("Magnitude")
    fig.supxlabel("Frequency [Hz]")
    fig.tight_layout()

    for i, channel_name, samples in zip(
        (0, 1), ("Left channel", "Right channel"), (left, right)
    ):
        samples = samples[start:end]

        N = len(samples)
        Z = fft(samples, norm="forward")[: N // 2] * 2
        magnitude = np.abs(Z)
        phase = np.angle(Z)
        f = fftfreq(N, 1.0 / fs)[: N // 2]

        steps_per_hz = N / fs

        if min_peak_dist < 0:
            min_peak_dist = 40
            if max_f - min_f <= 1200:
                min_peak_dist = 4
            if max_f - min_f <= 120:
                min_peak_dist = 0.4
            if max_f - min_f <= 12:
                min_peak_dist = 0.04

        peaks, _ = scipy.signal.find_peaks(
            magnitude, distance=min_peak_dist * steps_per_hz
        )

        if log_y:
            magnitude = 20 * np.log10(magnitude)
            ax[i].set_ylim(-120, vmax)

        ax[i].set_title(channel_name)
        ax[i].plot(f, magnitude, linewidth=0.25)
        ax[i].set_xlim(min_f, max_f)
        ax[i].grid(True, which="major", linestyle="-", linewidth=0.5)
        ax[i].grid(True, which="minor", linestyle="--", linewidth=0.25)
        ax[i].tick_params(axis="x", rotation=90)
        ax[i].minorticks_on()

        ax[i].xaxis.set_major_locator(plt.MultipleLocator(base=100))
        ax[i].xaxis.set_minor_locator(plt.MultipleLocator(base=10))
        if max_f - min_f <= 1200:
            ax[i].xaxis.set_major_locator(plt.MultipleLocator(base=10))
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(base=1))
        if max_f - min_f <= 120:
            ax[i].xaxis.set_major_locator(plt.MultipleLocator(base=1))
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(base=0.1))
        if max_f - min_f <= 12:
            ax[i].xaxis.set_major_locator(plt.MultipleLocator(base=0.1))
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(base=0.01))

        ax[i].plot(f[peaks], magnitude[peaks], ".")
        for peak in peaks:
            f_peak = f[peak]
            if min_f < f_peak < max_f:
                ax[i].annotate(
                    f"{f[peak]:.2f}@{magnitude[peak]:.1f}@{phase[peak]/np.pi:.2f}",
                    xy=(f[peak], magnitude[peak]),
                    xytext=(-0.5, 0.75),
                    textcoords="offset fontsize",
                    rotation=90,
                    fontsize=6,
                )
        del samples
        del magnitude
        del f
        del peaks
        del _
        gc.collect()

    return fig
