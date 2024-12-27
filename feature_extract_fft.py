import numpy as np
from scipy.stats import entropy, kurtosis, skew

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compute_fft_features(fft_amplitudes, freq):
    def band_power(fft_amplitudes, freq, band):
        band_indices = (freq >= band[0]) & (freq <= band[1])
        result = np.sum(np.square(fft_amplitudes[band_indices]))

        return float(result)

    def spectral_centroid(fft_amplitudes, freq):
        power = np.square(fft_amplitudes)
        result = np.sum(freq * power) / np.sum(power) if np.sum(power) > 0 else 0

        return float(result)

    def spectral_bandwidth(fft_amplitudes, freq, centroid=None):
        if centroid is None:
            centroid = spectral_centroid(fft_amplitudes, freq)
        power = np.square(fft_amplitudes)
        result = np.sqrt(np.sum(power * (freq - centroid)**2) / np.sum(power)) if np.sum(power) > 0 else 0

        return float(result)

    def peak_frequency(fft_amplitudes, freq):
        peak_idx = np.argmax(fft_amplitudes)
        result = freq[peak_idx]

        return float(result)

    def spectral_entropy(fft_amplitudes):
        power = np.square(fft_amplitudes)
        power /= np.sum(power)  # Normalize power
        result = entropy(power)

        return float(result)

    def power_ratio(fft_amplitudes, freq, band1, band2):
        power1 = band_power(fft_amplitudes, freq, band1)
        power2 = band_power(fft_amplitudes, freq, band2)
        result = power1 / power2 if power2 > 0 else 0

        return float(result)

    def harmonic_ratio(fft_amplitudes, freq, fundamental_freq):
        fundamental_idx = np.argmin(np.abs(freq - fundamental_freq))
        harmonic_idxs = [np.argmin(np.abs(freq - n * fundamental_freq)) for n in range(2, 5)]
        fundamental_power = np.square(fft_amplitudes[fundamental_idx])
        harmonic_power = np.sum([np.square(fft_amplitudes[i]) for i in harmonic_idxs if i < len(fft_amplitudes)])
        result = fundamental_power / harmonic_power if harmonic_power > 0 else 0

        return float(result)

    def dominant_band(fft_amplitudes, freq, bands):
        max_power = 0
        dominant = None
        for band in bands:
            power = band_power(fft_amplitudes, freq, band)
            if power > max_power:
                max_power = power
                dominant = band
        return dominant

    def statistical_features(fft_amplitudes):
        return {
            "mean": float(np.mean(fft_amplitudes)),
            "median": float(np.median(fft_amplitudes)),
            "variance": float(np.var(fft_amplitudes)),
            "skewness": float(skew(fft_amplitudes)),
            "kurtosis": float(kurtosis(fft_amplitudes))
        }

    def spectrum_slope(fft_amplitudes, freq):
        log_freq = np.log(freq + 1e-12)  # Avoid log(0)
        log_power = np.log(np.square(fft_amplitudes) + 1e-12)
        slope, _ = np.polyfit(log_freq, log_power, 1)
        return slope

    # Define frequency bands
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (13, 30),
        "Gamma": (30, 50)
    }

    # Compute features
    features = {}
    features["band_power"] = {band: band_power(fft_amplitudes, freq, range_) for band, range_ in bands.items()}
    features["relative_band_power"] = {band: features["band_power"][band] / sum(features["band_power"].values()) for band in bands}
    features["peak_frequency"] = peak_frequency(fft_amplitudes, freq)
    # features["spectral_entropy"] = spectral_entropy(fft_amplitudes)
    features["spectral_centroid"] = spectral_centroid(fft_amplitudes, freq)
    features["spectral_bandwidth"] = spectral_bandwidth(fft_amplitudes, freq)
    features["dominant_band"] = dominant_band(fft_amplitudes, freq, list(bands.values()))
    features["statistical_features"] = statistical_features(fft_amplitudes)
    # features["spectrum_slope"] = spectrum_slope(fft_amplitudes, freq)

    # Example ratios (Alpha/Beta, Beta/Gamma)
    features["power_ratios"] = {
        "Alpha/Beta": power_ratio(fft_amplitudes, freq, bands["Alpha"], bands["Beta"]),
        "Beta/Gamma": power_ratio(fft_amplitudes, freq, bands["Beta"], bands["Gamma"])
    }

    features = flatten_dict(features)

    return features
