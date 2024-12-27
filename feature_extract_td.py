import numpy as np
from scipy.stats import skew, kurtosis
import json

def extract_time_domain_features(eeg_data_channel):
    """
    Extracts time-domain features from EEG data and converts all values to Python float.
    
    Parameters:
        eeg_data (numpy.ndarray): EEG data where rows represent time steps
                                  and columns represent channels.
    
    Returns:
        dict: A dictionary containing extracted features for each channel.
    """
    features = {}

    features = {
        "Mean": float(np.mean(eeg_data_channel)),
        "Median": float(np.median(eeg_data_channel)),
        "Variance": float(np.var(eeg_data_channel)),
        "Standard_Deviation": float(np.std(eeg_data_channel)),
        "Skewness": float(skew(eeg_data_channel)),
        "Kurtosis": float(kurtosis(eeg_data_channel)),
        "Energy": float(np.sum(eeg_data_channel ** 2)),
        "RMS": float(np.sqrt(np.mean(eeg_data_channel ** 2))),
        "Maximum": float(np.max(eeg_data_channel)),
        "Minimum": float(np.min(eeg_data_channel)),
        "Zero_Crossings": int(np.sum(np.diff(np.sign(eeg_data_channel)) != 0)),  # Convert to int
        "Signal_Slope": float(np.mean(np.gradient(eeg_data_channel)))
    }
    
    return features

def extract_time_domain_features_total(eeg_data, per_channel=True):
    
    features = {}
    # Compute features for all channels combined
    flattened_data = eeg_data.flatten()

    # Handling NaN values
    flattened_data = np.nan_to_num(flattened_data, nan=0.0)

    # Compute features and convert to float
    features = {
        "Mean": float(np.mean(flattened_data)),
        "Median": float(np.median(flattened_data)),
        "Variance": float(np.var(flattened_data)),
        "Standard_Deviation": float(np.std(flattened_data)),
        "Skewness": float(skew(flattened_data)),
        "Kurtosis": float(kurtosis(flattened_data)),
        "Energy": float(np.sum(flattened_data ** 2)),
        "RMS": float(np.sqrt(np.mean(flattened_data ** 2))),
        "Maximum": float(np.max(flattened_data)),
        "Minimum": float(np.min(flattened_data)),
        "Zero_Crossings": int(np.sum(np.diff(np.sign(flattened_data)) != 0)),
        "Signal_Slope": float(np.mean(np.gradient(flattened_data)))
    }
    
    return features

if __name__ == "__main__":
    # Example usage
    # Simulate EEG data with random values (rows = time steps, columns = channels)
    eeg_data = np.random.rand(1000, 52)  # 1000 time steps, 52 channels

    # Extract features
    time_domain_features = extract_time_domain_features(eeg_data)

    # Save features to a JSON file
    with open("eeg_features.json", "w") as json_file:
        json.dump(time_domain_features, json_file, indent=4)

    print("Time-domain features saved to eeg_features.json.")
