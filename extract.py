"""## Other libraries"""

import numpy as np
import pywt
import matplotlib.pyplot as plt

import pandas as pd
import torch
import time
from torch.utils.data import Dataset
import os
import mne

import sys

from scipy.fft import fft, fftfreq
import gc

from feature_extract import compute_fft_features

"""## Paths"""

data_dir = './data/raw'
train_csv_file = './data/train_set.csv'
test_csv_file = './data/test_set.csv'
total_csv_file = './data/total.csv'

os.makedirs("logging", exist_ok=True)
sys.stdout = open("./logging/out.txt", "w")
sys.stderr = open("./logging/err.txt", "w")

"""# Test"""

def fft_plot(fft_result_channel, frequencies, channel_id, name, age, folder_save):
    file_save = os.path.join(folder_save, f'Age-{age}_Name-{name}.png')
    # Example: Plot the Fourier Transform of the first channel
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(fft_result_channel))
    plt.title(f'Fourier Transform - Channel {channel_id} - Participant {name} - Age {age}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.savefig(file_save)
    plt.close()

    print(f"Fourier Transform plot saved to {file_save}")


# Sampling frequency (Hz)
fs = 1000

def fill_nan_with_last_value(eeg_data):
    """
    Fills NaN values in EEG data with the last non-NaN value along each channel (axis 0).
    
    Parameters:
        eeg_data (numpy.ndarray or pandas.DataFrame): The EEG data where rows represent time steps
                                                      and columns represent channels.
    
    Returns:
        numpy.ndarray: The EEG data with NaN values filled.
    """
    mask = np.isnan(eeg_data)
    idx = np.where(~mask,np.arange(eeg_data.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = eeg_data[np.arange(idx.shape[0])[:,None], idx]

    nan_indices = np.where(np.isnan(out))

    print(nan_indices)
    # Convert back to numpy array if needed
    return out

final_df = None

def fft_52(file_name, name, age, fs=250, folder_save="./plot_channels"):
    global final_df
    rawdata = mne.io.read_raw_fif(file_name, preload=True)
    eeg_data_orig = rawdata.get_data()

    eeg_data_orig = eeg_data_orig[:52]

    eeg_data_fft = fill_nan_with_last_value(eeg_data_orig)

    assert np.isnan(np.sum(eeg_data_fft)) == False
    # eeg_data_fft = eeg_data_orig
    
    # Perform Fourier Transform on each channel
    # fft_results = []
    frequencies = fftfreq(eeg_data_fft.shape[1], 1/fs)
    for channel in range(eeg_data_fft.shape[0]):
        signal = eeg_data_fft[channel, :]
        fft_result = fft(signal)
        # fft_results.append(fft_result)
        features = compute_fft_features(fft_result, frequencies)

        features["name"] = name
        features["age"] = age
        features["channel_id"] = channel

        if final_df is None:
            final_df = pd.DataFrame(columns=features.keys())

        # Convert dictionary to DataFrame
        features_df = pd.DataFrame([features])

        # Concatenate the DataFrame and the dictionary DataFrame
        final_df = pd.concat([final_df, features_df], ignore_index=True)

    # fft_results = np.array(fft_results)

    return

# read total file
meta_pd = pd.read_csv(total_csv_file, sep='\t')

print("Start")
start = time.time()
for i in range(len(meta_pd)):
    age = meta_pd.at[i, 'age']
    name = meta_pd.at[i, 'participant_id']
    file_name = os.path.join(data_dir, f'{name}_sflip_parc-raw.fif')

    fft_52(file_name, name, age,
           fs=250, folder_save="./plot_channels_total")

end = time.time()
print(f"Time: {end - start}")
final_df.to_csv("data/fill_channel.csv", index=False)

sys.stdout.close()
sys.stderr.close()
