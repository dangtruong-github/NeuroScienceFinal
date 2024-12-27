"""## Other libraries"""

import numpy as np

import pandas as pd
import time
import os
import mne

import sys

from scipy.fft import fft, fftfreq

from feature_extract_fft import compute_fft_features
from feature_extract_td import extract_time_domain_features_total, extract_time_domain_features

"""## Paths"""

data_dir = './data/raw'
train_csv_file = './data/train_set.csv'
test_csv_file = './data/test_set.csv'
total_csv_file = './data/total.csv'

os.makedirs("logging", exist_ok=True)
sys.stdout = open("./logging/out_extract.txt", "w")
sys.stderr = open("./logging/err_extract.txt", "w")

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

df_fft_channel_fill = None
df_fft_channel_nofill = None
df_fft_total_fill = None
df_fft_total_nofill = None
df_td_channel_fill = None
df_td_channel_nofill = None
df_td_total_fill = None
df_td_total_nofill = None

def add_to_df(old_df, features, name, age, channel_id=None):
    features["name"] = name
    features["age"] = age
    if channel_id is not None:
        features["channel_id"] = channel_id

    if old_df is None:
        old_df = pd.DataFrame(columns=features.keys())

    # Convert dictionary to DataFrame
    features_df = pd.DataFrame([features])

    # Concatenate the DataFrame and the dictionary DataFrame
    new_df = pd.concat([old_df, features_df], ignore_index=True)

    return new_df

def extract_features(file_name, name, age, fs=250):
    global df_fft_channel_fill
    global df_fft_channel_nofill
    global df_fft_total_fill
    global df_fft_total_nofill
    global df_td_channel_fill
    global df_td_channel_nofill
    global df_td_total_fill
    global df_td_total_nofill

    rawdata = mne.io.read_raw_fif(file_name, preload=True)
    eeg_data_orig = rawdata.get_data()

    eeg_data_orig = eeg_data_orig[:52]

    eeg_data_filled = fill_nan_with_last_value(eeg_data_orig)

    assert np.isnan(np.sum(eeg_data_filled)) == False
    
    # Perform Fourier Transform on each channel
    frequencies = fftfreq(eeg_data_filled.shape[1], 1/fs)
    for channel in range(eeg_data_filled.shape[0]):
        signal_fill = eeg_data_filled[channel, :]
        fft_fill_each = fft(signal_fill)

        signal_nofill = eeg_data_orig[channel, :]
        fft_nofill_each = fft(signal_nofill)

        # print(fft_fill_each.shape)
        
        ## fft each channel
        features_fft_channel = compute_fft_features(fft_fill_each, frequencies)
        df_fft_channel_fill = add_to_df(
            df_fft_channel_fill, features_fft_channel,
            name, age, channel_id=channel
        )

        ## df_td_channel_fill
        features_td_channel = extract_time_domain_features(signal_fill)
        df_td_channel_fill = add_to_df(
            df_td_channel_fill, features_td_channel,
            name, age, channel_id=channel
        )
        
        ## fft each channel
        features_fft_channel = compute_fft_features(fft_nofill_each, frequencies)
        df_fft_channel_nofill = add_to_df(
            df_fft_channel_nofill, features_fft_channel,
            name, age, channel_id=channel
        )

        ## df_td_channel_fill
        features_td_channel = extract_time_domain_features(signal_nofill)
        df_td_channel_nofill = add_to_df(
            df_td_channel_nofill, features_td_channel,
            name, age, channel_id=channel
        )

    eeg_data_filled_flat = eeg_data_filled.flatten()
    eeg_data_orig_flat = eeg_data_orig.flatten()
    frequencies = fftfreq(eeg_data_filled_flat.shape[0], 1/fs)
    fft_fill_total = fft(eeg_data_filled_flat)
    fft_nofill_total = fft(eeg_data_orig_flat)

    # print(fft_fill_total.shape)
    # print(fft_nofill_total.shape)

    ## fft each channel
    features_fft_total = compute_fft_features(fft_fill_total, frequencies)
    df_fft_total_fill = add_to_df(
        df_fft_total_fill, features_fft_total,
        name, age
    )

    ## df_td_channel_fill
    features_td_total = extract_time_domain_features_total(eeg_data_filled)
    df_td_total_fill = add_to_df(
        df_td_total_fill, features_td_total,
        name, age
    )

    ## fft each channel
    features_fft_total = compute_fft_features(fft_nofill_total, frequencies)
    df_fft_total_nofill = add_to_df(
        df_fft_total_nofill, features_fft_total,
        name, age
    )

    ## df_td_channel_fill
    features_td_total = extract_time_domain_features_total(eeg_data_orig)
    df_td_total_nofill = add_to_df(
        df_td_total_nofill, features_td_total,
        name, age
    )

    return

# read total file
meta_pd = pd.read_csv(total_csv_file, sep='\t')

print("Start")
start = time.time()
for i in range(len(meta_pd)):
    age = meta_pd.at[i, 'age']
    name = meta_pd.at[i, 'participant_id']
    file_name = os.path.join(data_dir, f'{name}_sflip_parc-raw.fif')

    extract_features(file_name, name, age, fs=250)

end = time.time()
print(f"Time: {end - start}")
df_fft_channel_fill.to_csv("data/csv/fft_channel_fill.csv", index=False)
df_fft_channel_nofill.to_csv("data/csv/fft_channel_nofill.csv", index=False)
df_fft_total_fill.to_csv("data/csv/fft_total_fill.csv", index=False)
df_fft_total_nofill.to_csv("data/csv/fft_total_nofill.csv", index=False)
df_td_channel_fill.to_csv("data/csv/td_channel_fill.csv", index=False)
df_td_channel_nofill.to_csv("data/csv/td_channel_nofill.csv", index=False)
df_td_total_fill.to_csv("data/csv/td_total_fill.csv", index=False)
df_td_total_nofill.to_csv("data/csv/td_total_nofill.csv", index=False)

sys.stdout.close()
sys.stderr.close()
