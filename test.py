import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: knn

# mne.set_log_level('WARNING')

def get_data(channels: int, time_points: int) -> dict[str, np.ndarray]:
    data = dict()

    # maps from set_letter to set_letter_alternate (for the filenames)
    sets = {
        'A': 'Z',
        'B': 'O',
        'C': 'N',
        'D': 'F',
        'E': 'S'
    }

    for set_letter in sets:
        set_letter_alternate = sets[set_letter]

        set_data = np.zeros((channels, time_points))
        for i in range(channels):
            filename = f'data/bonn/SET {set_letter}/{set_letter_alternate}{str(i+1).zfill(3)}.txt'
            z = np.loadtxt(filename)
            set_data[i] = z[:time_points]

        data[set_letter] = set_data

    return data

total_subplots = 3

fig, ax = plt.subplots(total_subplots, sharex=True, sharey=True)
fig.set_tight_layout(True)

subplot = 0
def show(raw, title=''):
    global total_subplots
    global subplot
    total_subplots += 1
    # plt.subplot(total_subplots, 1, subplot)
    # raw.plot_psd(ax=ax[subplot], show=False)
    raw.plot(scalings=500)
    plt.show()
    ax[subplot].set_title(title)
    subplot += 1

channels, time_points = 100, 4096
freq = 173.61
data = get_data(channels, time_points)

set_letter = 'E'

# ------------------------------------ TEST PLOTS ------------------------------------
# channels_to_plot = 50
# duration_to_plot = 20  # in seconds
# info = mne.create_info(ch_names=[f'c{i}' for i in range(channels)], sfreq=freq, ch_types='eeg')
# raw = mne.io.RawArray(data[set_letter], info)
# raw.plot(duration=duration_to_plot, n_channels=channels_to_plot, scalings=300)
# raw.plot_psd(area_mode='range', tmax=10.0, average=False)
# plt.show()

# ------------------------------------ PREPROCESSING ------------------------------------

info = mne.create_info(ch_names=[f'c{i}' for i in range(channels)], sfreq=freq, ch_types='eeg')
raw = mne.io.RawArray(data[set_letter], info)

# semicolon to only graph once https://github.com/mne-tools/mne-python/issues/9832
# raw.plot(title="original data", scalings='auto');

# plot raw data on freq domain (psd = power spectral density)
# raw.plot_psd(area_mode='range', tmax=10.0, average=False)

# _ = raw.filter(None, 60., fir_design='firwin')
# raw.plot_psd(area_mode='range', tmax=10.0, average=False)

show(raw, title='original data')

# 1. NOTCH FILTER

raw.notch_filter(freqs=50)
show(raw, title='after notch filter')

# 2. BANDPASS FILTER

raw.filter(l_freq=0.1, h_freq=50)
show(raw, title='after bandpass filter')

# ICA

plt.show()
