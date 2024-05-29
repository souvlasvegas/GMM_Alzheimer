# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:49:35 2024

@author: AndreasMiltiadous
"""
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import mne
from mne_bids import (
    BIDSPath,
    read_raw_bids,
    print_dir_tree,
    make_report,
    find_matching_paths,
    get_entity_vals,
)
import mne.time_frequency
import numpy as np
from mne.time_frequency import psd_array_welch
######################################################################################################
def calculate_relative_band_power(psds, freqs, bands):
    band_powers = {}
    total_power = np.sum(psds, axis=-1, keepdims=True)
    for band, (fmin, fmax) in bands.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.sum(psds[:,:, idx_band], axis=-1, keepdims=True)
        band_powers[band] = band_power / total_power
    return np.concatenate([band_powers[band] for band in bands], axis=-1)


####################################################################################################################
root = tk.Tk()
root.withdraw()

bids_root = filedialog.askdirectory(title='Select BIDS Root Directory')
# define the BIDS root directory and the path to the dataset

# specify the BIDS path
print_dir_tree(bids_root, max_depth=5)

# read the BIDS dataset

entity_key='subject'
sessions = get_entity_vals(bids_root, entity_key)

datatype='eeg'
extensions = [".set"]

bids_paths = find_matching_paths(bids_root, datatypes=datatype, extensions=extensions)


#lets group the data
participants_tsv = os.path.join(bids_root, 'participants.tsv')
participants_df = pd.read_csv(participants_tsv, sep='\t')
grouped = participants_df.groupby('Group')
bids_paths_grouped = {'A': [], 'F': [], 'C': []}

# Populate the BIDS paths lists for each group
for group_name, group_df in grouped:
    if group_name in bids_paths_grouped:
        for _, row in group_df.iterrows():
            subject = row['participant_id'].replace('sub-', '')
            bids_path = BIDSPath(subject=subject, datatype='eeg', root=bids_root)
            if os.path.exists(bids_path.fpath):
                bids_paths_grouped[group_name].append(bids_path)
            else:
                print(f'File not found for subject {subject} in group {group_name}')
'''
for group_name, paths in bids_paths.items():
    print(f"\nProcessing Group {group_name}...")
    i=0
    for bids_path in paths:
        data = mne.io.read_raw_eeglab(bids_path.fpath, preload=False)
        pos=data.info.ch_names
        epochs=mne.make_fixed_length_epochs(data,duration=4,overlap=2)
        print(f'Loaded data for {bids_path.subject} in group {group_name}')
        # Process raw data as needed
'''

bands = {
    'Delta': (0.1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 40)
}

import sys

class suppress_output:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr



columns=["data","subject_id","label"]
df=pd.DataFrame(columns=columns)

#process each group
for group_name, paths in bids_paths_grouped.items():
    print(f"\nProcessing Group {group_name}...")
    #for bids_path, participant_id in paths:
    for path in paths:
        with suppress_output():

            participant_id=path.entities['subject']
            # Load raw data
            raw = mne.io.read_raw_eeglab(bids_path.fpath, preload=True,verbose=0)
            raw.pick_types(eeg=True)
    
            # Create fixed-length epochs
            epochs = mne.make_fixed_length_epochs(raw, duration=4, overlap=2, preload=True,verbose=0)
    
            # Compute PSD using psd_array_welch
            sfreq = raw.info['sfreq']
            psds = []
            freqs = None
            for epoch in epochs.get_data():
                psd, freqs = psd_array_welch(epoch, sfreq=sfreq, fmin=0, fmax=45, n_fft=1024,verbose=0)
                psds.append(psd)
            psds = np.array(psds)
    
            # Calculate Relative Band Power
            rel_band_power = calculate_relative_band_power(psds, freqs, bands)
            
            for i in range(rel_band_power.shape[0]):
                row_data = rel_band_power[i]
                to_append=pd.Series([row_data,participant_id,group_name],index=df.columns)
                df = pd.concat([df, to_append.to_frame().T], ignore_index=True)
            

path_to_save=filedialog.askdirectory(title='Select where to save Directory')
df.to_pickle(path_to_save +'/test_dataset.pkl')

