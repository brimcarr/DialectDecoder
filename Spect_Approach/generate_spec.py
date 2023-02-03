# # https://www.kaggle.com/code/karimsaieh/urbansound8k-classification-cnn-keras-librosa/notebook
import os
import gc
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
tqdm.pandas()
from functions import generate_spectrograms

#%% Establish needed directories and where the spectrograms need to go
current_direc = os.getcwd()
approach_direc = os.path.basename(os.path.normpath(current_direc))
needed_direc = current_direc.replace(approach_direc, '')

audio_data_direc = needed_direc + 'cut_songs/'
spect_direc = current_direc + '/sandy_specs/'

df = pd.read_csv(needed_direc + 'cut_songs_metadata.csv')


#%% Calls the function to generate spectrograms

# Makes a folder for each class and puts each spectrogram in the appropriate folder
for i in df.bird_label.unique():
    Path(spect_direc + "class_" + str(i)).mkdir(parents=True, exist_ok=True)
# Generates the spectrograms
df.progress_apply(generate_spectrograms, spec_direc = spect_direc, axis=1)

#%% Calls the function to generate the greyscale spectrograms

# # Makes a folder for each class and puts each spectrogram in the appropriate folder
# for i in df.bird_label.unique():
#     Path(grey_spectrograms_directory + "class_" + str(i)).mkdir(parents=True, exist_ok=True)
# # Generates the spectrograms
# df.progress_apply(generate_grey_spectrograms, spec_direc = spect_direc, axis=1)

