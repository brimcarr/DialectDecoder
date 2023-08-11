import os
import pandas as pd
from prep import cropper, gen_specs, make_metadata

#%% Directories
current_direc = os.getcwd()
### Place where the audio files live
audio_data_direc = current_direc + '/data/cut_songs/'
### Directory you're putting the spectrograms into and later cropping from
spect_direc = current_direc + '/data/testt_spect'
### Place you get the metadata for each file from
metadata = pd.read_csv(current_direc + '/metadata/wizard_metadata.csv')
### Place you save your new metafile to
csv_path = current_direc + '/metadata/testtt.csv'
### Directory where you store the cropped spectrograms
cropped_spec_direc = current_direc + '/data/cropped_spect_testt'

#%% Make the spectrograms
gen_specs.generate_spectrograms(audio_data_direc, spect_direc)

#%% Make the metadata
make_metadata.make_metadata(spect_direc, metadata, csv_path)

#%% Crop the spectrograms
cropper.crop_specs(spect_direc, cropped_spec_direc)