import os
import csv
from get_location import get_location
import pandas as pd

#%% Directories
# ### Current working directory that contains the folder containing the songs.
# current_direc = os.getcwd()
# ### Root directory we will be iterating over.
# rootdir = current_direc + '/data/experiment_0/isolated'
# ### Location Data
# metadata = pd.read_csv(current_direc + '/metadata/wizard_metadata.csv')
# ### Path to save the .csv file
# csv_path = current_direc + '/metadata/exp0/isolated_metadata_exp0.csv'

#%% Metadata generation
### Walks through the directories, pulls the files, and puts the name and
### corresponding labels into a .csv file for CNN. Need one elif for each bird
### dialect.

def make_metadata(rootdir, metadata, csv_path):
### Initialize list for metadata.
    song_data = []
    for subdir, _, fls in os.walk(rootdir):
        for fl in fls:
            lat, long = get_location(fl, metadata)
            if 'ABLA' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 0, lat, long])
            elif 'BATE' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 1, lat, long])
            elif 'BATW' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 2, lat, long])
            elif 'COMW' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 3, lat, long])
            elif 'FOFU' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 4, lat, long])
            elif 'FWSC' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 5, lat, long])
            elif 'LAME' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 6, lat, long])
            elif 'LODU' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 7, lat, long])
            elif 'RICH' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 8, lat, long])
            else:
                pass
    ### .csv file generation
    header = ['flock_year', 'file_name', 'bird_label', 'latitude', 'longitude']
    with open(csv_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
    # write the header
        writer.writerow(header)
    # write the data
        for a in song_data:
            writer.writerow(a)
