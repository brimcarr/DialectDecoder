import os
import csv

#%% Directories

### Location of songs (cut_songs)
file_loc = 'cut_songs'
### Current working directory that contains the folder containing the songs.
current_direc = os.getcwd()
### Generates the root directory we will be iterating over.
rootdir = current_direc+'/'+file_loc

#%% Metadata generation

### Walks through the directories, pulls the files, and puts the name and
### corresponding labels into a .csv file for CNN. Need one elif for each bird
### dialect.

### Initialize list for metadata.
song_data = []
for subdir, _, fls in os.walk(rootdir):
    for fl in fls:
        if 'ABLA' in os.path.join(subdir,fl):
            song_data.append([subdir, fl, 0])
        elif 'BATE' in os.path.join(subdir,fl):
            song_data.append([subdir, fl, 1])
        elif 'BATW' in os.path.join(subdir,fl):
            song_data.append([subdir, fl, 2])
        elif 'COMW' in os.path.join(subdir,fl):
            song_data.append([subdir, fl, 3])
        elif 'FOFU' in os.path.join(subdir,fl):
            song_data.append([subdir, fl, 4])
        elif 'FWSC' in os.path.join(subdir,fl):
            song_data.append([subdir, fl, 5])
        elif 'LAME' in os.path.join(subdir,fl):
            song_data.append([subdir, fl, 6])
        elif 'LODU' in os.path.join(subdir,fl):
            song_data.append([subdir, fl, 7])
        elif 'RICH' in os.path.join(subdir,fl):
            song_data.append([subdir, fl, 8])
        else:
            pass

#%% .csv file generation
header = ['flock_year', 'file_name', 'bird_label']
with open('cut_songs_metadata.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
# write the header
    writer.writerow(header)
# write the data
    for a in song_data:
        writer.writerow(a)
