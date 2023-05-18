import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import gc
from tqdm import tqdm

current_direc = os.getcwd()
approach_direc = os.path.basename(os.path.normpath(current_direc))
needed_direc = current_direc.replace(approach_direc, '')
audio_data_direc = needed_direc + 'cut_songs/'
spect_direc = current_direc + '/data/spect_data'

def generate_spectrograms(audio_data_direc, spec_direc):
    # generates spectrograms from raw audio files
    for root, _, files in os.walk(audio_data_direc):
        for wav in tqdm(files):
            if not wav.startswith('.'):
                y, sr = librosa.load(os.path.join(root, wav))
    
                spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(spectrogram, y_axis='linear')
    
                bird_type = os.path.basename(root)
                file_name = os.path.splitext(wav)[0] + '.png'
    
                os.makedirs(os.path.join(spec_direc, bird_type), exist_ok=True)
                plt.savefig(os.path.join(spec_direc, bird_type, file_name))
    
                plt.clf()
                plt.close('all')
                gc.collect()
            else:
                pass


generate_spectrograms(audio_data_direc, spect_direc)