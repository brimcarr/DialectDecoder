import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from PIL import Image
import librosa
import librosa.display
import gc
import random

#%% Function to load in a trained cnn

def load_cnn(cnn_name):
    json_file = open('CNNs/' + cnn_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # Loads pretrained weights into model
    loaded_model.load_weights('CNNs/'+ cnn_name +'.h5')
    loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])
    # loaded_model.summary()
    return loaded_model

#%% Function to convert an image into a CNN input

def img_to_cnn_input(img_path, loaded_model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    prob = np.max(score)
    print( "class : "+ str(np.argmax(score)) +" ; Probability = "+ str(np.max(score)))
    return(np.argmax(score), prob)


#%% Function to crop the spectrograms given by librosa
def crop_specs(spec_direc, cropped_spec_direc):
    for subdir, _, fls in os.walk(spec_direc):
        for fl in fls:
            if not fl.startswith('.'):
                spec = Image.open(subdir + '/' + fl)
                cropped_spec = spec.crop((80, 60, 560, 425))
                bird_type = os.path.basename(os.path.normpath(subdir))
                new_loc = os.path.join(cropped_spec_direc + '/' + bird_type)
                if os.path.isdir(new_loc) == False:
                    os.mkdir(new_loc, 0o666)
                cropped_spec.save(new_loc + '/' + fl)
            else:
                pass 
            
#%% Function to run an image through the CNN
def run_img_thru_cnn(ex_img, spectrograms_directory, model):
    img = tf.keras.preprocessing.image.load_img(spectrograms_directory + ex_img, 
                                                target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print( "class : "+ str(np.argmax(score)) +" ; Probability = "+ str(np.max(score)))
    
#%% Function to run an image through the CNN
def run_img_thru_cnn_class(ex_img, spectrograms_directory, model):
    img = tf.keras.preprocessing.image.load_img(spectrograms_directory + ex_img, 
                                                target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return str(np.argmax(score))

    
#%% Function to display the spectrogram of an audio file
def display_audio_file(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    plt.figure(figsize=(14,7))
    plt.subplot(2, 2, 1)
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(spectrogram, y_axis='linear')
    signal_plot = plt.subplot(2, 2, 2)
    signal_plot.plot(y, color="c")
    plt.show()

#%% Function to generate the spectrograms
def generate_spectrograms(row, spec_direc):
    audio_class = row["bird_label"]
    flock = row["flock_year"]
    bird_label = "class_" + str(audio_class)
    audio_file_name_wo_extension = row["file_name"][:-4]

    y, sr = librosa.load(flock + "/" + row["file_name"])

    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(spectrogram, y_axis='linear')

    plt.savefig(spec_direc + '/' + bird_label + "/" + audio_file_name_wo_extension + ".png")

    plt.clf()
    plt.close('all')
    gc.collect()
    
#%% A function to generate the greyscale spectrograms

# def generate_grey_spectrograms(grey_spec_direc, row):
#     audio_class = row["bird_label"]
#     flock = row["flock_year"]
#     bird_label = "class_" + str(audio_class)
#     audio_file_name_wo_extension = row["file_name"][:-4]

#     y, sr = librosa.load(flock + "/" + row["file_name"])

#     spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
#     librosa.display.specshow(spectrogram, cmap='gray_r', y_axis='linear')

#     plt.savefig(grey_spec_direc + '/' + bird_label + "/" + audio_file_name_wo_extension + ".png")

#     plt.clf()
#     plt.close('all')
#     gc.collect()

#%% A function to make an isolated test data set
def make_iso_specs(spec_direc, isolate_direc, per_of_specs):
    # Make new directory for the test specs if it does not exist
    if os.path.isdir(isolate_direc) == False:
        os.mkdir(isolate_direc, mode = 0o666)
    # Make all appropriate class folders
    spec_classes = os.listdir(spec_direc)
    spec_classes.remove('.DS_Store')
    for bird_type in spec_classes:
        if os.path.isdir(isolate_direc + '/' + bird_type) == False:
            os.mkdir(isolate_direc + '/' + bird_type, mode = 0o666)
        # Move x random images from each class to the isolate test directory
        num_of_specs=0
        for path in os.scandir(spec_direc + '/' + bird_type):
            if path.is_file():
                num_of_specs += 1
        for i in range(math.floor(per_of_specs*0.01*num_of_specs)):
            song_path = os.listdir(spec_direc + '/' + bird_type)
            song = random.choice(song_path)
            old_path = spec_direc + '/' + bird_type + '/' + song
            new_path = isolate_direc + '/' + bird_type + '/' + song
            os.rename(old_path, new_path)
        
#%% A function to put all of the test specs back in their appropriate folder
def reset_test_set(spec_direc, isolate_direc):
    for subdir, _, fls in os.walk(isolate_direc):
        for fl in fls:
            if not fl.startswith('.'):
                bird_type = os.path.basename(os.path.normpath(subdir))
                test_path = subdir + '/' + fl
                spec_path = spec_direc + '/' + bird_type + '/' + fl
                os.rename(test_path, spec_path)







