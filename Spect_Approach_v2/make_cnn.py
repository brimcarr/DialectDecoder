# https://www.kaggle.com/code/karimsaieh/urbansound8k-classification-cnn-keras-librosa/notebook
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from functions import run_img_thru_cnn, display_audio_file

### Directories
current_direc = os.getcwd()
spec_direc = '/specs_training/'
audio_direc = '/cut_songs_sorted/'

audio_data_directory = current_direc[-14] + audio_direc
spectrograms_directory = current_direc + spec_direc

# Note: Name your CNN's differently to keep track of them.
cnn_name = '80_percent_data_2'

#%% Display a file
# bird_labels = ["ABLA", "BATE", "BATW" "COMW", "FOFU", "FWSC", "LAME", " LODU", "RICH"]
# display_audio_file('/Users/story/Documents/Birdz/cut_songs/ABLA_2020/AB1_20200428r1_07100.wav')

#%% Makes the training and testing datasets
X_train = tf.keras.preprocessing.image_dataset_from_directory(spectrograms_directory,
                                                              validation_split = 0.225,
                                                              subset = "training", 
                                                              seed=7)
X_test = tf.keras.preprocessing.image_dataset_from_directory(spectrograms_directory,
                                                              validation_split = 0.225,
                                                              subset="validation", 
                                                              seed=7)

#%% Construct the CNN model
model = Sequential([

    layers.experimental.preprocessing.Rescaling(1./255,  input_shape=(256, 256, 3)),

    layers.Conv2D(filters = 256, kernel_size = 5, strides = 4, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D(pool_size=(5,5)),

    layers.Conv2D(filters = 128, kernel_size = 3, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D(pool_size=(3,3)),

    layers.Flatten(),

    layers.Dense(units = 128),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Dense(units = 9, activation='softmax')
]) 
# Note: For the final layer, your number of units should match the number of 
# classes you're trying to classify your data into.


#%% Display a summary of the model and run the model
model.summary()
# Compiles the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
# Runs the CNN
model_history = model.fit(X_train, 
                          validation_data=X_test, 
                          batch_size=64, 
                          epochs=35)


#%% Save the model and weights to be used in bird game

model_json = model.to_json()
with open('CNNs/' + cnn_name + '.json',"w") as json_file:
    json_file.write(model_json)
model.save_weights('CNNs/' + cnn_name + '.h5')
print("Saved model to disk")

#%% Plots the loss and accuracy over time
plt.figure(figsize=(15,5))

plt_loss = plt.subplot(121)
plt_loss.plot(model_history.history["loss"])
plt_loss.plot(model_history.history["val_loss"])
# plt.title("")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="upper right")

plt_accuracy = plt.subplot(122)
plt_accuracy.plot(model_history.history["accuracy"])
plt_accuracy.plot(model_history.history["val_accuracy"])
# plt.title("")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="lower right")

plt.show()

#%% Run an example image through the network
ex_img = '/class_1/Com_A_21_1236_06988.png'
run_img_thru_cnn(ex_img, spectrograms_directory, model)
