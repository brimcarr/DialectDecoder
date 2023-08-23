### https://keras.io/guides/transfer_learning/#an-endtoend-example-finetuning-an-image-classification-model-on-a-cats-vs-dogs-dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, model_from_json
from functions import load_cnn, run_img_thru_cnn

#%% Establishes the directories, loads CNN from file, and creates the 
### transfer learning CNN
current_direc = os.getcwd()
spec_direc = current_direc + '/trial_training_data'

old_cnn_name = 'trial_model_2_classes'
new_cnn_name = 'trial2_model_3_classes_270'
loaded_model = load_cnn(old_cnn_name)

#%% Loads first 10 layers out of the 14 layers of the original model
layer_names=[layer.name for layer in loaded_model.layers]
new_model = Sequential()
for i in range(0,(len(layer_names)-4)):
    new_model.add(loaded_model.get_layer(layer_names[i]))
    new_model.layers[i].trainable = False


#%% Adds in new layers
new_model.add(layers.Dense(units = 128))
new_model.add(layers.BatchNormalization(name = 'batch_norm_3'))
new_model.add(layers.Activation("relu", name = 'activation_2'))
new_model.add(layers.Dense(units = 3, activation='softmax'))


#%% Load data, compile model, and run the model. 

### Make train/test data
X_train = tf.keras.preprocessing.image_dataset_from_directory(spec_direc,
                                                              validation_split = 0.225,
                                                              subset = "training", 
                                                              seed=7)
X_test = tf.keras.preprocessing.image_dataset_from_directory(spec_direc,
                                                             validation_split = 0.225,
                                                             subset="validation", 
                                                             seed=7)

new_model.summary()

new_model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
# Runs the CNN
model_history = new_model.fit(X_train, 
                              validation_data=X_test, 
                              batch_size=64, 
                              epochs=20)

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
# img_name = '/class_2/BW1_20200422r1_08330.png'
# run_img_thru_cnn(img_name, spec_direc, new_model)


#%% Save the model and weights to be used in bird game
model_json = new_model.to_json()
with open('CNNs/'+ new_cnn_name + '.json',"w") as json_file:
    json_file.write(model_json)
new_model.save_weights('CNNs/'+ new_cnn_name +'.h5')
print("Saved model to disk")

