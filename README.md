Hello! Thank you for your interest in the labeling birds game. Here is an overview of how it all fits together. 

**What you need:** 

A selection of songs that you want to analyze, organized into folders based on labels. For example, in a folder titled cut_songs, there will be subfolders like ABLA_2020, ABLA_2021, COMW_2020, COMW_2021, etc. where each subfolder contains a variety of songs (.wav files) in that class. 

**What each script does:** 

make_metadata.py — Makes a .csv file with the appropriate metadata needed for the CNN and the label game. *** NOTE: THIS WILL NEED TO BE UPDATED TO INCLUDE THE LOCATION DATA, WHICH CAN BE FOUND IN wizard_metadata.py

**Spect_Approach_v2:**

button.py — Makes the button and dropdown classes for the game.

cropper.py — Takes in the spectrograms made by gen_specs.py, crops them down to the appropriate size, and saves them to a new folder (data/cropped_spect_data). You might have to grant python permission to do this, I solved this using “chmod 775 cropper.py”.

gen_specs.py  — The script used to generate the spectrograms from the audio data (.wav files), saved into data/spect_data.

knnpickle_file -- My k-NN network.

model_state_dict2.pth -- My CNN weights.

torch_func.py — An assortment of functions used to create, test, and modify the CNN. Also incudes the apply_cam function 

train_val_test_split.py — The file that splits the data into train/validate/test data.

wizard.py — The script that runs the labeling game. Run this in terminal by navigating to the correct directory and running "python wizard.py" to start the game.

**How to use the software:** (Note, make sure all of your directories/names are correct before running each script)

0. Decide how many classes you want to train on, how many and the type of anomalous classes you want to add in, etc. 

The following steps you should only have to do once

1. Have your songs in folders, similar to what is described above. 
2. Generate the metadata needed using make_metadata.py. This will be used for song classification by assigning labels to each dialect/song. ###NOTE AGAIN THIS WILL NEED TO BE ADJUSTED.
3. Generate spectrograms with gen_specs.py.
4. Then, crop out all of the white space surrounding the spectrograms with cropper.py.

These steps you’ll do more than once depending on what you’re using the labeling birds game for.

5. Split the cropped spectrograms into train_val_test_split.py.
6. Make the original CNN with torch_func.py or use the pretrained weights provided in model_state_dict2.pth.
7. Test the accuracy of the CNN by running the appropriate function in torch_func.py.
8. Run wizard.py with however many images you want to classify.

Of course, the above is just the recommended use. Feel free to play around with the files to accommodate your needs.







