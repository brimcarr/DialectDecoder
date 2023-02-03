Hello! Thank you for your interest in the labeling birds game. Here is an overview of how it all fits together. 

What you need: A selection of songs that you want to analyze. See the cut_songs folder for an example of how the songs are organized and sorted.

What each script does: 

make_metadata.py — Makes a .csv file with the appropriate metadata needed for the CNN and the label game. 

Spect_Approach:

button.py — Makes the button and dropdown classes for the game.

cnn_accuracy_test.py — Evaluates the accuracy of CNN’s based on partitioned testing data.

cropper.py — Takes in the spectrograms made by generate_spec.py, crops them down to the appropriate size, and saves them to a new folder (note, the new folder has to exist). You might have to grant python permission to do this, I solved this using “chmod 775 cropper.py”.

freeze_update_cnn.py — Takes in a model, freezes the first 10 layers, chops off the last 4 layers, adds in 4 new layers, and trains those 4 new layers. 

functions.py — An assortment of functions used across the different python files. 
	•	load_cnn
	•	img_to_cnn_input
	•	crop_specs
	•	run_img_thru_cnn
	•	display_audio_file
	•	generate_spectrograms
	•	make_iso_specs
	•	reset_test_set

generate_spec.py  — The script used to generate the spectrograms from the audio data.

label_game.py — The script that runs the labeling game. Run this in terminal to start the game.

make_cnn.py — The file that makes and trains CNN’s.

make_test_set.py — The file that pulls out random spectrograms from each class for testing purposes.

reset_test_specs.py — The file that puts all of the test spectrograms back so you can change the test size and do another random trial.

How to use the software: (Note, make sure all of your directories/names are correct before running each script)

0. Decide how many classes you want to train on, how many and the type of anomalous classes you want to add in, etc. 

The following steps you should only have to do once
1. Have your songs in folders, similar to what is shown in the cut_songs folder. 
2. Generate the metadata needed using make_metadata.py. This will be used for song classification by assigning labels to each dialect/song.
3. Generate spectrograms and sort them into the appropriate classes with generate_spec.py.
4. Then, crop out all of the white space with cropper.py

These you’ll do more than once depending on what you’re using the labeling birds game for.
5. Pull out x images from each class to make a test dataset with make_test_set.py (can undo with reset_test_specs.py)
6. Make the original CNN with make_CNN.py
7. Test the accuracy of the CNN by running cnn_accuracy_test.py on your CNN.
8. Play the label_game.py with however many images you want to classify.
9. After labeling all of the images, retrain the CNN with freeze_update_cnn.py
10. Test the accuracy of the CNN with cnn_accuracy_test.py.

Of course, the above is just the recommended use. Feel free to play around with the files to accommodate your needs.







