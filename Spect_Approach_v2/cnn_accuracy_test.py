"""
Test the networks classification abilites
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from functions import load_cnn


### Name of CNN you're testing
cnn_name = '80_percent_data_1' 

#%% Testing data directory
current_direc = os.getcwd()
test_data_direc = current_direc + '/specs_testing'

#%% Loads the CNN from file and creates the model to be used for classification

loaded_model = load_cnn(cnn_name)

#%% Loads in the testing data and runs it through the network
comp_ans = []
true_ans = []
comp_correct = []


def test_birds(path):
    files=os.listdir(path)
    files.remove('.DS_Store')
    for d in files:
        nested_file = "".join([str(path), "/", str(d)])
        files_2 = os.listdir(nested_file)
        for dd in files_2:
            if dd == '.DS_Store':
                pass
            else:
                picture_file = "".join([str(nested_file), "/", str(dd)])
                # print(str(picture_file))
                # spec = pygame.image.load(picture_file)
                true_ans.append(int(d[-1]))
                img = tf.keras.preprocessing.image.load_img(picture_file, target_size=(256, 256))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                predictions = loaded_model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                print('True class: ', d, 'Predicted_class: ', np.argmax(score))
                comp_ans.append(np.argmax(score))
                if comp_ans[-1]==true_ans[-1]:
                    comp_correct.append(1)
                else:
                    comp_correct.append(0)        
### Run the function                  
test_birds(test_data_direc) 
### Print the list of correct/incorrect data and the percentage correct.     
print('Computer Classification Results: ', sum(comp_correct), '/', len(comp_correct))     
print('Computer accuracy: ', sum(comp_correct)/len(comp_correct))


