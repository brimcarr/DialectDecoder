#%% Packages
import os
import pandas as pd
import datetime
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from train import split_data as sd
from train import CNN_classifier as cnn
from train import get_location
from train import knn_classifier as knn

#%% Edit the section below to make your experiment

### Name of your experiment
experiment_name = 'experiment_test'
rd = 1 # This will always be the first file you run for each experiment, so round will always be 1.
### Experiment split (should add to 1)
op_iso_split = [0.8, 0.2]
### Train/Validation/Test split for CNN (should add to 1)
train_val_test_split = [0.7, 0.2, 0.1]

#%% Directories
current_direc = os.getcwd()
### For sd
# Where the cropped spectrograms live
spect_direc = current_direc + '/data/cropped_spect_testt'
# Name of folder where you want to store the experiment's data
exp_direc = current_direc + '/data/' + experiment_name
# Name of the folder where you want to draw the CNN tvt data from
etvt_direc = current_direc + '/data/' + experiment_name + '/open'

### For cnn
state_dict_path = current_direc + '/CNN_networks/CNN_' + experiment_name +'.pth'
train_direc = current_direc + '/data/' + experiment_name + '/train'
val_direc = current_direc + '/data/' + experiment_name + '/val'
test_direc = current_direc + '/data/' + experiment_name + '/test'

### For knn
save_direc = current_direc + '/kNN_networks/'
metadata = pd.read_csv(current_direc + '/metadata/2022_metadata.csv')
csv_path = current_direc + '/metadata/' + experiment_name + '/knn_open_metadata_' + experiment_name + '.csv'
# Network name
knn_name = 'knn_' + experiment_name + '_rd' + str(rd)

### For isolated data
iso_csv_path = current_direc + '/metadata/' + experiment_name + '/isolated_metadata.csv'
iso_direc = current_direc + '/data/' + experiment_name +'/isolated/'

#%% Split data into open/isolated experiment folders
sd.create_experiment_split(spect_direc, exp_direc, op_iso_split)

#%% Split data into train/validation/test folders for CNN training
sd.create_train_val_test_split(etvt_direc, exp_direc, train_val_test_split)

#%% Train CNN model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = cnn.nn.Linear(512, 9)
model.eval()
print(datetime.datetime.now())
cnn.fully_train_model(train_direc, val_direc, 25, state_dict_path, model, current_direc)
print(datetime.datetime.now())

#%% Train knn model
knn.make_knn_metadata(train_direc, metadata, csv_path, experiment_name, current_direc)

location_data = pd.read_csv(csv_path)
knn.make_knn(location_data, save_direc, knn_name)

#%% Make the isolated metadata for DialectDecoder
knn.make_knn_metadata(iso_direc, metadata, iso_csv_path, experiment_name, current_direc)








