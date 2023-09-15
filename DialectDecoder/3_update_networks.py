#%% Import packages
import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18
from train import update_cnn
from train import knn_classifier as knn


#%% Edit the section below to make your experiment

### Name of your experiment
experiment_name = 'experiment_test'
exp_round = '2'
### Experiment split (should add to 1)
op_iso_split = [0.8, 0.2]
### Train/Validation/Test split for CNN (should add to 1)
train_val_test_split = [0.7, 0.2, 0.1]
### Number of epochs for CNN
epochs = 10


#%% Directories
current_direc = os.getcwd()
exp_direc = current_direc + '/data/' + experiment_name
spect_direc = exp_direc + '/isolated/'

### CNN directories
old_state_dict_path = current_direc + '/CNN_networks/CNN_' + experiment_name +'.pth'
new_state_dict_path = current_direc + '/CNN_networks/CNN_' + experiment_name + '_r' + exp_round +'.pth'
anom_direc = exp_direc + '/isolated_labeled/'
anom_tvt_direc = exp_direc + '/isolated_labeled_tvt_split/'

### k-NN directories
save_direc = current_direc + '/kNN_networks/'
metadata = pd.read_csv(current_direc + '/metadata/2022_metadata.csv')
train_direc = current_direc + '/data/' + experiment_name + '/train'
csv_path = (current_direc + '/metadata/' + experiment_name + '/knn_open_metadata_' 
                + experiment_name + '_rd' + exp_round + '.csv')
knn_name = 'knn_' + experiment_name + '_rd' + exp_round


#%% Add in the newly labeled data
### csv containing newly labeled data, a list of bird classes, and a list of tvt proportions
labeled_anomalies = pd.read_csv(current_direc + '/output_files/' + experiment_name 
                                + '/classified_anomalies_' + experiment_name + '_rd' + exp_round + '.csv')
bird_classes = ["ABLA", "BATE", "BATW", "COMW", "FOFU", "FWSC", "LAME", "LODU", "RICH"]
train_val_test = [0.7, 0.2, 0.1]

### Make directories with the anomaly data
update_cnn.make_anom_direc(labeled_anomalies, spect_direc, anom_direc)

### Determine tvt split
update_cnn.create_tvt_iso_split(anom_direc, anom_tvt_direc)

### Add in labeled data into the open training set
update_cnn.combine_data(anom_tvt_direc, exp_direc)


#%% Retrain CNN
### Load the old CNN in
loaded_model = resnet18()
loaded_model.fc = nn.Linear(512, 9)
loaded_model.load_state_dict(torch.load(old_state_dict_path))
loaded_model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Retrain the CNN
update_cnn.retrain_cnn(loaded_model, device, epochs, exp_direc, new_state_dict_path, current_direc)


#%% Train k-nn model
### Make metadata for knn and pull necessary location
knn.make_knn_metadata(train_direc, metadata, csv_path, experiment_name, current_direc)
location_data = pd.read_csv(csv_path)

### Retrain k-nn
knn.make_knn(location_data, save_direc, knn_name)









