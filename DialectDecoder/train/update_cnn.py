### https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# License: BSD
# Author: Sasank Chilamkurthy
import os
from tqdm import tqdm
import pandas as pd
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import datetime
from train import CNN_classifier

cudnn.benchmark = True
plt.ion()   # interactive mode

#%% Directories
# current_direc = os.getcwd()
# exp_direc = current_direc + '/data/experiment_3'
# spect_direc = exp_direc + '/isolated/'
# old_state_dict_path = current_direc + '/CNN_networks/CNN_exp_3.pth'
# new_state_dict_path = current_direc + '/CNN_networks/CNN_exp_3_rd2.pth'
# anom_direc = exp_direc + '/isolated_labeled/'
# anom_tvt_direc = exp_direc + '/isolated_labeled_tvt_split/'

#%% Loads old CNN
# loaded_model = resnet18()
# loaded_model.fc = nn.Linear(512, 9)
# loaded_model.load_state_dict(torch.load(old_state_dict_path))
# loaded_model.eval()

### .csv labeled anomaly file
# labeled_anomalies = pd.read_csv(current_direc + '/output_files/exp3/classified_anomalies_exp3_rd1.csv')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# bird_classes = ["ABLA", "BATE", "BATW", "COMW", "FOFU", "FWSC", "LAME", "LODU", "RICH"]

#%% Function to make anomaly directory
def make_anom_direc(anom_data, spect_direc, anom_direc):
    for row_index, x in anom_data.iterrows():
        bird_class = os.path.basename(x[0])
        og_spec_file = spect_direc + bird_class + '/' + (x[1])[:-3] +'png'
        new_spec_file = anom_direc + bird_class + '/' + (x[1])[:-3] +'png'
        if os.path.exists(anom_direc + '/' + bird_class + '/'):
            shutil.copyfile(og_spec_file, new_spec_file)
        else:
            os.makedirs(anom_direc + '/' + bird_class + '/')
            shutil.copyfile(og_spec_file, new_spec_file)
            
# make_anom_direc(labeled_anomalies, spect_direc, anom_direc)

#%% Function to tvt split the isolated data
train_val_test = [0.7, 0.2, 0.1]
def create_tvt_iso_split(spec_dir, new_dir):
    train_dir = os.path.join(new_dir, 'train')
    val_dir = os.path.join(new_dir, 'val')
    test_dir = os.path.join(new_dir, 'test')
    for subdir, dirs, fls in os.walk(spec_dir):
        for idx, fl in tqdm(enumerate(fls)):
            old_path = os.path.join(subdir, fl)
            if idx <= train_val_test[0] * len(fls):
                os.makedirs(os.path.join(train_dir, os.path.basename(os.path.normpath(subdir))), exist_ok=True)
                new_path = os.path.join(train_dir, os.path.basename(os.path.normpath(subdir)), fl)
                os.rename(old_path, new_path)
            elif idx <= (train_val_test[0] + train_val_test[1]) * len(fls):
                os.makedirs(os.path.join(val_dir, os.path.basename(os.path.normpath(subdir))), exist_ok=True)
                new_path = os.path.join(val_dir, os.path.basename(os.path.normpath(subdir)), fl)
                os.rename(old_path, new_path)
            else:
                os.makedirs(os.path.join(test_dir, os.path.basename(os.path.normpath(subdir))), exist_ok=True)
                new_path = os.path.join(test_dir, os.path.basename(os.path.normpath(subdir)), fl)
                os.rename(old_path, new_path)
                
# create_tvt_iso_split(anom_direc, anom_tvt_direc)

#%% Function to add in new data to algorithm
def combine_data(anom_tvt_dir, exp_direc):
    for subdir, dirs, fls in os.walk(anom_tvt_dir):
        pp1, birdclass = os.path.split(subdir)
        pp2, tvt = os.path.split(pp1)
        parent_path, old_folder = os.path.split(pp2)
        for idx, fl in tqdm(enumerate(fls)):
            if tvt == 'train' or tvt == 'val' or tvt == 'test':
                old_path = os.path.join(subdir, fl)
                os.makedirs(os.path.join(exp_direc, tvt, birdclass), exist_ok=True)
                new_path = os.path.join(exp_direc, tvt, birdclass, fl)
                os.rename(old_path, new_path)
         
        
# combine_data(anom_tvt_direc) 

#%% Freeze the layers
def retrain_cnn(loaded_model, device, epochs, exp_direc, new_state_dict_path, current_direc):
    for param in loaded_model.parameters():
        param.requires_grad = False # freezes the layers
        
    num_ftrs = loaded_model.fc.in_features
    loaded_model.fc = nn.Linear(num_ftrs, 9)
    
    loaded_model = loaded_model.to(device) 
    
    train_dir = exp_direc + '/train'
    val_dir = exp_direc + '/val'
    
    print(datetime.datetime.now())
    new_model = CNN_classifier.fully_train_model(train_dir, val_dir, 25, new_state_dict_path, loaded_model, current_direc)
    print(datetime.datetime.now())
    return new_model

