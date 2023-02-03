import os
from functions import crop_specs


#%% Set the directories for the data and a place for the spectrograms to go
current_direc = os.getcwd()
spec_direc = current_direc + '/test_spectrograms'
cropped_spec_direc = current_direc + '/crop_test'


#%% Runs the function. All of your cropped specs should appear in your 
### cropped_spec_direc 

crop_specs(spec_direc, cropped_spec_direc)


