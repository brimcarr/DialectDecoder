import os
from functions import reset_test_set


# Define appropriate paths and number of specs desisred
current_direc = os.getcwd()
spec_direc = current_direc + '/trial_3_class_data'
isolate_direc = current_direc + '/trial_training_data'


reset_test_set(spec_direc, isolate_direc)

