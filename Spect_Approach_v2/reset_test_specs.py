import os
from functions import reset_test_set


# Define appropriate paths and number of specs desisred
current_direc = os.getcwd()
spec_direc = current_direc + '/specs_training'
isolate_direc = current_direc + '/specs_testing'


reset_test_set(spec_direc, isolate_direc)

