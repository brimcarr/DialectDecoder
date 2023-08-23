import os
from functions import make_iso_specs


# Define appropriate paths and number of specs desisred
current_direc = os.getcwd()
spec_direc = current_direc + '/trial_3_class_data'
isolate_direc = current_direc + '/trial_training_data'
num_of_specs = 270

# Run the function
make_iso_specs(spec_direc, isolate_direc, num_of_specs)

