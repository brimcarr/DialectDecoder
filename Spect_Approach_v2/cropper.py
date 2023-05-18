import os
from PIL import Image
from tqdm import tqdm


# Set the directories for the data and a place for the spectrograms to go
current_direc = os.getcwd()
spec_direc = current_direc + '/data/spect_data'
cropped_spec_direc = current_direc + '/data/cropped_spect_data'

def crop_specs(spec_direc, cropped_spec_direc):
    for subdir, _, fls in os.walk(spec_direc):
        for fl in tqdm(fls):
            spec = Image.open(os.path.join(subdir, fl))
            cropped_spec = spec.crop((80, 60, 560, 425))
            bird_type = os.path.basename(os.path.normpath(subdir))
            new_loc = os.path.join(cropped_spec_direc, bird_type)
            os.makedirs(new_loc, exist_ok=True)
            cropped_spec.save(os.path.join(new_loc, fl))





# Runs the function. All of your cropped specs should appear in your
# cropped_spec_direc

crop_specs(spec_direc, cropped_spec_direc)