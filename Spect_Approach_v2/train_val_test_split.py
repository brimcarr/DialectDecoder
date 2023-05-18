import os
from tqdm import tqdm


current_direc = os.getcwd()
spect_dir = current_direc + '/data/cropped_spect_data'
tvt_dir = current_direc + '/train_val_test'

train_val_test = [0.7, 0.2, 0.1]
# percentages of train, val, and test data


# Moves cropped spectrograms to new directory and splits them into training, validation, and testing folders
def create_train_val_test_split(spec_dir, new_dir):
    train_dir = os.path.join(new_dir, 'train')
    val_dir = os.path.join(new_dir, 'val')
    test_dir = os.path.join(new_dir, 'test')
    for subdir, _, fls in os.walk(spec_dir):
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



create_train_val_test_split(spect_dir, tvt_dir)