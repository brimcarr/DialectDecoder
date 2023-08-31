import os
import shutil
from tqdm import tqdm

#%% Function to split data into 2 different folders
# Moves cropped spectrograms to open/isolated folders
def create_experiment_split(spec_dir, new_dir, train_test):
    print("Entering create_experiment_split")
    print((spec_dir, new_dir, train_test))
    train_dir = os.path.join(new_dir, 'open')
    test_dir = os.path.join(new_dir, 'isolated')
    print((train_dir,test_dir))
    for subdir, _, fls in os.walk(spec_dir):
        for idx, fl in tqdm(enumerate(fls)):
            old_path = os.path.join(subdir, fl)
            if idx <= train_test[0] * len(fls):
                os.makedirs(os.path.join(train_dir, os.path.basename(os.path.normpath(subdir))), exist_ok=True)
                new_path = os.path.join(train_dir, os.path.basename(os.path.normpath(subdir)), fl)
                shutil.copyfile(old_path, new_path)
            else:
                os.makedirs(os.path.join(test_dir, os.path.basename(os.path.normpath(subdir))), exist_ok=True)
                new_path = os.path.join(test_dir, os.path.basename(os.path.normpath(subdir)), fl)
                shutil.copyfile(old_path, new_path)
  

#%% Function to create train, val, test datasets
### Moves cropped spectrograms to new directory and splits them into training, 
### validation, and testing folders
def create_train_val_test_split(spec_dir, new_dir, train_val_test):
    print("Entering create_train_val_test_split")
    print((spec_dir, new_dir, train_val_test))
    train_dir = os.path.join(new_dir, 'train')
    val_dir = os.path.join(new_dir, 'val')
    test_dir = os.path.join(new_dir, 'test')
    print((train_dir,val_dir,test_dir))
    for subdir, _, fls in os.walk(spec_dir):
        for idx, fl in tqdm(enumerate(fls)):
            old_path = os.path.join(subdir, fl)
            if idx <= train_val_test[0] * len(fls):
                os.makedirs(os.path.join(train_dir, os.path.basename(os.path.normpath(subdir))), exist_ok=True)
                new_path = os.path.join(train_dir, os.path.basename(os.path.normpath(subdir)), fl)
                shutil.copyfile(old_path, new_path)
            elif idx <= (train_val_test[0] + train_val_test[1]) * len(fls):
                os.makedirs(os.path.join(val_dir, os.path.basename(os.path.normpath(subdir))), exist_ok=True)
                new_path = os.path.join(val_dir, os.path.basename(os.path.normpath(subdir)), fl)
                shutil.copyfile(old_path, new_path)
            else:
                os.makedirs(os.path.join(test_dir, os.path.basename(os.path.normpath(subdir))), exist_ok=True)
                new_path = os.path.join(test_dir, os.path.basename(os.path.normpath(subdir)), fl)
                shutil.copyfile(old_path, new_path)
                
        



             
                
                
                
                
                
                