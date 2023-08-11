### https://realpython.com/knn-python/#use-knn-to-predict-the-age-of-sea-slugs
import os
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
from train import get_location


#%% Function to make the metadata for the knn
def make_knn_metadata(rootdir, metadata, csv_path, exp_name, current_direc):
    song_data = []
    for subdir, _, fls in os.walk(rootdir):
        for fl in fls:
            lat, long = get_location.get_loc(fl, metadata)
            if 'ABLA' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 0, lat, long])
            elif 'BATE' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 1, lat, long])
            elif 'BATW' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 2, lat, long])
            elif 'COMW' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 3, lat, long])
            elif 'FOFU' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 4, lat, long])
            elif 'FWSC' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 5, lat, long])
            elif 'LAME' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 6, lat, long])
            elif 'LODU' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 7, lat, long])
            elif 'RICH' in os.path.join(subdir,fl):
                song_data.append([subdir, fl, 8, lat, long])
            else:
                pass
    ### .csv file generation
    meta_dir = current_direc + '/metadata/' + exp_name
    if os.path.exists(meta_dir):
        pass
    else:
        os.mkdir(meta_dir)
    header = ['flock_year', 'file_name', 'bird_label', 'latitude', 'longitude']
    with open(csv_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
    # write the header
        writer.writerow(header)
    # write the data
        for a in song_data:
            writer.writerow(a)

#%% Function to make the kNN
def make_knn(loc_data, save_direc, knn_name):
    loc_data = loc_data.dropna(axis = 0)
    ### Make the dataset
    X = loc_data.drop(['bird_label','flock_year', 'file_name'], axis=1)
    X = X.values
    y = loc_data['bird_label']
    y = y.values
    ### Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.2,
                                                        random_state = 12345)
    ### Train kNN model
    knn_model = KNeighborsRegressor(n_neighbors=4)
    knn_model.fit(X_train, y_train)
    ### Save model to be used later
    test_predictions = knn_model.predict(X_test)
    mse = mean_squared_error(y_test, test_predictions)
    rmse = sqrt(mse)
    knnPickle = open(save_direc + knn_name, 'wb')
    # source, destination 
    pickle.dump(knn_model, knnPickle)  
    # close the file
    knnPickle.close()   
    ### load the model from disk
    loaded_model = pickle.load(open(save_direc + knn_name, 'rb'))
    result = loaded_model.predict(X_test) 
    # Plot classified location coordinates
    f, ax = plt.subplots()
    points = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, cmap='Set1')
    ax.grid(True)
    # Note, legend is currently broken.
    ax.legend(['0','1','2','3','4','5','6','7','8'])
    plt.show()












