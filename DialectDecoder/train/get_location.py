import os
import csv
import pandas as pd

#%% A function to get location out of a csv file
def get_loc(file_name, metadata_csv):
    for row_index, x in metadata_csv.iterrows():
        if file_name[:-3] == str(x[1])[:-3]:
            lat = x[3]
            long = x[4]
        try:
            lat, long
        except NameError:
            lat = 'NA'
            long = 'NA'
    return lat, long
