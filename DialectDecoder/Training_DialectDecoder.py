#%% Import packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pickle
import csv
import pandas as pd
import pygame
import pygame.freetype
from train import button 
from train import CNN_classifier
import torch
from torchvision.models import resnet18
import torch.nn as nn
import datetime

#%% Edit the section below to make your experiment
### Note, make sure your experiment name here matches the one you used in train.py
### Name of your experiment
experiment_name = 'experiment_test'
### Round of DialectDecoder for that experiment
rd = 2 # use 1 with the first run of DialectDecoder and increase by 1 after that


#helper functions needed
def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

#%% Establish appropriate directories and load metadata.
current_direc = os.getcwd()
spect_direc = current_direc + '/data/cropped_spect_testt/'
audio_data_direc = current_direc + '/data/cut_songs/'
### CNN name
state_dict_path = current_direc + '/CNN_networks/CNN_' + experiment_name +'.pth'
### kNN name
knn = 'knn_' + experiment_name + '_rd' + str(rd-1)
knn_name = current_direc + '/kNN_networks/' + knn
### Metadata file name 
metadata = pd.read_csv(current_direc + '/metadata/' + experiment_name + '/isolated_metadata.csv')
iso_direc = current_direc + '/data/' + experiment_name +'/isolated/' # used to back transform path names
### Shuffles the array so that the classes get mixed up to better classify a wide range of songs.
metadata = metadata.sample(frac=1)
metadata_array = metadata.to_numpy()
### .csv file names to save to
anomaly_csv_name = current_direc + '/output_files/' + experiment_name + '/current_anomalies_' + experiment_name + '_rd' + str(rd) + '.csv'
classified_csv_name = current_direc + '/output_files/' + experiment_name + '/classified_anomalies_' + experiment_name + '_rd' + str(rd) + '.csv'
### Empty arrays and bird classes
anomaly_array = []
resolved_array = []
incorrect_array = []
bird_classes = ["ABLA", "BATE", "BATW", "COMW", "FOFU", "FWSC", "LAME", "LODU", "RICH"]

#%% Load in the CNN and the k-nearest neighbors models.

### k-NN model
location_model = pickle.load(open(knn_name, 'rb'))

### CNN model
model = resnet18()
model.fc = nn.Linear(512, 9)
model.load_state_dict(torch.load(state_dict_path))
model.eval()

#%% Construct the anomaly array

num_songs = len(metadata_array)

### Evaluate CNN and k-NN on each image
for row_index, x in enumerate(metadata_array[:-1]):
    if str(x[3]) == 'nan':
        pass
    else:
        bird_class = str(int(x[2]))
    ### Classify file based on spectrogram.
        spect_journey = str(x[0])
        spect_name = str(x[1])
        file_name = spect_journey +'/' + spect_name
        act_map, bird_label_spect_dec = CNN_classifier.apply_cam(model, file_name)
        bird_label_spect = str(int(bird_label_spect_dec))
    ### Classify file based on location. 
        bird_label_loc = str(int(location_model.predict([[x[3], x[4]]])[0]))
    ### Compare all three labels
        if bird_label_loc==bird_label_spect==bird_class:
            pass
    ### If CNN and kNN match but are wrong, record as incorrect.
        elif bird_label_loc==bird_label_spect!=bird_class:
            labels = [bird_label_spect, bird_label_loc]
            y = np.append(x, labels)
            incorrect_array.append(y)
    ### If they don't match, save as anomalous        
        else:
            labels = [bird_label_spect, bird_label_loc]
            y = np.append(x, labels)
            anomaly_array.append(y)
        print(row_index+1,'/',num_songs-2 ,'completed')
    
### Write .csv file with anomalies    
anom_dir = current_direc + '/output_files/' + experiment_name
if os.path.exists(anom_dir):
    pass
else:
    os.mkdir(anom_dir)
header = ['flock_path', 'file_name', 'bird_label', 'latitude', 'longitude', 'cnn_label', 'knn_label']
with open(anomaly_csv_name, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
### Write the header
    writer.writerow(header)
#### Write the data
    for a in anomaly_array:
        writer.writerow(a)
### anomaly_array = [path, file_name, true_class, lat, long, spect_label, loc_label]
print(anomaly_array)
### Tells you how many anomalies there are
print(np.shape(anomaly_array))
print(incorrect_array)
print(np.shape(incorrect_array))


#%% Game and screen initialization
pygame.init()

white = (255,255,255)
black = (0,0,0)
ltgrey = (207, 207, 207)
dkgrey = (79, 79, 79)

color_passive = black
color_text_active = ltgrey
color_box_active = dkgrey

new_text_color = color_passive
new_box_color = color_passive

existing_text_color = color_passive

start_text_color = color_text_active
start_box_color = color_box_active

disp_width=800
disp_height=900

pygame.display.set_caption('Training DialectDecoder')
window_surface = pygame.display.set_mode((disp_width,disp_height))

background = pygame.Surface((800, 900))
background.fill(pygame.Color('#000000'))

font = pygame.font.Font(None, 40)

mode = 0o666


#%% Game functions
### Function to load an anomalous bird song spectrogram to the game
def anomaly_bird(anomaly_matrix, i):
    spect_journey = (anomaly_matrix[i])[0]
    spect_name = str((anomaly_matrix[i])[1])
    file_name = spect_journey +'/' + spect_name
    cam_spec, bird_label = CNN_classifier.apply_cam(model, file_name, use_cam = True)
    spec = pygame.image.load(file_name)
    return(file_name, spec, cam_spec)

### Function to draw text
def draw_text(text, font, text_col, x, y):
  img = font.render(text, True, text_col)
  window_surface.blit(img, (x, y))
  
#%% Create Buttons
# Classify buttons
spect_button = button.Button(25, 575, 150, 50, ltgrey, 'CNN Label')
loc_button = button.Button(225, 575, 150, 50, ltgrey, 'k-NN Label')
existing_song_button = button.Button(425, 575, 150, 50, ltgrey, 'Other Label')
new_song_button = button.Button(625, 575, 150, 50, ltgrey, 'New Label')

# Play song button
play_song_button = button.Button(560, 250, 200, 50, ltgrey, 'Play Song')

# Verification buttons
add_to_birds_button = button.Button(550, 835, 180, 50, ltgrey, 'Classify Song')
existing_confirm_button = button.Button(350, 835, 180, 50, ltgrey, 'Confirm Class')

# Class activation button
activation_button = button.Button(540, 325, 250, 50, ltgrey, 'Show Activation Map')

# Start game button
start_button = button.Button(300, 400, 200, 50, ltgrey, 'Start Game')

#%% Initializes all necessary lists and pieces
### Initialize lists
spect_guess = []
loc_guess = []
human_label = []

### Generates first spectrogram that appears on launch
spectrogram = anomaly_bird(anomaly_array, 0)

### Displayes the corresponding location
text_loc = font.render('[' + str((anomaly_array[0])[3]) + ', ' + str((anomaly_array[0])[4]) + ']' ,1,ltgrey)

### Pulls up the appropriate labels for spect and loc

### Initial strings for labeling portion and the intial index
n=0
label_user_text = ''
type_user_text = ''
name_user_text = ''


#%% Create drop down list
num_of_classes = len(anomaly_array)
bird_class_list = button.DropDown([dkgrey, ltgrey],
                            [dkgrey, ltgrey],
                            100, 650, 200, 50, 
                            pygame.font.SysFont(None, 30), 
                            "Select Bird Class",
                            bird_classes)
scroll_index = 0


#%% Game states
# Game loop state
is_running = True
start = True
finished = False
name_active = False

# New song states
new_song = False
label_active = False
type_active = False
existing_pressed = False

# CAM states
show_cam = False


#%% Game Loop
while is_running:
#%% Welcome screen of game
    if start == True:
        ### Makes the start screen with the background, text, textbox, and buttons
        window_surface.blit(background,(0,0))
        draw_text("Welcome to DialectDecoder!", font, white, 50, 50)
        draw_text("Please enter your name/identifier below.", font, white, 50, 150)
        
        name_rect = pygame.Rect(200, 300, 400, 40)
        start_button.draw(window_surface)
        
        ### Possible actions
        event_list = pygame.event.get()
        for event in event_list:
        ### Allows you to close the game
            pos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                is_running = False
        ### Allows you to click in the textbox
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.isOver(pos):
                    start = False
                if name_rect.collidepoint(event.pos):
                    name_active = True
                else:
                    name_active = False
        ### Enables typing and storing text in textbox
            if event.type == pygame.KEYDOWN and name_active == True:
                ### Check for backspace
                if event.key == pygame.K_BACKSPACE:
                ### get text input from 0 to -1 i.e. end.
                    name_user_text = label_user_text[:-1]
        ### Unicode standard is used for string formation
                else:
                    name_user_text += event.unicode
        ### Draws and updates the surface
        pygame.draw.rect(window_surface, start_box_color, name_rect)
        name_text_surface = font.render(name_user_text, True, (255, 255, 255))
        window_surface.blit(name_text_surface, (name_rect.x+5, name_rect.y+5))
        name_rect.w = max(2, name_text_surface.get_width()+10)
                    
                
#%% End screen of game
    elif finished == True:
        window_surface.blit(background,(0,0))
        draw_text("You have classified all anomalies!", font, white, 50, 50)
        event_list = pygame.event.get()
    ### Allows you to close the game
        for event in event_list:
            pos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                is_running = False
                finished = True
                
#%% Classify portion of game                
    else:
    ### Base screen draw
        window_surface.blit(background,(0,0))
        ### Draw constant text
        draw_text("Anomaly Classification", font, white, 50, 25)
        draw_text("Spectrogram:", font, white, 50, 60)
        draw_text("Location:", font, white, 290, 60)
        draw_text("CNN label:", font, white, 50, 480)
        draw_text("k-NN label:", font, white, 500, 480)
        draw_text("Label Song", font, white, 315, 535)
        draw_text(str(str(n+1) + '/' + str(np.shape(anomaly_array)[0]) + " anomalies"), font, white, 560, 175)
        ### Load and display spectrogram
        if show_cam == False:
            spectrogram = pygame.image.load(spect_direc + str((anomaly_array[n])[0][len(iso_direc):])+ '/' + str((anomaly_array[n])[1][:-3]) + 'png')
        else:
            spectrogram = pygame.image.load(current_direc + '/temp_cam.png')
        window_surface.blit(spectrogram, (50, 100))
        ### Load and display guesses
        spect_bird_guess = (anomaly_array[n])[5]
        loc_bird_guess = (anomaly_array[n])[6]
        text_spect_class = font.render(bird_classes[int(spect_bird_guess)],1,ltgrey)
        text_loc_class = font.render(bird_classes[int(loc_bird_guess)],1,ltgrey)
        ### Draw buttons
        spect_button.draw(window_surface)
        loc_button.draw(window_surface)
        existing_song_button.draw(window_surface)
        new_song_button.draw(window_surface)
        play_song_button.draw(window_surface)
        activation_button.draw(window_surface)
    
    
    ### New song label screen draw
        label_rect = pygame.Rect(550, 700, 150, 40)
        type_rect = pygame.Rect(550, 775, 150, 40)
        bird_label = font.render('Enter class label (number): ',1, new_text_color)
        bird_type = font.render('Enter class name (BIRD): ',1, new_text_color)
        if label_active == True or type_active == True:
            add_to_birds_button.draw(window_surface)
    
    ### Existing song screen draw
        if existing_pressed == True:
            existing_confirm_button.draw(window_surface)
#%% Events
        event_list = pygame.event.get()
        for event in event_list:
            pos = pygame.mouse.get_pos()
        ### Allows you to close the game
            if event.type == pygame.QUIT:
                is_running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
#%% CNN and k-NN buttons
                if spect_button.isOver(pos) or loc_button.isOver(pos):
                    ### Labels song with the CNN label
                    if spect_button.isOver(pos):
                        spect_time = str(datetime.datetime.now())
                        spect_row = np.append(anomaly_array[n], (anomaly_array[n])[5])
                        spect_row = list(spect_row)
                        spect_row.extend([spect_time, name_user_text])
                        resolved_array.append(spect_row)
                    ### Labels song with k-NN label
                    if loc_button.isOver(pos):
                        loc_time = str(datetime.datetime.now())
                        loc_row = np.append(anomaly_array[n], (anomaly_array[n])[6])
                        loc_row = list(loc_row)
                        loc_row.extend([loc_time, name_user_text])
                        resolved_array.append(loc_row)
                    ### Ends game if all anomalies are classified
                    if n == len(anomaly_array)-1:
                        finished = True
                    ### Pulls up next anomalous image and resets screen
                    else:
                        n = n+1
                        show_cam = False
                        new_text_color = color_passive
                        new_box_color = color_passive
                    
#%% Play song button
                elif play_song_button.isOver(pos):
                    song_name = audio_data_direc + str(((anomaly_array[n])[0])[len(iso_direc):] +'/' + str((anomaly_array[n])[1])[:-3]) + 'wav'
                    if not os.path.exists(song_name):
                        # search for file as quick method does not work
                        song_name = find_file(str((anomaly_array[n])[1])[:-3] + 'wav',audio_data_direc)
                    song_time = pygame.mixer.Sound(song_name)
                    pygame.mixer.Sound.play(song_time)
                    
#%% Show CAM button
                elif activation_button.isOver(pos):
                    show_cam = not show_cam
                    if show_cam == True:
                        file_name_n, spec_n, cam_spec = anomaly_bird(anomaly_array, n)
                        cam_spec.save('temp_cam.png', 'PNG')

#%% New song button events
                if new_song_button.isOver(pos):
                    ### Displays buttons associated with new songs
                    new_song = True
                    ### Makes all appropriate text and text boxes appear
                    new_text_color = color_text_active
                    new_box_color = color_box_active
                    window_surface.blit(bird_label, (300, 600))
                    window_surface.blit(bird_type, (300, 600))
                    existing_pressed = False
                    ### Hides text from the existing class option
                    existing_text_color = color_passive 
                ### Activate each text box for typing
                if label_rect.collidepoint(event.pos):
                    label_active = True
                else:
                    label_active = False
                if type_rect.collidepoint(event.pos):
                    type_active = True
                else:
                    type_active = False
                if add_to_birds_button.isOver(pos):
                ### Add to bird class list
                    bird_classes.append(type_user_text)
                ### Add data to resolved_array
                    new_time = str(datetime.datetime.now())
                    new_row = np.append(anomaly_array[n], label_user_text)
                    new_row = list(new_row)
                    new_row.extend([new_time, name_user_text])
                    resolved_array.append(new_row)
                ### Game updates (reset for next song or done)
                    label_active = False
                    type_active = False
                    new_text_color = color_passive
                    new_box_color = color_passive
                    label_user_text = ''
                    type_user_text = ''
                    if n == len(anomaly_array)-1:
                        finished = True
                    else:
                        n = n+1
                        show_cam = False
                        
#%% Existing song button events
                if existing_song_button.isOver(pos):
                    existing_pressed = True
                ### Hide all text/boxes for new songs
                    new_text_color = color_passive
                    new_box_color = color_passive
                    existing_text_color = color_text_active
            ### Confirm class choice from drop down menu        
                if existing_confirm_button.isOver(pos):
                    bcl_main = str(bird_class_list.main)
                    bci = str(bird_classes.index(bcl_main))
            ### Add data to resolved_array
                    existing_time = str(datetime.datetime.now())
                    existing_row = np.append(anomaly_array[n], bci)
                    existing_row = list(existing_row)
                    existing_row.extend([existing_time, name_user_text])
                    resolved_array.append(existing_row)
            ### Move to next anomaly
                    if n == len(anomaly_array)-1:
                        finished = True
                    else:
                        n = n+1
                        show_cam = False
                    existing_pressed = False
                    bird_class_list.main = "Select Bird Class"
                    scroll_index = 0
#%% New song label event
            if event.type == pygame.KEYDOWN and label_active == True:
                # Check for backspace
                if event.key == pygame.K_BACKSPACE:
                    # get text input from 0 to -1 i.e. end.
                    label_user_text = label_user_text[:-1]
                    # Unicode standard is used for string formation
                else:
                    label_user_text += event.unicode
#%% New song type events
            if event.type == pygame.KEYDOWN and type_active == True:
                # Check for backspace
                if event.key == pygame.K_BACKSPACE:
                    # get text input from 0 to -1 i.e. end.
                    type_user_text = type_user_text[:-1]
            # Unicode standard is used for string formation
                else:
                    type_user_text += event.unicode
    
#%% Draw new song label items
        pygame.draw.rect(window_surface, new_box_color, label_rect)
        label_text_surface = font.render(label_user_text, True, (255, 255, 255))
        window_surface.blit(label_text_surface, (label_rect.x+5, label_rect.y+5))
        label_rect.w = max(2, label_text_surface.get_width()+10)
        window_surface.blit(bird_label, (100, 700))
#%% Draw new song type items
        pygame.draw.rect(window_surface, new_box_color, type_rect)   
        type_text_surface = font.render(type_user_text, True, (255, 255, 255))
        window_surface.blit(type_text_surface, (type_rect.x+5, type_rect.y+5))
        type_rect.w = max(2, type_text_surface.get_width()+10)
        window_surface.blit(bird_type, (100, 775))
#%% Draw existing song items
        if existing_pressed == True:
            bird_class_list.options = bird_classes
            bird_class_list.draw(window_surface, scroll_index)
            selected_option = bird_class_list.update(event_list, scroll_index)
            scroll_index = selected_option[1]
            if selected_option[0] >= 0:
                bird_class_list.main = bird_class_list.options[selected_option[0]+scroll_index]
        window_surface.blit(text_spect_class, (225, 480))  
        window_surface.blit(text_loc_class, (675, 480)) 
        text_loc = font.render('[' + str((anomaly_array[n])[3]) + ', ' + str((anomaly_array[n])[4]) + ']' ,1,ltgrey)
        window_surface.blit(text_loc, (425, 60)) 
          
                
#%% Update the display
    pygame.display.update()
    
if finished == True:
    for i, j in enumerate(resolved_array):
        print(resolved_array[i])
        
### Write .csv file with anomalies    
header2 = ['flock_path', 'file_name', 'bird_label', 'latitude', 'longitude', 'cnn_label', 'knn_label', 'human_label', 'date/time', 'labeler']
with open(classified_csv_name, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
### Write the header
    writer.writerow(header2)
#### Write the data
    for b in resolved_array:
        writer.writerow(b)


