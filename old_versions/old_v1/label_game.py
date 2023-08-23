#%% Import packages
import pygame
import pygame.freetype
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from button import Button, DropDown 
import shutil
from functions import load_cnn, img_to_cnn_input


#%% Game and screen initialization
pygame.init()

white = (255,255,255)
black = (0,0,0)
ltgrey = (207, 207, 207)
dkgrey = (79, 79, 79)

color_passive = black
new_text_color_active = ltgrey
new_box_color_active = dkgrey
existing_text_color_active = ltgrey

new_text_color = color_passive
new_box_color = color_passive
existing_text_color = color_passive

disp_width=800
disp_height=850

pygame.display.set_caption('Bird Classifier')
window_surface = pygame.display.set_mode((disp_width,disp_height))

background = pygame.Surface((800, 850))
background.fill(pygame.Color('#000000'))

font = pygame.font.Font(None, 40)

mode = 0o666

### Directories
current_direc = os.getcwd()
cropped_direc = '/trial_3_class_data'
training_direc = '/trial_training_data'
song_direc = 'test_cut_songs_sorted'

### Path to cropped spectrograms
pathy = current_direc + cropped_direc
### Path to add in new songs
path_for_training = current_direc + training_direc
### Path to play song
path_song = current_direc[-14] + song_direc

### Name of the CNN you want to use for classification
cnn_name = 'model'

### Number of anomalies you want to identify
num_of_anom = 1 # (Do one less than desired)

#%% Game functions
### Function to load a random bird song spectrogram to the game
def random_bird(path):
    files=os.listdir(path)
    files.remove('.DS_Store')
    d=random.choice(files)
    nested_file = "".join([str(path), "/", str(d)])
    files_2 = os.listdir(nested_file)
    dd = random.choice(files_2)
    picture_file = "".join([str(nested_file), "/", str(dd)])
    spec = pygame.image.load(picture_file)
    print(d)
    return(d, dd, spec)

### Draw text
def draw_text(text, font, text_col, x, y):
  img = font.render(text, True, text_col)
  window_surface.blit(img, (x, y))

#%% Loads the CNN from file and creates the model to be used for classification

loaded_model = load_cnn(cnn_name)

#%% Create Buttons
# Label game buttons
abla_button = Button(275, 660, 100, 50, ltgrey, 'ABLA')
comw_button = Button(400, 660, 100, 50, ltgrey, 'COMW')
anomaly_button = Button(550, 660, 150, 50, ltgrey, 'Anomaly')
show_comp_button = Button(100, 510, 200, 50, ltgrey, 'Show Comp. Ans.')
play_song_button = Button(500, 510, 200, 50, ltgrey, 'Play Song')

# Anomoly label buttons
new_song_button = Button(100, 500, 150, 50, ltgrey, 'New Song')
existing_song_button = Button(300, 500, 170, 50, ltgrey, 'Existing Song')
true_anomaly_button = Button(520, 500, 150, 50, ltgrey, 'Anomaly')
add_to_birds_button = Button(550, 750, 180, 50, ltgrey, 'Classify Song')
existing_confirm_button = Button(350, 750, 180, 50, ltgrey, 'Confirm Class')

#%% Initializes all necessary lists and pieces
### Initialize lists
human_ans = []
comp_ans = []
correct_ans = []
human_correct = []
comp_correct = []
anomaly_list = []

### List of classes the computer can classify as
bird_classes = ['ABLA', 'COMW', 'Anomaly']

### Generates first spectrogram that appears on launch
d, dd, spectrogram = random_bird(pathy)


### Classifies the spectrogram using img_to_cnn_input from functions.py
bird_guess, prob = img_to_cnn_input(pathy+'/'+d+'/'+dd, loaded_model)
comp_ans.append(bird_guess)

### Appends whether the guess was wrong or right to list
if comp_ans[-1] == int(d[-1]):
    comp_correct.append(1)
else:
    comp_correct.append(0)

### Displays the initial guess for the initial spectrogram and the comp's accuracy
text_comp_class = font.render(bird_classes[bird_guess],1,white)
text_comp_accuracy = font.render(str(round(sum(comp_correct)/len(comp_correct),3)),1,white)
text_prob = font.render(str(round(prob,3)),1,white)
text_human_accuracy = font.render('---',1,white)

### Initial strings for labeling portion
n=0
label_user_text = ''
type_user_text = ''

#%% Drop down list
bird_class_list = DropDown([dkgrey, ltgrey],
                            [dkgrey, ltgrey],
                            100, 625, 200, 50, 
                            pygame.font.SysFont(None, 30), 
                            "Select Bird Class", bird_classes[:-1])


#%% Game states

# Game loop state
is_running = True
# Toggle state for showing computer answer
show_comp_guess_state = True
# Label portion state
name_songs = False
# New song states
new_song = False
label_active = False
type_active = False
existing_pressed = False
finished = False
#%% Game Loop
while is_running:
#%% Done portion of game
    if finished == True:
        window_surface.blit(background,(0,0))
        draw_text("You have classified all marked anomalies!", font, white, 50, 50)
        event_list = pygame.event.get()
        for event in event_list:
            pos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                is_running = False
#%% Naming portion of game
    elif name_songs == True:
    ### Base screen draw
        window_surface.blit(background,(0,0))
        draw_text("Time to label birds!", font, white, 50, 50)
        spectrogram = pygame.image.load(pathy + '/' + (anomaly_list[n])[0] + '/' + (anomaly_list[n])[1])
        window_surface.blit(spectrogram, (150, 100))
        new_song_button.draw(window_surface)
        existing_song_button.draw(window_surface)
        true_anomaly_button.draw(window_surface)


    ### New song label screen draw
        label_rect = pygame.Rect(550, 600, 150, 40)
        type_rect = pygame.Rect(550, 675, 150, 40)
        bird_label = font.render('Enter class label (number): ',1, new_text_color)
        bird_type = font.render('Enter class name (BIRD): ',1, new_text_color)
        if label_active == True or type_active == True:
            add_to_birds_button.draw(window_surface)
    
    ### Existing song screen draw
        existing_song_text = font.render('Choose class then hit confirm ',1, existing_text_color)
        if existing_pressed == True:
            existing_confirm_button.draw(window_surface)
    #%% Events
        event_list = pygame.event.get()
        for event in event_list:
            pos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                is_running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
    ### New song mouse events
                if new_song_button.isOver(pos):
                    new_text_color = new_text_color_active
                    new_box_color = new_box_color_active
                    window_surface.blit(bird_label, (300, 600))
                    window_surface.blit(bird_type, (300, 600))
                    existing_pressed = False
                    existing_text_color = color_passive
                if label_rect.collidepoint(event.pos):
                    label_active = True
                else:
                    label_active = False
                if type_rect.collidepoint(event.pos):
                    type_active = True
                else:
                    type_active = False
                if add_to_birds_button.isOver(pos):
            ### Make new folder for the new bird type
                    new_folder=('class_'+label_user_text)
                    path = os.path.join(path_for_training +'/'+ new_folder)
                    if os.path.isdir(path) == False:
                        os.mkdir(path, mode)
            ### Copy in spectrogram to new folder
                    current_spec_path = pathy + '/' + (anomaly_list[n])[0] + '/' + (anomaly_list[n])[1]
                    new_spec_path = path + '/' + (anomaly_list[n])[1]
                    shutil.copy(current_spec_path, new_spec_path)
            ### Add to bird class list
                    bird_classes.insert(-1, type_user_text)
                    print(bird_classes)
            ### Game updates
                    label_active = False
                    type_active = False
                    new_text_color = color_passive
                    new_box_color = color_passive
                    label_user_text = ''
                    type_user_text = ''
                    n = n+1
                    print(n)
                    if n == len(anomaly_list):
                        finished = True
                    else:
                        spectrogram = pygame.image.load(pathy + '/' + (anomaly_list[n])[0] + '/' + (anomaly_list[n])[1])
                    
                    
                      
    ### Existing song mouse events
                if existing_song_button.isOver(pos):
                    existing_pressed = True
                    new_text_color = color_passive
                    new_box_color = color_passive
                    existing_text_color = existing_text_color_active
            ### Confirm class choice        
                if existing_confirm_button.isOver(pos):
                    bcl_main = str(bird_class_list.main)
                    bci = str(bird_classes.index(bcl_main))
                    existing_spec_path = pathy + '/' + (anomaly_list[n])[0] + '/' + (anomaly_list[n])[1]
                    new_exist_spec_path = path_for_training + '/class_' + bci + '/' + (anomaly_list[n])[1]
                    shutil.copy(existing_spec_path, new_exist_spec_path)
                    n = n+1
                    print(n)
                    if n == len(anomaly_list):
                        finished = True
                    else:
                        spectrogram = pygame.image.load(pathy + '/' + (anomaly_list[n])[0] + '/' + (anomaly_list[n])[1])
    ### Anomaly mouse events
                if true_anomaly_button.isOver(pos):
                    new_text_color = color_passive
                    new_box_color = color_passive
                    print(3)
    ### New song label event
            if event.type == pygame.KEYDOWN and label_active == True:
                # Check for backspace
                if event.key == pygame.K_BACKSPACE:
                    # get text input from 0 to -1 i.e. end.
                    label_user_text = label_user_text[:-1]
            # Unicode standard is used for string formation
                else:
                    label_user_text += event.unicode
    ### New song type events
            if event.type == pygame.KEYDOWN and type_active == True:
                # Check for backspace
                if event.key == pygame.K_BACKSPACE:
                    # get text input from 0 to -1 i.e. end.
                    type_user_text = type_user_text[:-1]
            # Unicode standard is used for string formation
                else:
                    type_user_text += event.unicode

    ### Draw new song label items
        window_surface.blit(spectrogram, (150, 100))
        pygame.draw.rect(window_surface, new_box_color, label_rect)
        label_text_surface = font.render(label_user_text, True, (255, 255, 255))
        window_surface.blit(label_text_surface, (label_rect.x+5, label_rect.y+5))
        label_rect.w = max(2, label_text_surface.get_width()+10)
        window_surface.blit(bird_label, (100, 600))
    ### Draw new song type items
        pygame.draw.rect(window_surface, new_box_color, type_rect)   
        type_text_surface = font.render(type_user_text, True, (255, 255, 255))
        window_surface.blit(type_text_surface, (type_rect.x+5, type_rect.y+5))
        type_rect.w = max(2, type_text_surface.get_width()+10)
        window_surface.blit(bird_type, (100, 675))
    ### Draw existing song items
        window_surface.blit(existing_song_text, (100, 575))
        if existing_pressed == True:
            bird_class_list.options = bird_classes[:-1]
            bird_class_list.draw(window_surface)
            selected_option = bird_class_list.update(event_list)
            if selected_option >= 0:
                    bird_class_list.main = bird_class_list.options[selected_option]
        
        
#%% Label portion of game
    else:

        window_surface.blit(background,(0,0))
        # text_human_accuracy = font.render('---',1,white)
        draw_text("Comp. guess:", font, ltgrey, 75, 600)
        draw_text("Comp. confidence:", font, ltgrey, 425, 600)
        draw_text("Your guess:", font, ltgrey, 100, 675)
        draw_text("Comp. accuracy:", font, ltgrey, 425, 750)
        draw_text("Your accuracy:", font, ltgrey, 75, 750)
        window_surface.blit(spectrogram, (150, 100))
        abla_button.draw(window_surface)
        comw_button.draw(window_surface)
        anomaly_button.draw(window_surface)
        show_comp_button.draw(window_surface)
        play_song_button.draw(window_surface)
     
    
        #%% Events
        for event in pygame.event.get():
            pos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                is_running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
            # Record human's answer
                if abla_button.isOver(pos) or comw_button.isOver(pos) or anomaly_button.isOver(pos):
                    if abla_button.isOver(pos):
                        human_ans.append(0)
                        print(human_ans)
                    if comw_button.isOver(pos):
                        human_ans.append(1)
                        print(human_ans)
                    if anomaly_button.isOver(pos):
                        human_ans.append(2)
                        print(human_ans)
                    correct_ans.append(int(d[-1]))
                        
                    # Record if human is correct
                    if human_ans[-1]==correct_ans[-1]:
                        human_correct.append(1)
                        print('human correct? ', human_correct)
                    else:
                        human_correct.append(0)
                        print('human correct? ', human_correct)
                    if human_ans[-1]==correct_ans[-1] and human_ans[-1]==2:
                        anomaly_list.append([d,dd])
                    # Calculate and display human's accuracy
                    text_human_accuracy = font.render(str(round(sum(human_correct)/len(human_correct),3)),1,white)
    
                    # Generate a new random spectrogram
                    d, dd, spectrogram = random_bird(pathy)
    
                    # Computer classifies new spectrogram
                    bird_guess, prob = img_to_cnn_input(pathy+'/'+d+'/'+dd, loaded_model)
                    comp_ans.append(bird_guess)
                    # Record if computer is correct
                    if comp_ans[-1] == int(d[-1]):
                        comp_correct.append(1)
                    else:
                        comp_correct.append(0)
                    # Render the computer's guess and the computer's accuracy
                    text_comp_class = font.render(bird_classes[bird_guess],1,white)
                    text_comp_accuracy = font.render(str(round(sum(comp_correct)/len(comp_correct),3)),1,white)
                    text_prob = font.render(str(round(prob,3)),1,white)
                elif show_comp_button.isOver(pos):
                    show_comp_guess_state = not show_comp_guess_state
                elif play_song_button.isOver(pos):
                    song_time = pygame.mixer.Sound(path_song+'/'+d+'/'+dd[:-3]+'wav')
                    pygame.mixer.Sound.play(song_time)
                # file_name_text = ["File name: " + d + "/" + dd]
                # print(file_name_text)
                if show_comp_guess_state == False:
                    text_comp_class = font.render("---",1,white)
                else:
                    text_comp_class = font.render(bird_classes[bird_guess],1,white)
                if len(anomaly_list) > num_of_anom:
                      name_songs=True
    
        # Update the screen
        window_surface.blit(text_human_accuracy, (300, 750))   
        window_surface.blit(text_comp_class, (275, 600))
        window_surface.blit(text_comp_accuracy, (675, 750))
        window_surface.blit(text_prob, (700, 600))

#%% Update the display
    pygame.display.update()
