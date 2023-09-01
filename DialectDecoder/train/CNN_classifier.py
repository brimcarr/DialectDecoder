import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn as nn
import os
from tqdm import tqdm
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import datetime

#%% Converts a PIL Image to a torch tensor of shape INPUT_SIZE
INPUT_SIZE = (224, 224)
BATCH_SIZE = 64

img_transform = Compose([ToTensor(), Resize(INPUT_SIZE, antialias=True)])
bird_labels = ["ABLA", "BATE", "BATW", "COMW", "FOFU", "FWSC", "LAME", "LODU", "RICH"]


#%% Creates a pytorch dataset object from a list of (PIL image, label) pairs
class BirdData(Dataset):
    def __init__(self, data_label_pairs):
        super().__init__()
        self.transform = Compose([ToTensor(), Resize(INPUT_SIZE, antialias=True)])
        self.data_label_pairs = data_label_pairs

    def __len__(self):
        return len(self.data_label_pairs)

    def __getitem__(self, item):
        return self.transform(self.data_label_pairs[item][0]), self.data_label_pairs[item][1]


#%% Creates training and validation dataloaders given a path to a single directory containing spectrograms
# (directory is assumed to contain folders like 'ABLA_2020' ect.)
def create_dataloaders(data_dir, percent_train):
    train_data_label_pairs, val_data_label_pairs = [], []
    for root, _, files in tqdm(os.walk(data_dir)):
        for idx, spec_file in enumerate(files):
            if not spec_file.startswith('.'):
                spec = Image.open(os.path.join(root, spec_file)).convert('RGB')
                bird_str = os.path.basename(os.path.normpath(root))[:4]
                bird_type = bird_labels.index(bird_str)
                if idx / len(files) < percent_train:
                    train_data_label_pairs.append((spec, bird_type))
                else:
                    val_data_label_pairs.append((spec, bird_type))
            else:
                pass

    train_dataset = BirdData(train_data_label_pairs)
    val_dataset = BirdData(val_data_label_pairs)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, val_dataloader


#%% Creates a single dataloader given a path to a single directory containing spectrograms
# (directory is assumed to contain folders like 'ABLA_2020' ect.)
def create_dataloader(data_dir):
    print(data_dir)
    data_label_pairs = []
    for root, _, files in tqdm(os.walk(data_dir)):
        for idx, spec_file in enumerate(files):
            if not spec_file.startswith('.'):
                spec = Image.open(os.path.join(root, spec_file)).convert('RGB')
                bird_str = os.path.basename(os.path.normpath(root))[:4]
                bird_type = bird_labels.index(bird_str)
                data_label_pairs.append((spec, bird_type))
            else:
                pass
    dataset = BirdData(data_label_pairs)
    print(len(dataset))
    m_len = min(4,len(dataset))
    print(data_label_pairs[0:m_len])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader


#%% Trains the model for one epoch
def train(model, data_loader, optimizer, criterion):
    model.train()
    loss_sum, correct_sum, num_samples = 0, 0, 0
    # num_batches = len(data_loader)

    for idx, i in enumerate(data_loader):
        sample, label = i
        label = label.to('cpu')
        output = model(sample).to('cpu')
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.cpu().detach() * float(label.size(0))
        pred_y = torch.max(output, 1)[1].squeeze()
        correct_sum += (pred_y == label).sum().item()
        num_samples += float(label.size(0))

        # if idx % int(num_batches / 4) == 0:
        #     print(f'Batch loss: {loss.item()} --- Batch number: [{idx}/{num_batches}]')

    loss = loss_sum / num_samples
    accuracy = correct_sum / num_samples
    return loss, accuracy


#%% Tests the model on the given dataloader
def model_test(model, dataloader, criterion):
    model.eval()
    loss_sum, correct_sum, num_samples = 0, 0, 0

    for i in dataloader:
        sample, label = i
        label = label.to('cpu')
        output = model(sample).to('cpu')
        loss_sum += (criterion(output, label).cpu().detach() * float(label.size(0)))
        pred_y = torch.max(output, 1)[1].squeeze()
        correct_sum += (pred_y == label).sum().item()
        num_samples += float(label.size(0))

    loss = loss_sum / num_samples
    accuracy = correct_sum / num_samples
    print(f'Validation Loss: {loss}')
    print(f'Validation Accuracy: {accuracy * 100} %')
    return loss, accuracy


#%% Loads png_file to PIL image, converts it to a tensor, and feeds it to model
# Output is an integer --- can be used as index of the 'bird_labels' list to get a string like 'ABLA'
def png_file_into_model(model, png_file):
    img = Image.open(png_file).convert('RGB')
    output = model(img_transform(img).unsqueeze(0))

    return output.squeeze(0).argmax().item()


#%% Loads png file, feeds it into model, and computes the class activation map
# Output is a PIL image of the original image with the CAM overlayed on top, along with
# the models prediction (as an integer)
# If you want the CAM separate, you can have the function return the 'activation_map' instead
# The default is to not use the CAM, as it's slow. If you want the CAM, change use_cam to True
def apply_cam(model, png_file, use_cam=False):

    cam_extractor = SmoothGradCAMpp(model, target_layer='layer4')

    img = Image.open(png_file).convert('RGB')
    output = model(img_transform(img).unsqueeze(0))
    model_prediction = output.squeeze(0).argmax().item()
    
    if use_cam:
        activation_map = cam_extractor(model_prediction, output)
        result = overlay_mask(img, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.65)
    else:
        result = img
        
    return result, model_prediction


#%% Helper function to get png file path from a directory containing the spectrograms which is
# structured like the original directory for the raw audio files
def get_spec_path(data_dir, img_idx=(1, 1)):

    bird_dir = sorted(os.listdir(data_dir))[img_idx[0]+1]
    bird_file = os.listdir(os.path.join(data_dir, bird_dir))[img_idx[1]]
    img_path = os.path.join(data_dir, bird_dir, bird_file)

    return img_path


#%% Function to plot losses and accuracies, or save to a png file
def plot_loss_and_acc(train_losses, train_accs, val_losses, val_accs, current_direc, model_name='', save_path=None):
    # Plots the loss and accuracy over time
    plt.figure(figsize=(15, 5))

    plt_loss = plt.subplot(121)
    plt_loss.plot(train_losses)
    plt_loss.plot(val_losses)
    plt.title(model_name)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper right")

    plt_accuracy = plt.subplot(122)
    plt_accuracy.plot(train_accs)
    plt_accuracy.plot(val_accs)
    plt.title(model_name)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="lower right")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


#%% Trains the model for the given number of epochs, given a directory containing the training data,
# and a directory containing the validation data
# Saves the models parameters to 'state_dict_path' each time the validation accuracy is an all-time high
def fully_train_model(train_dir, val_dir, num_epochs, state_dict_path, model, current_direc):

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_dl, val_dl = create_dataloader(train_dir), create_dataloader(val_dir)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_acc = 0

    for _ in range(num_epochs):

        train_loss, train_acc = train(model, train_dl, optimizer, loss_fn)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = model_test(model, val_dl, loss_fn)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), state_dict_path)

    fig_path = current_direc + '/output_files/exp0/loss_acc_fig_exp_0.png'
    os.makedirs(os.path.dirname(fig_path),exist_ok=True)
    plot_loss_and_acc(train_losses, train_accs, val_losses, val_accs, current_direc=current_direc, model_name='ResNet18_exp0', save_path=fig_path)

    return train_losses, train_accs, val_losses, val_accs



#%% Train the model (takes around 1 hour 30 min on my laptop to train for 25 epochs)
# But only around 15 epochs seem necessary
### Uncomment the 6 lines below to run model by itself! Recomment when done.

# model = resnet18(weights=ResNet18_Weights.DEFAULT)
# model.fc = nn.Linear(512, 9)
# model.eval()
# print(datetime.datetime.now())
# fully_train_model(train_dir, val_dir, 25, state_dict_path, model)
# print(datetime.datetime.now())

#%% Initialize the model and load the model's parameters
# model = resnet18()
# model.fc = nn.Linear(512, 9)
# model.load_state_dict(torch.load(state_dict_path))
# model.eval()

#%% Test the model on the hold-out dataset
# loss_fn = nn.CrossEntropyLoss()
# test_dl = create_dataloader(test_dir)
# model_test(model, test_dl, loss_fn)

#%% Produce an example CAM
# anom_direc = current_direc + '/data/cropped_spect_anom/LODU_anom/'
# img_path = anom_direc + '/presldD_13bRS.png'
# result, pred = apply_cam(model, img_path)
# print(img_path)
# print(f'Model\'s prediction is: {bird_labels[pred]}')
# result.show()