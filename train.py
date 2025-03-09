import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from CNN3D.cnn_model import Simple3DCNN
from dataloader import FocalStackDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
#from torchsummary import summary

from tqdm import tqdm
import torch
import torch.nn as nn
import os
import re

os.environ['KMP_DUPLICATE_LIB_OK']='True'

##################################
## This File for model training ##
##################################

from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.optimizer.param_groups]
        else:
            # Return base learning rate after warmup
            return [self.base_lr for _ in self.optimizer.param_groups]
        
class WeightedMSELoss(nn.Module):
    def __init__(self, weight_zero=1.0, weight_nonzero=1.0):
        super(WeightedMSELoss, self).__init__()
        self.weight_zero = weight_zero
        self.weight_nonzero = weight_nonzero
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, input, target):
        # Calculate the base MSE loss
        loss = self.mse_loss(input, target)

        # Create a mask where the target is zero
        zero_mask = (target == 0).float()

        # Apply different weights to zero and non-zero regions
        loss = loss * (zero_mask * self.weight_zero + (1 - zero_mask) * self.weight_nonzero)

        # Return the mean loss
        return loss.mean()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        Args:
            patience (int): How long to wait after last time the validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.1):
        super(CustomLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for decreasing errors

    def forward(self, model_output, target_value):

        xx = self.custom_loss(model_output.squeeze(1) , target_value[:, 0, 0, 0])

        return xx

    def custom_loss(self, output, target):
        mse_loss = nn.MSELoss()(output, target)
        return mse_loss

# Training Loop
def train_model(model, dataloader, validate_loader, criterion, optimizer, scheduler, warmup, num_epochs):
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=3, delta=0.0001)
    mse_results_train = []
    mse_results_val = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Wrap the dataloader with tqdm to show progress for each batch
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        model.train()

        for i, (inputs, targets) in enumerate(progress_bar):
            optimizer.zero_grad()

            outputs = model(inputs.unsqueeze(0).permute(1, 0, 2, 3, 4).cuda())
            loss = criterion(outputs, targets.unsqueeze(0).permute(1, 0, 2, 3, 4).cuda()[:, :, 0, :, :])

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(dataloader)
        mse_results_train.append(epoch_loss)

        #####################
        # Validation Loader #
        #####################

        model.eval()
        running_loss = 0.0

        # Wrap the dataloader with tqdm to show progress for each batch
        progress_bar = tqdm(validate_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (inputs, targets) in enumerate(progress_bar):
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(0).permute(1, 0, 2, 3, 4).cuda())

            loss = criterion(outputs, targets.unsqueeze(0).permute(1, 0, 2, 3, 4).cuda()[:, :, 0, :, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_val_loss = running_loss / len(validate_loader)
        mse_results_val.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        # Check early stopping condition
        early_stopping(epoch_val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.8f}, Validate Loss: {epoch_val_loss:.8f}")
    return mse_results_train, mse_results_val



def natural_key(string):
    # Use regex to split the string into numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string)]

if __name__ == '__main__':
    # Hyperparameters and settings
    batch_size = 128
    PATH_DIR = 'data/train'
    folders = os.listdir(PATH_DIR)
    folders = sorted(folders, key=natural_key)

    with open('layers_data.txt', "r") as file:
        axils = file.readlines()

    # Data Preparation
    for index, folder in enumerate(folders[324:]):
        print(folder)
        
        num_epochs = 100

        # Data Preparation
        print('##################### Loading Data ##############################')

        dataset = FocalStackDataset(os.path.join(PATH_DIR, folder))

        # Example: Split dataset into 80% training and 20% validation
        train_size = int(0.80 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validate_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

        print('##################### Training #####################')
        # Model, Loss, Optimizer
        model = Simple3DCNN(in_channelss=int(axils[index].split()[-1])).cuda()
        
        criterion = CustomLoss()
        warmup_steps = 1000  # Number of warmup steps
        base_lr = 0.0001  # Base learning rate after warmup
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-5)
        warmup = WarmupScheduler(optimizer, warmup_steps, base_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.0001, patience=3, verbose=True)

        # Start Training
        mse_results_training, mse_results_validate = train_model(model, train_dataloader, validate_dataloader, criterion, optimizer, scheduler, warmup ,num_epochs)

        model_save_path = 'checkpoint'
        torch.save(model.state_dict(), os.path.join(model_save_path, folder+'.pth'))
        print("Model saved as 3d_unet_model.pth")
        
