import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
######################################################
## This File Loading the data for training sessions ##
######################################################
class FocalStackDataset(Dataset):
    def __init__(self, path):
        self.paths = self.get_all_paths(path)
        self.path = path

    def load_pickle(self, idx):

        with open(os.path.join(self.path, idx), 'rb') as file:
            data = pickle.load(file)
        return data

    def get_all_paths(self, path):
        return os.listdir(path)
            
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        item_data = self.load_pickle(self.paths[idx])

        if 'zero' in self.paths[idx]:
            target = torch.zeros_like(torch.from_numpy(item_data['input'])).float()
        else:
            target = torch.from_numpy((item_data['target'])).float()

        return torch.from_numpy(item_data['input']).float(), target
    

# if __name__ == '__main__':
#     # dataset  = FocalStackDataset('dataset')
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     dataset = FocalStackDataset(r'd:\training_data\Layer_330\upper_part\2x2x3')

#     print(dataset .__getitem__(4)[0].shape)