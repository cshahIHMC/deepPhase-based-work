### Author - Chinmay Shah

''' This file defines a custom data loader to read from a csv file and we only load the index on initialization
and loads one sequence at a time. '''

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd 
from sklearn.preprocessing import StandardScaler


class dataLoader_seq_loader(Dataset):
    def __init__(self, df, seq_length):
        
        
        # Save all the data
        self.original_df = df.copy()
        self.seq_length = seq_length
        
        # Store the indices
        self.indices = df.index.tolist()
        
        # Normalize the data (column-wise)
        self.mean = self.original_df.mean()
        self.std = self.original_df.std()
        self.std[self.std == 0] = 1
        
        self.normalized_df = (self.original_df - self.mean) / self.std
        
    def __len__(self):
        return len(self.indices) - self.seq_length
    
    def __getitem__(self,idx):
        
        row_start_idx = self.indices[idx]
        row_end_idx = self.indices[idx + self.seq_length]
        
        # Extract all col with that sequence length of data
        rows = self.original_df.iloc[row_start_idx:row_end_idx, :].values
               
        # Inputs ( Transpose it to give cols, sequence length data)
        inputs = torch.tensor(rows, dtype=torch.float32).T
        
        # Window mean
        window_mean = inputs.mean(dim=0, keepdim=True)
        
        # Window mean centering
        inputs_centered = inputs - window_mean

        return inputs_centered
        
        
        