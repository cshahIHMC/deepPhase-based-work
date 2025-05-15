### Author - Chinmay Shah

''' This file defines a custom data loader to read from a csv file and we only load the index on initialization
and loads one sequence at a time. '''

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd 
from sklearn.preprocessing import StandardScaler


class dataLoader_seq_loader(Dataset):
    def __init__(self, df, seq_length, PAE_inputs, Predictor_inputs, predictor_seq_length):
        
        
        # Save all the data
        self.original_df = df.copy()
        self.seq_length = seq_length
        self.predictor_seq_length = predictor_seq_length
        
        # PAE inputs
        self.PAE_inputs = PAE_inputs
        
        # Predictor inputs
        self.Predictor_inputs = Predictor_inputs
        
        # Store the indices
        self.indices = df.index.tolist()
        
        # Normalize the data (column-wise)
        self.mean = self.original_df.mean()
        self.std = self.original_df.std()
        self.std[self.std == 0] = 1
        
        self.normalized_df = (self.original_df - self.mean) / self.std
        
    def __len__(self):
        return len(self.indices) - self.seq_length - self.predictor_seq_length 
    
    def __getitem__(self,idx):
        
        
        
        ######## Extract PAE inputs
        row_start_idx = self.indices[idx]
        row_end_idx = self.indices[idx + self.seq_length]
        
        # Extract all col with that sequence length of data
        rows = self.original_df.iloc[row_start_idx:row_end_idx, 0:self.PAE_inputs].values
               
        # Inputs ( Transpose it to give cols, sequence length data)
        PAE_inputs = torch.tensor(rows, dtype=torch.float32).T
        
        # Window mean
        window_mean = PAE_inputs.mean(dim=0, keepdim=True)
        
        # Window mean centering
        PAE_inputs_centered = PAE_inputs - window_mean
        
        ######## Extract Predictor outputs
        input_row_start_idx = self.indices[idx + self.seq_length - self.predictor_seq_length]
        input_row_end_idx = self.indices[idx + self.seq_length]
        
        prediction_row_idx = self.indices[idx + self.seq_length + 1]
        
        # Extract all col with that sequence length of data
        input_rows = self.original_df.iloc[input_row_start_idx:input_row_end_idx, self.PAE_inputs:self.PAE_inputs + self.Predictor_inputs].values
        output_rows = self.original_df.iloc[prediction_row_idx, self.PAE_inputs:self.PAE_inputs + self.Predictor_inputs].values
        
        # Inputs ( Transpose it to give cols, sequence length data)
        Predictor_inputs = torch.tensor(input_rows, dtype=torch.float32).T
        Predictor_outputs = torch.tensor(output_rows, dtype=torch.float32)
        

        return PAE_inputs_centered, Predictor_inputs, Predictor_outputs
        
        
        