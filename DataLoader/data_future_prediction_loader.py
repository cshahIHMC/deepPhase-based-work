### Author - Chinmay Shah

''' This file defines a custom data loader to read from a csv file and we only load the index on initialization
and loads one sequence at a time. '''

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd 
from sklearn.preprocessing import StandardScaler


class dataLoader_future_prediction_loader(Dataset):
    def __init__(self, gyro_df, quat_df, gyro_seq_length=301, predictor_seq_length=1):
        
        
        # Save all the data
        self.original_gyro_df = gyro_df.copy()
        self.original_quat_df = quat_df.copy()
        self.gyro_seq_length = gyro_seq_length
        self.predictor_seq_length = predictor_seq_length
        
        # Store the indices
        self.indices = gyro_df.index.tolist()
        
        # If trianing directy from raw ss to joint angles
        self.quat_input_df = self.original_quat_df.iloc[:, 28:56]
        self.quat_output_df = self.original_quat_df.iloc[:, 0:28]
        
        
        # If training on ss-captury model output

        
    def __len__(self):
        # return len(self.indices) - self.seq_length - self.predictor_seq_length 
        return len(self.indices) - self.gyro_seq_length
    
    def __getitem__(self,idx):
        
        
        
        ######## Extract PAE inputs
        row_start_idx = self.indices[idx]
        row_end_idx = self.indices[idx + self.gyro_seq_length]
        
        # Extract all col with that sequence length of data
        rows = self.original_gyro_df.iloc[row_start_idx:row_end_idx, :].values
               
        # Inputs ( Transpose it to give cols, sequence length data)
        PAE_inputs = torch.tensor(rows, dtype=torch.float32).T
        
        # Window mean
        window_mean = PAE_inputs.mean(dim=0, keepdim=True)
        
        # Window mean centering
        PAE_inputs_centered = PAE_inputs - window_mean
        
        # ######## Extract Predictor outputs
        input_row_start_idx = self.indices[idx + self.gyro_seq_length - self.predictor_seq_length - 150]
        input_row_end_idx = self.indices[idx + self.gyro_seq_length - 150 ]
        
        prediction_row_idx = self.indices[idx + self.gyro_seq_length - 150]
        
        # Extract all col with that sequence length of data
        input_rows = self.quat_input_df.iloc[input_row_start_idx:input_row_end_idx, :].values
        output_rows = self.quat_output_df.iloc[prediction_row_idx, :].values
        
        # Inputs ( Transpose it to give cols, sequence length data)
        Predictor_inputs = torch.tensor(input_rows , dtype=torch.float32).T
        Predictor_outputs = torch.tensor(output_rows , dtype=torch.float32)
        

        return PAE_inputs_centered, Predictor_inputs, Predictor_outputs
        
        
        