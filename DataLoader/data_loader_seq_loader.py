### Author - Chinmay Shah

''' This file defines a custom data loader to read from a csv file and we only load the index on initialization
and loads one sequence at a time. '''

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd 
from sklearn.preprocessing import StandardScaler


class dataLoader_seq_loader(Dataset):
    def __init__(self, df, seq_length, PAE_inputs=21, Predictor_inputs=None, predictor_seq_length=None):
        
        
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
        
        # # Normalize the data (column-wise)
        # self.mean = self.original_df.mean()
        # self.std = self.original_df.std()
        # self.std[self.std == 0] = 1
        
        # # Joint angle Data
        # # self.joint_angle_data = self.original_df.iloc[:,self.PAE_inputs:self.PAE_inputs + self.Predictor_inputs]
        # self.joint_angle_data = self.original_df
        # # Joint Angle Mean
        # self.joint_angle_mean = self.joint_angle_data.mean()
        # # Joint Angle Std
        # self.joint_angle_std = self.joint_angle_data.std()
        # self.joint_angle_std[self.joint_angle_std == 0] = 1
        
        # self.joint_angle_normalized = self.joint_angle_data - self.joint_angle_mean / self.joint_angle_std
        
        # self.normalized_df = (self.original_df - self.mean) / self.std
        
    def __len__(self):
        # return len(self.indices) - self.seq_length - self.predictor_seq_length 
        return len(self.indices) - self.seq_length
    
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
        
        # ######## Extract Predictor outputs
        # input_row_start_idx = self.indices[idx + self.seq_length - self.predictor_seq_length]
        # input_row_end_idx = self.indices[idx + self.seq_length]
        
        # prediction_row_idx = self.indices[idx + self.seq_length + 1]
        
        # # Extract all col with that sequence length of data
        # input_rows = self.joint_angle_normalized.iloc[input_row_start_idx:input_row_end_idx, 0:48].values
        # output_rows = self.joint_angle_normalized.iloc[prediction_row_idx, 48:54].values
        
        # # Inputs ( Transpose it to give cols, sequence length data)
        # Predictor_inputs = torch.tensor(input_rows , dtype=torch.float32).T
        # Predictor_outputs = torch.tensor(output_rows , dtype=torch.float32)
        

        # return PAE_inputs_centered, Predictor_inputs, Predictor_outputs
        return PAE_inputs_centered
        
        
        