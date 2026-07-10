### Author - Chinmay Shah

''' This file defines a custom data loader to read from a csv file and we only load the index on initialization
and loads one sequence at a time. '''

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.signal import butter, filtfilt

def highpass_filter(data, cutoff=0.5, fs=100.0, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

def quaternion_to_sincos(flat_quat):
    """
    Input: flat_quat of shape (28,) -> 7 quaternions
    Output: sin-cos encoded angles, shape (42,)
    """
    quat_reshaped = flat_quat.reshape(7, 4)  # shape: 7x4
    eulers = R.from_quat(quat_reshaped).as_euler('xyz', degrees=False)  # shape: 7x3
    
    # eulers_filt = highpass_filter(eulers, cutoff=0.5, fs=150.0)
    
    sin = np.sin(eulers)
    cos = np.cos(eulers)
    
    sincos = np.concatenate([sin, cos], axis=1)  # shape: 7x6
    return sincos.flatten()  # shape: 42,

class dataLoader_simple_loader(Dataset):
    def __init__(self, df, gyro_df, seq_length, inputs, outputs):
    
    
        # Save all the data
        self.original_df = df.copy()
        self.gyro_original_df = gyro_df.copy()
        self.seq_length = seq_length

        # inputs
        self.inputs = inputs

        # outputs
        self.outputs = outputs
        
        # Input data
   
        self.input_quat_data = self.original_df.iloc[:, 0:28]
        # self.input_acc_gyro_data = self.original_df.iloc[:, 56:98]
        
        # # Col wise mean
        # acc_gyro_mean = self.input_acc_gyro_data.mean()
        # acc_gyro_std = self.input_acc_gyro_data.std()
        # acc_gyro_std[acc_gyro_std == 0] = 1e-8
        
        # self.input_acc_gyro_data_norm =  (self.input_acc_gyro_data - acc_gyro_mean) / acc_gyro_std
        # self.input_acc_gyro_data_norm = self.input_acc_gyro_data_norm.fillna(0)
        
        # self.input_data = pd.concat([
        #                     self.input_quat_data,
        #                     self.input_acc_gyro_data_norm
        #                     ], axis=1)
        
        self.input_data = self.input_quat_data
        
        # Gyro data
        self.gyro_input_data = self.gyro_original_df.iloc[:,0:42]
        
        # Normalize the output data
        self.gyro_input_data_mean = self.gyro_input_data.mean()
        self.gyro_input_data_std = self.gyro_input_data.std()
        self.gyro_input_data_std[self.gyro_input_data_std == 0] = 1
        
        self.normalized_gyro_input_data = (self.gyro_input_data - self.gyro_input_data_mean) / self.gyro_input_data_std
        
        # Output data
        self.output_data = self.original_df.iloc[:, 28:]
        
        # Normalize the output data
        self.output_data_mean = self.output_data.mean()
        self.output_data_std = self.output_data.std()
        self.output_data_std[self.output_data_std == 0] = 1
        
        self.normalized_output_data= (self.output_data - self.output_data_mean) / self.output_data_std

        
    def __len__(self):
        return len(self.input_data) - self.seq_length
    
    def __getitem__(self,idx):
        
        
        input_quat_rows = self.input_data.iloc[idx:idx+self.seq_length, :].values
        gyro_input_rows = self.normalized_gyro_input_data.iloc[idx:idx+self.seq_length, :].values
        
        input_rows = np.hstack((input_quat_rows, gyro_input_rows))

        
        output_rows = self.normalized_output_data.iloc[idx+self.seq_length-1, :].values
        
        # Apply quaternion → sin-cos transform across each timestep
        # input_transformed = np.array([quaternion_to_sincos(row) for row in input_rows])   # shape: (seq_len, 42)
        # output_transformed = np.array([quaternion_to_sincos(row) for row in output_rows])
        
        
        input_window = torch.tensor(input_rows, dtype=torch.float32)
        output_window = torch.tensor(output_rows,  dtype=torch.float32)
        
        # print(input_window.shape)
        
        # input_window = input_window.transpose(0,1)      
        # input_window = input_window.reshape(1, self.seq_length * self.inputs)
        input_window = input_window.reshape(1, self.seq_length * 70)
        output_window = output_window.unsqueeze(0)

        
        return input_window, output_window
        