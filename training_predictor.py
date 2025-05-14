######### Author - Chinmay Shah #################


## Imports
import wandb
from Library import utility
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DataLoader.data_loader_seq_loader import dataLoader_seq_loader
from torch.utils.data import Dataset, DataLoader, Subset
from Models import PAE
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import torch
from Library import Plotting as plot
from Library import IMUMotionAnalyzer


## Setup all the parameters
def parameter_setup(file_name, project_name):
    config = {
        "training_tag": file_name,
        "project_name": project_name,
        "epochs": 1,
        "batch_size": 32,
        "num_workers": 8,
        "momentum":0.9,
        "lr": 1e-4,
        "dropout": 0.0,
        "dataset": "IHMC Senorsuit",
        "seq_length": 301,
        "PAE_inputs": 24,
        "PAE_outputs": 24,
        "PAE_phases": 10,
        "PAE_intermediate_channels": 16,
        "training_window": 2.0, # How many seconds of data you are reviewing
        "data_recorded_rate": 150, # 
        "FFNN_inputs": 1,
        "FFNN_outputs": 6,
        "FFNN_num_hidden_layers": 4,
        "FFNN_num_hidden_neurons": 256
    }
    
    joint_imu_map = {
    "back": "imu3",
    "pelvis": "imu2",
    "thigh_l": "imu1",
    "thigh_r": "imu5", 
    "shank_l": "imu4",
    "shank_r": "imu6",
    "foot_l": "L_insole",
    "foot_r": "R_insole"
    }

    imu_joint_map = {
        "imu3": "back",
        "imu2": "pelvis",
        "imu1": "thigh_l",
        "imu5": "thigh_r", 
        "imu4": "shank_l",
        "imu6": "shank_r",
        "L_insole": "foot_l",
        "R_insole": "foot_r"
    }
    
    return config, joint_imu_map, imu_joint_map
 
## Data Setup - Reads the csv files and add them to datasets
def setup_datasets(file_path, joint_imu_map, config):
    
    # file path to the data
    csv_path = file_path

    # Read the csv file into a pandas data frame
    data = pd.read_csv(csv_path)
    
    ## Calculate the joint angles and get it as a dataframe
    motion_analyzer = IMUMotionAnalyzer(csv_path)
    joint_angles = motion_analyzer.analyze()
    
    ## Add the joint angle cols to data columns
    
    
    # # Split the pandas dataframe into a training and validation dataset
    # # data_split = 179580 # 04_21_2025
    # data_split_start = 80427 # 05_08_2025
    # data_split_end = 95427
    
    # training_range_of_data = list(range(0,data_split_start)) + list(range(data_split_end,len(data)))
    # validation_range_of_data = list(range(data_split_start,data_split_end)) 
                              
    
    # training_df = data.iloc[training_range_of_data].reset_index(drop=True)
    # validation_df = data.iloc[validation_range_of_data].reset_index(drop=True)
    
    # # Checking the Size of the data frame
    # print("Gyro Training DF size:", training_df.shape)
    # print("Gyro Validation DF size:", validation_df.shape)
    # print("Joint Angle Training DF size:", training_df.shape)
    # print("Joint Angle Validation DF size:", validation_df.shape)
    
    # # Above data frames have the entire dataset
    # # Get only gyro data
    
    # extracted_training_df = utility.extract_data(training_df, joint_imu_map)
    # extracted_validation_df = utility.extract_data(validation_df, joint_imu_map)
    
    # ## Data preprocessing
    # # The xsensor data is recorded in deg/sec while microstrain data is recorded in rad/sec.
    # # To make everything consistent we convert the xsensor data to rad/sec
    # col_to_modify = ["R_insole_gyro_x", "R_insole_gyro_y" , "R_insole_gyro_z",
    #                  "L_insole_gyro_x", "L_insole_gyro_y" , "L_insole_gyro_z"]

    # for col in col_to_modify:
    #     extracted_training_df[col] = extracted_training_df[col] * np.pi / 180
    #     extracted_validation_df[col] = extracted_validation_df[col] * np.pi / 180
    #     # extracted_testing_df[col] = extracted_testing_df[col] * np.pi / 180

    # print("Extracted DF size:", extracted_training_df.shape)
    
    # # Setup custom datasets
    # training_dataset = dataLoader_seq_loader(extracted_training_df, config["seq_length"])
    # validation_dataset = dataLoader_seq_loader(extracted_validation_df, config["seq_length"])
    
    
    # return training_dataset, validation_dataset, extracted_training_df.columns
    
    
def main():
    
    # Logging Flag
    log_wandB = False

    
    PAE_model_file = "/home/cshah/workspaces/deepPhase based work/Saved Models/20250514_1208_PAE - sensor suit Walking Data (132000) - seq-length-300, 10 Phases 24x16xembedded channels conv and mean centered.pth"
    
    file_name = "trial"
    # file_name = "FFNN - Future joint angle predictor"
    project_name = "PAE - Sensor suit Walking Data - With Future predictions + New Thigh IMU location"
    
    # Setup all the system paramters
    config, joint_imu_map, imu_joint_map = parameter_setup( file_name=file_name, project_name=project_name)
    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
        
    ## Data Setup
    data_path = "/home/cshah/workspaces/deepPhase based work/Data/05_08_2025_walking_data.csv"
    
    training_dataset, validation_dataset, col_names = setup_datasets( file_path=data_path, joint_imu_map=joint_imu_map, config=config)
    