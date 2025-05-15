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
from Models.FCNN import FCNN
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import torch
from Library import Plotting as plot
from Library.IMUMotionAnalyzer import IMUMotionAnalyzer


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
        "PAE_seq_length": 301,
        "PAE_inputs": 24,
        "PAE_outputs": 24,
        "PAE_phases": 10,
        "PAE_intermediate_channels": 16,
        "training_window": 2.0, # How many seconds of data you are reviewing
        "data_recorded_rate": 150, # 
        "FCNN_inputs": 6,
        "FCNN_outputs": 6,
        "FCNN_seq_length":12,
        "FCNN_num_hidden_layers": 4,
        "FCNN_num_hidden_neurons": 256,
        "FCNN_dropout": 0.4
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
    motion_analyzer.analyze()
    joint_angles_df = motion_analyzer.get_joint_angles()
    
    ## Add the joint angle cols to data columns
    df_combined = pd.concat([data,joint_angles_df], axis=1)
        
    # Split the pandas dataframe into a training and validation dataset
    # data_split = 179580 # 04_21_2025
    data_split_start = 80427 # 05_08_2025
    data_split_end = 95427
    
    training_range_of_data = list(range(0,data_split_start)) + list(range(data_split_end,len(data)))
    validation_range_of_data = list(range(data_split_start,data_split_end)) 
                              
    
    training_df = df_combined.iloc[training_range_of_data].reset_index(drop=True)
    validation_df = df_combined.iloc[validation_range_of_data].reset_index(drop=True)
    
    # Checking the Size of the data frame
    print("Training DF size:", training_df.shape)
    print("Validation DF size:", validation_df.shape)
    
    # Above data frames have the entire dataset
    # Get only gyro data
    
    extracted_training_df = utility.extract_data(training_df, joint_imu_map)
    extracted_validation_df = utility.extract_data(validation_df, joint_imu_map)
    
    ## Data preprocessing
    # The xsensor data is recorded in deg/sec while microstrain data is recorded in rad/sec.
    # To make everything consistent we convert the xsensor data to rad/sec
    # Convert all the angle columns to radians
    col_to_modify = ["R_insole_gyro_x", "R_insole_gyro_y" , "R_insole_gyro_z",
                     "L_insole_gyro_x", "L_insole_gyro_y" , "L_insole_gyro_z",
                     "thigh_r_angle_y", "thigh_l_angle_y", "shank_r_angle_y",
                     "shank_l_angle_y", "foot_r_angle_y", "foot_l_angle_y"]

    for col in col_to_modify:
        extracted_training_df[col] = extracted_training_df[col] * np.pi / 180
        extracted_validation_df[col] = extracted_validation_df[col] * np.pi / 180
        # extracted_testing_df[col] = extracted_testing_df[col] * np.pi / 180

    print("Extracted DF size:", extracted_training_df.shape)
    
    # Setup custom datasets
    training_dataset = dataLoader_seq_loader(extracted_training_df, config["PAE_seq_length"], config["PAE_inputs"], config["FCNN_inputs"], config["FCNN_seq_length"])
    validation_dataset = dataLoader_seq_loader(extracted_validation_df, config["PAE_seq_length"], config["PAE_inputs"], config["FCNN_inputs"], config["FCNN_seq_length"])
    
    
    return training_dataset, validation_dataset, extracted_training_df.columns

## Training Function
def train_model(model, config, training_dataloader, validation_dataloader, PAE_model_file, log_wandB=False):   
    
    
    ## Load PAE file
    weights = torch.load(PAE_model_file, weights_only=True)
    PAE_model = utility.ToDevice(PAE.Model(
                          input_channels=config["PAE_inputs"],
                          embedding_channels=config["PAE_phases"],
                          intermediate_channels=config["PAE_intermediate_channels"],
                          time_range=config["PAE_seq_length"],
                          window=config["training_window"]
                         ))
    
    PAE_model.load_state_dict(weights)
    
    ## Setting up an optimizer and a loss function - Original Paper used a AdamWr optimizer We using a simple SGD
    learning_rate = config["lr"]
    momentum = config["momentum"]

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    lossFn = nn.MSELoss()
    lossFn_no_reduction = nn.MSELoss(reduction='none')

    ## Training the periodic auto encoder
    print("Starting Training........")
    training_losses = []
    validation_losses = []
    individual_losses = []
    individual_test_losses = []
    testing_losses = []

    epochs = config["epochs"]
    
    ## Training Loop
    for epoch in range(epochs):
    
        running_loss = 0.0
    
        for batch in training_dataloader:
        
            PAE_inputs, FCNN_inputs, FCNN_outputs = batch
            # print("PAE inputs size: ", PAE_inputs.shape)
            # print("FNN inputs size: ", FCNN_inputs.shape)
            # print("FNN outputs size: ", FCNN_outputs.shape)
            
            PAE_inputs = utility.ToDevice(PAE_inputs)
            
            PAE_model.eval()
            _, _, _, params  = PAE_model(PAE_inputs)
            
            # Flattening the inputs for the motion prediction network
            flattened_inputs = utility.ToDevice(FCNN_inputs.reshape(FCNN_inputs.shape[0], -1))


            phaseInputs = torch.flatten(torch.stack(params, dim=1),1,2) 
            phaseInputs = phaseInputs[:,:,-1]
            
            FCNN_combine_inputs = torch.cat((flattened_inputs, phaseInputs), dim=1)
            
            # Forwards pass
            y_pred = model(FCNN_combine_inputs)
            
            # Calculate the loss
            # weightedLoss = weightedMSELossFunction(ypred, groundTruth, conditions)
            loss = lossFn(y_pred, utility.ToDevice(FCNN_outputs))

            # # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item() * FCNN_inputs.size(0)

                
        train_loss = running_loss / len(training_dataloader.dataset)
        training_losses.append(train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss}')
        
        val_loss, individual_loss = utility.cal_validation_loss_future_prediction(model, PAE_model, validation_dataloader, lossFn, lossFn_no_reduction)
        validation_losses.append(val_loss)
        individual_losses.append(individual_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
        
        if log_wandB:
            wandb.log({"train/train_loss": train_loss,
                        "train/epoch": epoch,
                        "val/val_loss": val_loss,
                        "val/epoch":epoch})
        
            
        return training_losses, validation_losses    

        
    
def main():
    
    # Logging Flag
    log_wandB = False
    
     # Prediction_Plotting_Slice
    training_prediction_start = 2531
    validation_prediction_start = 1990

    
    PAE_model_file = "/home/cshah/workspaces/deepPhase based work/Saved Models/20250514_1208_PAE - sensor suit Walking Data (132000) - seq-length-300, 10 Phases 24x16xembedded channels conv and mean centered.pth"
    
    # file_name = "trial"
    file_name = "FFNN - Future joint angle predictor"
    project_name = "PAE - Sensor suit Walking Data - With Future predictions + New Thigh IMU location"
    
    # Setup all the system paramters
    config, joint_imu_map, imu_joint_map = parameter_setup( file_name=file_name, project_name=project_name)
    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
        
    ## Data Setup
    data_path = "/home/cshah/workspaces/sensorsuit/logs/05_08_2025/05_08_2025_start_0_walk_test.csv"
        
    training_dataset, validation_dataset, col_names = setup_datasets( file_path=data_path, joint_imu_map=joint_imu_map, config=config)
    
    training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    
    # Taking a subset of the training and validation to plot a window / slice of data
    training_dataset_plotting = Subset(training_dataset, range(training_prediction_start, training_prediction_start+900))
    validation_dataset_plotting = Subset(validation_dataset, range(validation_prediction_start, validation_prediction_start+900))
    
    training_dataloader_plotting = DataLoader(training_dataset_plotting, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    validation_dataloader_plotting = DataLoader(validation_dataset_plotting, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    
    ## Load Model
    FCNN_inputs = config["FCNN_inputs"] * config["FCNN_seq_length"] + 4 * config["PAE_phases"]
    model = utility.ToDevice(FCNN(FCNN_inputs, 
                            config["FCNN_outputs"], 
                            config["FCNN_num_hidden_layers"],
                            config["FCNN_num_hidden_neurons"],
                            config["FCNN_dropout"]))
    
    # Train
    training_losses, validation_losses = train_model(model=model, config=config, training_dataloader=training_dataloader, 
                                                   validation_dataloader=validation_dataloader, PAE_model_file=PAE_model_file, log_wandB=log_wandB)
    
    # Save the Model
    model_save_location = "Saved Models/" + datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"] + ".pth"
    torch.save(model.state_dict(), model_save_location)
    
    # Plot files
    # Option to plot loss plot
    testing_losses = None
    plot_save_location = "Plots/" + config["training_tag"] 
    loss_plot_save_location = plot_save_location + "_loss_plot.png"
    utility.loss_plot(training_losses, validation_losses, testing_losses, loss_plot_save_location)
    
    return 0
    
    
    
    

if __name__ == "__main__":
    raise SystemExit(main())
    
