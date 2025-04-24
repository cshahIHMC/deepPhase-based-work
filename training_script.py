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

## Setup all the parameters

def parameter_setup(file_name, project_name):
    config = {
        "training_tag": file_name,
        "project_name": project_name,
        "epochs": 25,
        "batch_size": 32,
        "num_workers": 8,
        "momentum":0.9,
        "lr": 1e-4,
        "dropout": 0.0,
        "dataset": "IHMC Senorsuit",
        "seq_length": 301,
        "inputs": 24,
        "outputs": 24,
        "phases": 8,
        "intermediate_channels": 16,
        "training_window": 2.0, # How many seconds of data you are reviewing
        "data_recorded_rate": 150 # 
    }
    
    joint_imu_map = {
    "back": "imu3",
    "pelvis": "imu2",
    "thigh_l": "imu1",
    "thigh_r": "imu6", 
    "shank_l": "imu4",
    "shank_r": "imu5",
    "foot_l": "L_insole",
    "foot_r": "R_insole"
    }

    imu_joint_map = {
        "imu3": "back",
        "imu2": "pelvis",
        "imu1": "thigh_l",
        "imu6": "thigh_r", 
        "imu4": "shank_l",
        "imu5": "shank_r",
        "L_insole": "foot_l",
        "R_insole": "foot_r"
    }
    
    return config, joint_imu_map, imu_joint_map
    

## Data Setup - Reads the csv files and add them to datasets
def setup_datasets(file_path, joint_imu_map, config):
    
    # file path to the data
    csv_path = "/home/cshah/workspaces/deepPhase based work/Data/04_21_2025_walking_data.csv"
    csv_path = file_path

    # Read the csv file into a pandas data frame
    data = pd.read_csv(csv_path)
    
    
    # Split the pandas dataframe into a training and validation dataset
    training_df = data.iloc[0:179580].reset_index(drop=True)
    validation_df = data.iloc[179580:].reset_index(drop=True)
    
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
    col_to_modify = ["R_insole_gyro_x", "R_insole_gyro_y" , "R_insole_gyro_z",
                     "L_insole_gyro_x", "L_insole_gyro_y" , "L_insole_gyro_z"]

    for col in col_to_modify:
        extracted_training_df[col] = extracted_training_df[col] * np.pi / 180
        extracted_validation_df[col] = extracted_validation_df[col] * np.pi / 180
        # extracted_testing_df[col] = extracted_testing_df[col] * np.pi / 180

    print("Extracted DF size:", extracted_training_df.shape)
    
    # Setup custom datasets
    training_dataset = dataLoader_seq_loader(extracted_training_df, config["seq_length"])
    validation_dataset = dataLoader_seq_loader(extracted_validation_df, config["seq_length"])
    
    
    return training_dataset, validation_dataset, extracted_training_df.columns
    
## Training Function
def train_model(model, config, training_dataloader, validation_dataloader, log_wandB=False):
    
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
        
            PAE_inputs = batch
            PAE_inputs = utility.ToDevice(PAE_inputs)
        
            # Zero the parameter gradients
            optimizer.zero_grad()
        
            # Forward
            outputs, latent, signal, params  = model(PAE_inputs)
        
            # Flattening the outputs and inputs and calculating the loss
            flattened_inputs = PAE_inputs.reshape(PAE_inputs.shape[0], -1)
            flattened_outputs = outputs.reshape(outputs.shape[0], -1)
        
            # Calculate Loss
            loss = lossFn(flattened_inputs, flattened_outputs)
        
            # Backward
            loss.backward()
            optimizer.step()
        
            # Calculate running loss
            running_loss += loss.item() * PAE_inputs.size(0)
        
        train_loss = running_loss / len(training_dataloader.dataset)
        training_losses.append(train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss}')
    
    
        val_loss, individual_loss = utility.cal_validation_loss(model, validation_dataloader, lossFn, lossFn_no_reduction)
        validation_losses.append(val_loss)
        individual_losses.append(individual_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
    
        # test_loss, individual_test_loss = cal_validation_loss(model, testing_dataloader, lossFn, lossFn_no_reduction)
        # testing_losses.append(test_loss)
        # individual_test_losses.append(individual_test_loss)
        # print(f'Epoch [{epoch+1}/{epochs}], Testing Loss: {test_loss}')

        if log_wandB:
            wandb.log({"train/train_loss": train_loss,
                        "train/epoch": epoch,
                        "val/val_loss": val_loss,
                        "val/epoch":epoch})
                        # "test/test_loss": test_loss,
                        # "test/epoch":epoch}) 

    print('Finished Training')
    
    return training_losses, validation_losses


def plot_model_predictions(training_dataloader, validation_dataloader, model_file, imu_joint_map, config, col_names, folder_name):
    
    weights = torch.load(model_file, weights_only=True)
    loaded_model = PAE.Model(
                          input_channels=config["inputs"],
                          embedding_channels=config["phases"],
                          intermediate_channels=config["intermediate_channels"],
                          time_range=config["seq_length"],
                          window=config["training_window"]
                         )
    loaded_model.load_state_dict(weights)
    
    utility.plot_predictions(training_dataloader, validation_dataloader, loaded_model, imu_joint_map, folder_name, col_names)

def main():
   
    # Logging Flag
    log_wandB = True
    # file_name = "trial"
    file_name = "PAE - sensor suit Walking Data - seq-length-300, 8 Phases 24x16xembedded channels conv and mean centered"
    project_name = "PAE - Sensor suit Walking Data"
    
    # Prediction_Plotting_Slice
    training_prediction_start = 2531
    validation_prediction_start = 1990
    
        
    # Setup all the system paramters
    config, joint_imu_map, imu_joint_map = parameter_setup( file_name=file_name, project_name=project_name)
    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
    
    # Data setup
    data_path = "/home/cshah/workspaces/deepPhase based work/Data/04_21_2025_walking_data.csv"
    
    training_dataset, validation_dataset, col_names = setup_datasets( file_path=data_path, joint_imu_map=joint_imu_map, config=config)
    
    training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    
    # Taking a subset of the training and validation to plot a window / slice of data
    training_dataset_plotting = Subset(training_dataset, range(training_prediction_start, training_prediction_start+900))
    validation_dataset_plotting = Subset(validation_dataset, range(validation_prediction_start, validation_prediction_start+900))
    
    training_dataloader_plotting = DataLoader(training_dataset_plotting, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    validation_dataloader_plotting = DataLoader(validation_dataset_plotting, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    
    # Model Setup
    model = utility.ToDevice(PAE.Model(
                          input_channels=config["inputs"],
                          embedding_channels=config["phases"],
                          intermediate_channels=config["intermediate_channels"],
                          time_range=config["seq_length"],
                          window=config["training_window"]
                         ))
    
    # Train Model
    training_losses, validation_losses = train_model(model=model, config=config, training_dataloader=training_dataloader, 
                                                   validation_dataloader=validation_dataloader, log_wandB=log_wandB)

    
    # Save the Model
    model_save_location = "Saved Models/" + datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"] + ".pth"
    torch.save(model.state_dict(), model_save_location)
    
    # Plot files
    # Option to plot loss plot
    testing_losses = None
    plot_save_location = "Plots/" + config["training_tag"] 
    loss_plot_save_location = plot_save_location + "_loss_plot.png"
    utility.loss_plot(training_losses, validation_losses, testing_losses, loss_plot_save_location)
    
    model_file = model_save_location
    # model_file = "/home/cshah/workspaces/deepPhase based work/Saved Models/20250424_1118_PAE - sensor suit Walking Data - seq-length-300, 8 Phases and mean centered.pth"
    
    plot_model_predictions(training_dataloader_plotting, validation_dataloader_plotting, model_file, imu_joint_map, config, col_names, plot_save_location)
    
    
    # End wandB logging
    if log_wandB:
        wandb.finish()
        
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
    
