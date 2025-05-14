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

## Setup all the parameters

def parameter_setup(file_name, project_name):
    config = {
        "training_tag": file_name,
        "project_name": project_name,
        "epochs": 40,
        "batch_size": 32,
        "num_workers": 8,
        "momentum":0.9,
        "lr": 1e-4,
        "dropout": 0.0,
        "dataset": "IHMC Senorsuit",
        "seq_length": 301,
        "inputs": 24,
        "outputs": 24,
        "phases": 10,
        "intermediate_channels": 16,
        "training_window": 2.0, # How many seconds of data you are reviewing
        "data_recorded_rate": 150 # 
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
    
    
    # Split the pandas dataframe into a training and validation dataset
    # data_split = 179580 # 04_21_2025
    data_split_start = 80427 # 05_08_2025
    data_split_end = 95427
    
    training_range_of_data = list(range(0,data_split_start)) + list(range(data_split_end,len(data)))
    validation_range_of_data = list(range(data_split_start,data_split_end)) 
                              
    
    training_df = data.iloc[training_range_of_data].reset_index(drop=True)
    validation_df = data.iloc[validation_range_of_data].reset_index(drop=True)
    
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
    
    # #Initialize drawing
    # plotting_interval = 1000
    # plt.ion()
    # _, ax1 = plt.subplots(6,1)
    # _, ax2 = plt.subplots(config["phases"],5)
    # _, ax3 = plt.subplots(1,2)
    # _, ax4 = plt.subplots(2,1)
    # dist_amps = []
    # dist_freqs = []
    # loss_history = utility.PlottingWindow("Loss History", ax=ax4, min=0, drawInterval=plotting_interval)
    
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
            
        #     # #Start Visualization Section
        #     # _a_ = utility.Item(params[2]).squeeze().numpy()
        #     # for i in range(_a_.shape[0]):
        #     #     dist_amps.append(_a_[i,:])
        #     # while len(dist_amps) > 10000:
        #     #     dist_amps.pop(0)

        #     # _f_ = utility.Item(params[1]).squeeze().numpy()
        #     # for i in range(_f_.shape[0]):
        #     #     dist_freqs.append(_f_[i,:])
        #     # while len(dist_freqs) > 10000:
        #     #     dist_freqs.pop(0)

        #     # loss_history.Add(
        #     #     (utility.Item(loss).item(), "Reconstruction Loss")
        #     # )
            
        #     # if loss_history.Counter == 0:
        #     #     model.eval()

        #     #     plot.Functions(ax1[0], utility.Item(flattened_inputs[0]).reshape(model.input_channels,config["seq_length"]), -1.0, 1.0, -5.0, 5.0, title="Motion Curves" + " " + str(model.input_channels) + "x" + str(config["seq_length"]), showAxes=False)
        #     #     plot.Functions(ax1[1], utility.Item(latent[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Convolutional Embedding" + " " + str(config["phases"]) + "x" + str(config["seq_length"]), showAxes=False)
        #     #     plot.Circles(ax1[2], utility.Item(params[0][0]).squeeze(), utility.Item(params[2][0]).squeeze(), title="Learned Phase Timing"  + " " + str(config["phases"]) + "x" + str(2), showAxes=False)
        #     #     plot.Functions(ax1[3], utility.Item(signal[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Parametrized Signal" + " " + str(config["phases"]) + "x" + str(config["seq_length"]), showAxes=False)
        #     #     plot.Functions(ax1[4], utility.Item(flattened_outputs[0]).reshape(model.input_channels,config["seq_length"]), -1.0, 1.0, -5.0, 5.0, title="Curve Reconstruction" + " " + str(model.input_channels) + "x" + str(config["seq_length"]), showAxes=False)
        #     #     plot.Function(ax1[5], [utility.Item(flattened_inputs[0]), utility.Item(flattened_outputs[0])], -1.0, 1.0, -5.0, 5.0, colors=[(0, 0, 0), (0, 1, 1)], title="Curve Reconstruction (Flattened)" + " " + str(1) + "x" + str(model.input_channels*config["seq_length"]), showAxes=False)
        #     #     plot.Distribution(ax3[0], dist_amps, title="Amplitude Distribution")
        #     #     plot.Distribution(ax3[1], dist_freqs, title="Frequency Distribution")

        #     #     # indices = gather_window + random.choice(test_sequences)
        #     #     # _, _, _, params = network(LoadBatches(indices))

        #     #     for i in range(config["phases"]):
        #     #         phase = params[0][:,i]
        #     #         freq = params[1][:,i]
        #     #         amps = params[2][:,i]
        #     #         offs = params[3][:,i]
        #     #         plot.Phase1D(ax2[i,0], utility.Item(phase), utility.Item(amps), color=(0, 0, 0), title=("1D Phase Values" if i==0 else None), showAxes=False)
        #     #         plot.Phase2D(ax2[i,1], utility.Item(phase), utility.Item(amps), title=("2D Phase Vectors" if i==0 else None), showAxes=False)
        #     #         plot.Functions(ax2[i,2], utility.Item(freq).transpose(0,1), -1.0, 1.0, 0.0, 4.0, title=("Frequencies" if i==0 else None), showAxes=False)
        #     #         plot.Functions(ax2[i,3], utility.Item(amps).transpose(0,1), -1.0, 1.0, 0.0, 1.0, title=("Amplitudes" if i==0 else None), showAxes=False)
        #     #         plot.Functions(ax2[i,4], utility.Item(offs).transpose(0,1), -1.0, 1.0, -1.0, 1.0, title=("Offsets" if i==0 else None), showAxes=False)
                
        #     #     #Visualization
        #     #     # pca_indices = []
        #     #     # pca_batches = []
        #     #     # pivot = 0
        #     #     # pca_sequence_count = 100
        #     #     # for i in range(pca_sequence_count):
        #     #     #     # indices = gather_window + random.choice(test_sequences)
        #     #     #     # _, _, _, params = network(LoadBatches(indices))
        #     #     #     a = utility.Item(params[2]).squeeze()
        #     #     #     p = utility.Item(params[0]).squeeze()
        #     #     #     b = utility.Item(params[3]).squeeze()
        #     #     #     m_x = a * np.sin(2.0 * np.pi * p) + b
        #     #     #     m_y = a * np.cos(2.0 * np.pi * p) + b
        #     #     #     manifold = torch.hstack((m_x, m_y))
        #     #     #     pca_indices.append(pivot + np.arange(len(indices)))
        #     #     #     pca_batches.append(manifold)
        #     #     #     pivot += len(indices)

        #     #     # plot.PCA2D(ax4[0], pca_indices, pca_batches, "Phase Manifold (" + str(pca_sequence_count) + " Random Sequences)")

            #     plt.gcf().canvas.draw_idle()
            # plt.gcf().canvas.start_event_loop(1e-5)
            # #End Visualization Section
            
        
        train_loss = running_loss / len(training_dataloader.dataset)
        training_losses.append(train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss}')
    
    
        val_loss, individual_loss, latents_np, signal_np = utility.cal_validation_loss(model, validation_dataloader, lossFn, lossFn_no_reduction)
        validation_losses.append(val_loss)
        individual_losses.append(individual_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
        
        if (epoch+1)==epochs:
            file_name = "Plots/" + config["training_tag"] + "_Latent_signal.png"
            utility.plot_conv_deconv(latents_np, signal_np, config, file_name)
    
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
    file_name = "PAE - sensor suit Walking Data (132000) - seq-length-300, 10 Phases 24x16xembedded channels conv and mean centered"
    project_name = "PAE - Sensor suit Walking Data - With Future predictions + New Thigh IMU location"
    
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
    data_path = "/home/cshah/workspaces/deepPhase based work/Data/05_08_2025_walking_data.csv"
    
    training_dataset, validation_dataset, col_names = setup_datasets( file_path=data_path, joint_imu_map=joint_imu_map, config=config)
    
    training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    
    # Taking a subset of the training and validation to plot a window / slice of data
    training_dataset_plotting = Subset(training_dataset, range(training_prediction_start, training_prediction_start+900))
    validation_dataset_plotting = Subset(validation_dataset, range(validation_prediction_start, validation_prediction_start+900))
    
    training_dataloader_plotting = DataLoader(training_dataset_plotting, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    validation_dataloader_plotting = DataLoader(validation_dataset_plotting, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    
    # Model Setup
    
    # model_file = "/home/cshah/workspaces/deepPhase based work/Saved Models/20250429_1104_PAE - sensor suit Walking Data (300000) - seq-length-300, 10 Phases 24x16xembedded channels conv and mean centered.pth"
    # weights = torch.load(model_file, weights_only=True)
    
    model = utility.ToDevice(PAE.Model(
                          input_channels=config["inputs"],
                          embedding_channels=config["phases"],
                          intermediate_channels=config["intermediate_channels"],
                          time_range=config["seq_length"],
                          window=config["training_window"]
                         ))
    
    # model.load_state_dict(weights)
    
    # # Train Model
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
    
