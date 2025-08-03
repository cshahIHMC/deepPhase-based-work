######### Author - Chinmay Shah #################


## Imports
import wandb
from Library import utility
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DataLoader.data_future_prediction_loader import dataLoader_future_prediction_loader
from torch.utils.data import Dataset, DataLoader, Subset
from Models import PAE
from Models.FCNN import FCNN
from Models.MANN_org import Model
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import torch
from Library import Plotting as plot
from Library.IMUMotionAnalyzer import IMUMotionAnalyzer
from scipy.spatial.transform import Rotation as R

def normalize_quaternions(q):
    return q / q.norm(dim=2, keepdim=True).clamp(min=1e-8)

def geodesic_loss_batched(preds, targets):
    # preds, targets: (B, 1, 28) → reshape to (B, 7, 4)
    preds = preds.view(-1, 7, 4)
    targets = targets.view(-1, 7, 4)

    preds = normalize_quaternions(preds)
    targets = normalize_quaternions(targets)

    # Inner product per quaternion
    inner_prod = torch.sum(preds * targets, dim=2).abs()  # (B, 7)
 
    inner_prod = torch.clamp(inner_prod, -1.0 + 1e-6, 1.0 - 1e-6)


    theta = 2 * torch.acos(inner_prod)  # (B, 7) in radians

    return theta.mean()  # scalar loss

def joint_angle_calculator(np_array):
    
    euler_angles = np.zeros((np_array.shape[0] , np_array.shape[1] // 4, 3))
    
    for i in range(np_array.shape[0]):
        for j in range(0, np_array.shape[1], 4):
            
            q = np_array[i, j:j+4]
            r = R.from_quat(q)
            euler = r.as_euler('xyz', degrees=True)
            euler_angles[i, j//4, :] = euler
            

    
    return euler_angles

def normalize_np_quat(q):
    # Assume preds is your (50000, 7, 4) array of quaternions
    norm = np.linalg.norm(q, axis=-1, keepdims=True)  # shape: (50000, 7, 1)

    # Avoid divide-by-zero
    norm[norm == 0] = 1.0

    # Normalize
    q_norm = q / norm
    
    return q_norm       

## Setup all the parameters
def parameter_setup(file_name, project_name):
    config = {
        "training_tag": file_name,
        "project_name": project_name,
        "epochs": 10,
        "batch_size": 32,
        "num_workers": 8,
        "momentum":0.9,
        "lr": 1e-4,
        "dropout": 0.0,
        "dataset": "IHMC Senorsuit",
        "PAE_seq_length": 301,
        "PAE_inputs": 21,
        "PAE_outputs": 21,
        "PAE_phases": 10,
        "PAE_intermediate_channels": 16,
        "training_window": 2.0, # How many seconds of data you are reviewing
        "data_recorded_rate": 150, # 
        "FCNN_inputs": 28,
        "FCNN_outputs": 28,
        "FCNN_seq_length":1,
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
def setup_datasets(pae_train_path, quat_train_path, pae_val_path, quat_val_path, joint_imu_map, config):
    
    # file path to the data
    pae_train_df = pd.read_csv(pae_train_path)
    quat_train_df = pd.read_csv(quat_train_path)
    # quat_train_df = quat_train_df.iloc[0:203000,:]
    
    pae_val_df = pd.read_csv(pae_val_path)
    # pae_val_df = pae_val_df.iloc[0:24000,:]
    quat_val_df = pd.read_csv(quat_val_path)

    # ## Calculate the joint angles and get it as a dataframe
    # motion_analyzer = IMUMotionAnalyzer(csv_path)
    # motion_analyzer.analyze()
    # joint_angles_df = motion_analyzer.get_joint_angles()
    
    # ## Add the joint angle cols to data columns
    # df_combined = pd.concat([data,joint_angles_df], axis=1)
        
    # # Split the pandas dataframe into a training and validation dataset
    # # data_split = 179580 # 04_21_2025
    # data_split_start = 80427 # 05_08_2025
    # data_split_end = 95427
    
    # training_range_of_data = list(range(0,data_split_start)) + list(range(data_split_end,len(data)))
    # validation_range_of_data = list(range(data_split_start,data_split_end)) 
                              
    
    # training_df = df_combined.iloc[training_range_of_data].reset_index(drop=True)
    # validation_df = df_combined.iloc[validation_range_of_data].reset_index(drop=True)
    
    # Checking the Size of the data frame
    print("PAE Training DF size:", pae_train_df.shape)
    print("PAE Validation DF size:", pae_val_df.shape)
    
    print("Quat Training DF size:", quat_train_df.shape)
    print("Quat Validation DF size:", quat_val_df.shape)
    
    # Above data frames have the entire dataset
    # Get only gyro data
    
    # extracted_training_df = utility.extract_data(training_df, joint_imu_map)
    # extracted_validation_df = utility.extract_data(validation_df, joint_imu_map)
    
    ## Data preprocessing
    # The xsensor data is recorded in deg/sec while microstrain data is recorded in rad/sec.
    # To make everything consistent we convert the xsensor data to rad/sec
    # Convert all the angle columns to radians
    col_to_modify = ["R_insole_gyro_x", "R_insole_gyro_y" , "R_insole_gyro_z",
                     "L_insole_gyro_x", "L_insole_gyro_y" , "L_insole_gyro_z"]
                    #  "thigh_r_angle_y", "thigh_l_angle_y", "shank_r_angle_y",
                    #  "shank_l_angle_y", "foot_r_angle_y", "foot_l_angle_y"]

    for col in col_to_modify:
        pae_train_df[col] = pae_train_df[col] * np.pi / 180
        pae_val_df[col] = pae_val_df[col] * np.pi / 180
        # extracted_testing_df[col] = extracted_testing_df[col] * np.pi / 180

    
    # Setup custom datasets
    training_dataset = dataLoader_future_prediction_loader(pae_train_df, quat_train_df, config["PAE_seq_length"], config["FCNN_seq_length"])
    validation_dataset = dataLoader_future_prediction_loader(pae_val_df, quat_val_df, config["PAE_seq_length"], config["FCNN_seq_length"])
    
    
    return training_dataset, validation_dataset, quat_train_df.columns

## Training Function
def train_model(model, config, training_dataloader, validation_dataloader, PAE_model, log_wandB=False):   
    
    

    
    ## Setting up an optimizer and a loss function - Original Paper used a AdamWr optimizer We using a simple SGD
    learning_rate = config["lr"]
    momentum = config["momentum"]

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    # lossFn = nn.MSELoss()
    # lossFn_no_reduction = nn.MSELoss(reduction='none')
    
    lossFn = geodesic_loss_batched

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
            # print("FNN inputs last slice ", FCNN_inputs.shape)
            # print("FNN outputs:", FCNN_outputs.shape)

            
            PAE_inputs = utility.ToDevice(PAE_inputs)
            
            PAE_model.eval()
            _, _, _, params  = PAE_model(PAE_inputs)
            
            params_cat = torch.cat(params, dim=2)
            # phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
            # phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
            # phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])
            
            # phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 
 
            # phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)
            
            
            # Flattening the inputs for the motion prediction network
            flattened_inputs = utility.ToDevice(FCNN_inputs.reshape(FCNN_inputs.shape[0], -1))
            
            # FCNN_combine_inputs = torch.cat((flattened_inputs, phaseInputs), dim=1)
            
            # Forwards pass (Combine data with PAE)
            # y_pred = model(FCNN_combine_inputs)
            
            # MANN
            # y_pred, _ = model(phaseInputs, flattened_inputs)
            
            # 1 TIme step future prediction
            FCNN_inputs = FCNN_inputs.squeeze(-1)
            y_pred = model(utility.ToDevice(FCNN_inputs.squeeze(-1)))
            
            # 20 time step - Predict 1
            # y_pred = model(flattened_inputs)
            
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
        
        val_loss = utility.cal_validation_loss_future_prediction(model, PAE_model, validation_dataloader, lossFn)
        validation_losses.append(val_loss)
        # individual_losses.append(individual_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
        
        if log_wandB:
            wandb.log({"train/train_loss": train_loss,
                        "train/epoch": epoch,
                        "val/val_loss": val_loss,
                        "val/epoch":epoch})
        
            
    return training_losses, validation_losses           

def plot_results(training_dataloader, validation_dataloader, PAE_model, model):    
    plot_dl(training_dataloader, PAE_model, model)
    plot_dl(validation_dataloader, PAE_model, model)       

def plot_dl(dataloader, PAE_model, model, plot_save_name=None):
    model.eval()
    
    ground_truth = []
    preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            
            PAE_inputs, FCNN_inputs, FCNN_outputs = batch
            # print("PAE inputs size: ", PAE_inputs.shape)
            # print("FNN inputs last slice ", FCNN_inputs[-1,:,-1])
            # print("FNN outputs:", FCNN_outputs[-1,:])
            
            PAE_inputs = utility.ToDevice(PAE_inputs)
            
            PAE_model.eval()
            _, _, _, params  = PAE_model(PAE_inputs)
            
            # Flattening the inputs for the motion prediction network
            flattened_inputs = utility.ToDevice(FCNN_inputs.reshape(FCNN_inputs.shape[0], -1))

            # params_cat = torch.cat(params, dim=2)
            # # phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
            # phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
            # phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])
            
            # phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 
 
            # phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)
            
            # FCNN_combine_inputs = torch.cat((flattened_inputs, phaseInputs), dim=1)
            # FCNN_combine_inputs = flattened_inputs
            
             # MANN
            # y_pred, _ = model(phaseInputs, flattened_inputs)
            
            # Forwards pass (Combine data with PAE)
            # y_pred = model(FCNN_combine_inputs)
            
            
            # 1 Time step - 1 Time step
            FCNN_inputs = FCNN_inputs.squeeze(-1)
            y_pred = model(utility.ToDevice(FCNN_inputs))
            
            # 20 time step - Predict 1
            # y_pred = model(flattened_inputs)

            output_np = FCNN_outputs.numpy()
            pred_np = utility.Item(y_pred).numpy()

            preds.append(pred_np)
            ground_truth.append(output_np)
                  

    # Concatenate all batch outputs
    preds_arr = np.concatenate(preds, axis=0)         # shape: (N, 1, 28)
    ground_truth_arr = np.concatenate(ground_truth, axis=0)  # shape: (N, 1, 28)
    
    # preds, targets: (B, 1, 28) → reshape to (B, 7, 4)
    preds = preds_arr.reshape(preds_arr.shape[0], 7, 4)
    ground_truth = ground_truth_arr.reshape(ground_truth_arr.shape[0], 7, 4)
    
    pred_euler_angles = joint_angle_calculator( normalize_np_quat(preds_arr.squeeze()) )
    ground_truth_euler_angles = joint_angle_calculator( normalize_np_quat(ground_truth_arr.squeeze()) )
    
    # Convert the euler angles sin cos presentations to euler angles
    # preds_sin_cos = sincos_to_euler_all(preds)
    # ground_truth_sin_cos = sincos_to_euler_all(ground_truth)
    
    # pred_euler_angles = preds_sin_cos
    # ground_truth_euler_angles = ground_truth_sin_cos
    
    abs_errors = np.abs(pred_euler_angles - ground_truth_euler_angles)  # (N, 7, 3)
    mae_per_joint_per_channel = abs_errors.mean(axis=0)  # (7, 3)
    
    # Standard deviation over the samples (dim=0)
    std_per_joint_per_channel = abs_errors.std(axis=0)
    
    
    joints = ["pelvis", "thigh_r", "thigh_l", "shank_r", "shank_l", "foot_r", "foot_l"]  

    # Print results
    for joint_idx, joint in enumerate(joints):
        print(f"Joint {joint}: MAE [X, Y, Z] = {mae_per_joint_per_channel[joint_idx].tolist()}")
        
    for joint_idx, joint in enumerate(joints):
        print(f"Joint {joint}: STD [X, Y, Z] = {std_per_joint_per_channel[joint_idx].tolist()}")
    
    
        
    
    fig, axes = plt.subplots(pred_euler_angles.shape[2], 7, figsize=(20,12), sharex=True)
    
         
    
    for i in range(pred_euler_angles.shape[1]):
        
        # for j in range(len(pred_euler_angles)):
        #     sum = pred_euler_angles[j,i,0] ** 2 + pred_euler_angles[j,i,1] ** 2 + pred_euler_angles[j,i,2] ** 2 + pred_euler_angles[j,i,3] ** 2
        #     sum2 = ground_truth_euler_angles[j,i,0] ** 2 + ground_truth_euler_angles[j,i,1] ** 2 + ground_truth_euler_angles[j,i,2] ** 2 + ground_truth_euler_angles[j,i,3] ** 2
            
            # print("Sum 1: ", sum)
            # print("Sum 2: ", sum2)
            
        # Combine to compute display-centered y-axis limits
        combined = np.concatenate([pred_euler_angles, ground_truth_euler_angles], axis=0)  # shape (2N, 3)

        
        axes[0,i].plot( pred_euler_angles[:,i,0], linewidth=1, color="red")
        axes[0,i].plot( ground_truth_euler_angles[:,i,0], linewidth=0.75, color="black")
        axes[0,i].set_title(joints[i] + "_X ")
    
        
        axes[1,i].plot( pred_euler_angles[:,i,1], linewidth=1, color="green")
        axes[1,i].plot( ground_truth_euler_angles[:,i,1], linewidth=0.75, color="black")
        axes[1,i].set_title(joints[i] + "_Y ")
        
        axes[2,i].plot( pred_euler_angles[:,i,2], linewidth=1, color="blue")
        axes[2,i].plot( ground_truth_euler_angles[:,i,2], linewidth=0.75, color="black")
        axes[2,i].set_title(joints[i] + "_Z ")
        
        if pred_euler_angles.shape[2] >= 4:
            for k in range(3, pred_euler_angles.shape[2]):
                axes[k,i].plot( pred_euler_angles[:,i,k], linewidth=1, color="blue")
                axes[k,i].plot( ground_truth_euler_angles[:,i,k], linewidth=0.75, color="black")
                
                
                axes[k,i].set_title(joints[i] + "_W ")
        
        
        # axes[0,i].set_ylim(-1.5,1.5)
        # axes[1,i].set_ylim(-1.5,1.5)
        # axes[2,i].set_ylim(-1.5,1.5)

    plt.tight_layout()
    # plt.savefig(plot_save_name, dpi=300, bbox_inches='tight')
    # plt.close()
    plt.show()
        
def main():
    
    # Logging Flag
    log_wandB = False
    
     # Prediction_Plotting_Slice
    training_prediction_start = 2531
    validation_prediction_start = 1990

    
    PAE_model_file = "/home/cshah/workspaces/deepPhase based work/Saved Models/20250721_1118_PAE - Sensor-Suit-Captury-Data.pth"
    ss_captury_model_file = "/home/cshah/workspaces/deepPhase based work/Saved Models/20250718_1633_Sensor-Suit-Captury-Data.pth"
    
    # file_name = "trial"
    file_name = "1 time step future predictor (W/O Phase) - SS (t-75 - t-1) - Captury - t"
    project_name = "Future Predictor"
    
    # Setup all the system paramters
    config, joint_imu_map, imu_joint_map = parameter_setup( file_name=file_name, project_name=project_name)
    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
        
    ## Data Setup
    PAE_train_path = "/home/cshah/workspaces/deepPhase based work/Data/Full Training - Gyro + Joint Angles/PAE_new_sub2_train.csv"
    # Quat_train_path= "/home/cshah/workspaces/deepPhase based work/Data/Full Training - Gyro + Joint Angles/Quat_new_sub2_train.csv"
    Quat_train_path = "/home/cshah/workspaces/deepPhase based work/Data/Full Training - Gyro + Joint Angles/Quat_sub2_model_output_train.csv"
    
    PAE_val_path = "/home/cshah/workspaces/deepPhase based work/Data/Full Training - Gyro + Joint Angles/PAE_new_sub2_val.csv"
    # Quat_val_path = "/home/cshah/workspaces/deepPhase based work/Data/Full Training - Gyro + Joint Angles/Quat_new_sub2_val.csv"
    Quat_val_path = "/home/cshah/workspaces/deepPhase based work/Data/Full Training - Gyro + Joint Angles/Quat_sub2_model_output_val.csv"  
    
    training_dataset, validation_dataset, col_names = setup_datasets(pae_train_path=PAE_train_path, quat_train_path=Quat_train_path,
                                                                     pae_val_path=PAE_val_path, quat_val_path=Quat_val_path,
                                                                     joint_imu_map=joint_imu_map, config=config)
    
    training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    training_dataloader_plotting = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    # # Taking a subset of the training and validation to plot a window / slice of data
    # training_dataset_plotting = Subset(training_dataset, range(training_prediction_start, training_prediction_start+900))
    # validation_dataset_plotting = Subset(validation_dataset, range(validation_prediction_start, validation_prediction_start+900))
    
    # training_dataloader_plotting = DataLoader(training_dataset_plotting, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    # validation_dataloader_plotting = DataLoader(validation_dataset_plotting, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    
    ## Load Model
    # FCNN_inputs = config["FCNN_inputs"] * config["FCNN_seq_length"] + 4 * config["PAE_phases"]
    inputs = config["FCNN_inputs"] * config["FCNN_seq_length"]
    model = utility.ToDevice(FCNN(inputs, 
                            config["FCNN_outputs"], 
                            config["FCNN_num_hidden_layers"],
                            config["FCNN_num_hidden_neurons"],
                            config["FCNN_dropout"]))
    
    # model = utility.ToDevice(Model(50,256,4,inputs,512,28, 0.3))
    
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
    # Train
    
    training_losses, validation_losses = train_model(model=model, config=config, training_dataloader=training_dataloader, 
                                                   validation_dataloader=validation_dataloader, PAE_model=PAE_model, log_wandB=log_wandB)
    
    
    plot_results( training_dataloader_plotting, validation_dataloader, PAE_model, model)
        
    # Save the Model
    # model_save_location = "Saved Models/" + datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"] + ".pth"
    # torch.save(model.state_dict(), model_save_location)
    
    # Plot files
    # Option to plot loss plot
    testing_losses = None
    plot_save_location = "Plots/" + config["training_tag"] 
    loss_plot_save_location = plot_save_location + "_loss_plot.png"
    utility.loss_plot(training_losses, validation_losses, testing_losses, loss_plot_save_location)
    
    return 0
    
    
    
    

if __name__ == "__main__":
    raise SystemExit(main())
    
