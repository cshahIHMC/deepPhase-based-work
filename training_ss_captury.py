######### Author - Chinmay Shah #################


## Imports
import wandb
from Library import utility
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DataLoader.data_loader_simple_loader import dataLoader_simple_loader
from torch.utils.data import Dataset, DataLoader, Subset
from Models.FCNN import FCNN
from Models.AutoEncoder import AutoEncoder
from Models.QuaternionValueNetwork import QuaternionNet, QVNN_AutoEncoder, QVAE
from Models.TCNN import TCNModel
from Models.LSTM import LSTM
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import torch
from Library import Plotting as plot
from scipy.spatial.transform import Rotation as R
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import torch.nn.functional as F
# from geodesic_loss import GeodesicLoss


def sincos_to_euler_all(flat_sin_cos):
    """
    Converts shape (42,) → Euler angles of shape (7, 3) based on [sin(x,y,z), cos(x,y,z)] format per joint
    """


    sin_part = flat_sin_cos[:, :, 0:3]  # sin(x), sin(y), sin(z)
    cos_part = flat_sin_cos[:, :, 3:6]  # cos(x), cos(y), cos(z)

    euler_angles = np.arctan2(sin_part, cos_part) * 180 / np.pi  # shape (7, 3)
    return euler_angles  # in radians


def normalize_quaternions(q):
    return q / q.norm(dim=2, keepdim=True).clamp(min=1e-8)

def normalize_np_quat(q):
    # Assume preds is your (50000, 7, 4) array of quaternions
    norm = np.linalg.norm(q, axis=-1, keepdims=True)  # shape: (50000, 7, 1)

    # Avoid divide-by-zero
    norm[norm == 0] = 1.0

    # Normalize
    q_norm = q / norm
    
    return q_norm

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

def vae_loss_function(recon_x, target_x, mu, logvar, beta=1.0, use_cosine=False):
    if use_cosine:
        recon_loss = torch.mean(1 - torch.cos(recon_x - target_x))
    else:
        recon_loss = F.mse_loss(recon_x, target_x)

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss



def batch_sincos_to_euler(sin_cos_array):
    """
    Converts an array of shape (N, 7, 6) from sin–cos representation to Euler angles in radians.
    
    Parameters:
    - sin_cos_array: shape (N, 7, 6)

    Returns:
    - euler_angles: shape (N, 7, 3) in radians
    """
    assert sin_cos_array.shape[-1] == 6, "Last dimension must be 6 for sin–cos pairs"

    sin = sin_cos_array[..., 0::2]  # shape (N, 7, 3) → sin(x), sin(y), sin(z)
    cos = sin_cos_array[..., 1::2]  # shape (N, 7, 3) → cos(x), cos(y), cos(z)

    angles = np.arctan2(sin, cos)  # still shape (N, 7, 3)
    return angles

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
        "inputs": 28,
        "outputs": 10,
        "model_inputs": 42,
        "model_outputs": 42,
        "seq_length": 1,
        "num_hidden_layers": 4,
        "num_hidden_neurons": 256
    }
      
    return config
# Training Function
def train_model(model, config, training_dataloader, validation_dataloader, log_wandB=False):   
     
    ## Setting up an optimizer and a loss function - Original Paper used a AdamWr optimizer We using a simple SGD
    learning_rate = config["lr"]
    momentum = config["momentum"]

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    # lossFn = geodesic_loss_batched
    lossFn = nn.MSELoss()
    # lossFn = vae_loss_function

    ## Training the periodic auto encoder
    print("Starting Training........")
    training_losses = []
    validation_losses = []


    epochs = config["epochs"]
    
    ## Training Loop
    for epoch in range(epochs):
        
        model.train()
    
        running_loss = 0.0
    
        for batch in training_dataloader:
        
            inputs, outputs = batch    
            
            # inputs = inputs.squeeze(1)  # Now shape is (B, 28)
            # inputs = inputs.view(inputs.size(0), 7, 4)  # Now shape is (B, 7, 4)
               
            # Forward pass
            y_pred = model(utility.ToDevice(inputs))
            
            # y_pred = y_pred.view(-1, 7, 4)
            # outputs = outputs.view(-1, 7, 4)
            
            # Calculate the loss
            loss = lossFn(y_pred, utility.ToDevice(outputs))
            # loss, recon_loss, kl_loss = lossFn(y_pred, utility.ToDevice(outputs.squeeze(1)), mu, logvar, beta=0.5)

            # # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(training_dataloader.dataset)
        training_losses.append(train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss}')
        
        val_loss = utility.cal_val_loss(model, validation_dataloader, lossFn)
        validation_losses.append(val_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
        
        if log_wandB:
            wandb.log({"train/train_loss": train_loss,
                        "train/epoch": epoch,
                        "val/val_loss": val_loss,
                        "val/epoch":epoch})
        
            
    return training_losses, validation_losses           


# def plot_results(treadmill_walking_val, walking_val, squats_val, random_val, training_dataloader, validation_dataloader, model):
def plot_results(training_dataloader, validation_dataloader,  model, col_names):    

    # plot all data
    plot_dl(training_dataloader, model, col_names)
    plot_dl(validation_dataloader, model, col_names)
    
    
def plot_dl(dataloader, model, col_names, plot_save_name=None):
    model.eval()
    
    ground_truth = []
    preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            
            inputs, outputs = batch
            
            # inputs = inputs.squeeze(1)  # Now shape is (B, 28)
            # inputs = inputs.view(inputs.size(0), 7, 4)  # Now shape is (B, 7, 4)
            
            pred = model(utility.ToDevice(inputs))
            

            output_np = outputs.numpy()
            pred_np = utility.Item(pred).numpy()
            
            preds.append(pred_np)
            ground_truth.append(output_np)
                  

    # Concatenate all batch outputs
    preds_arr = np.concatenate(preds, axis=0).squeeze(axis=1)         # shape: (N, 1, 10)
    ground_truth_arr = np.concatenate(ground_truth, axis=0).squeeze(axis=1)  # shape: (N, 1, 10)
    
    # preds_arr = np.concatenate(preds, axis=0)       # shape: (N, 1, 10)
    # ground_truth_arr = np.concatenate(ground_truth, axis=0).squeeze(1)  # shape: (N, 1, 10)

    
    # Unnormalize data
    preds_arr_unnormalized = preds_arr * dataloader.dataset.output_data_std.to_numpy() + dataloader.dataset.output_data_mean.to_numpy()
    ground_truth_arr_unnormalized = ground_truth_arr * dataloader.dataset.output_data_std.to_numpy() + dataloader.dataset.output_data_mean.to_numpy()
    
    # Joint Wise MAE
    abs_errors = np.abs(preds_arr_unnormalized - ground_truth_arr_unnormalized) 
    mae_per_joint_per_channel = abs_errors.mean(axis=0)  
    
    # Joint wise RMSE
    squared_errors = (preds_arr_unnormalized - ground_truth_arr_unnormalized) ** 2
    rmse_per_joint_per_channel = np.sqrt(squared_errors.mean(axis=0))

    
    # Standard deviation over the samples (dim=0)
    std_per_joint_per_channel = abs_errors.std(axis=0)
    
    
    joints = col_names 

    # Print results
    for joint_idx, joint in enumerate(joints):
        print(f"Joint {joint} MAE = {mae_per_joint_per_channel[joint_idx]}, STD = {std_per_joint_per_channel[joint_idx]}, RMSE = {rmse_per_joint_per_channel[joint_idx]} ")
        
    # Create subplots
    fig, axs = plt.subplots(5, 2, figsize=(12, 16), sharex=True, sharey=True)
    axs = axs.flatten()  # Flatten to loop easily

    for i in range(10):
        axs[i].plot(preds_arr_unnormalized[:, i], label='Prediction')
        axs[i].plot(ground_truth_arr_unnormalized[:, i], label='Ground Truth')
        axs[i].set_title(col_names[i])
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


def joint_angle_calculator(np_array):
    
    euler_angles = np.zeros((np_array.shape[0] , np_array.shape[1] // 4, 3))
    
    for i in range(np_array.shape[0]):
        for j in range(0, np_array.shape[1], 4):
            
            q = np_array[i, j:j+4]
            r = R.from_quat(q)
            euler = r.as_euler('xyz', degrees=True)
            euler_angles[i, j//4, :] = euler
            

    
    return euler_angles
        
def main():
    
    # Logging Flag
    log_wandB = False
       
    # file_name = "trial"
    file_name = "EST - Sensor-Suit-Vicon-Data - Sub 1 - 150 milliseconds in the past"
    project_name = "Full Pipeline Training"
    
    # Setup all the system paramters
    config = parameter_setup( file_name=file_name, project_name=project_name)
    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
        
    ## Data Setup
    data_path = "/home/cshah/workspaces/deepPhase based work/Data/Vicon_SS_Data_trial/train/EST_Vicon_train_lim.csv"
    gyro_data = "/home/cshah/workspaces/deepPhase based work/Data/Vicon_SS_Data_trial/train/PAE_Vicon_train_lim.csv"
    
    val_path = "/home/cshah/workspaces/deepPhase based work/Data/Vicon_SS_Data_trial/validate/EST_Vicon_val_lim.csv"
    gyro_val_path = "/home/cshah/workspaces/deepPhase based work/Data/Vicon_SS_Data_trial/validate/PAE_Vicon_val_lim.csv"
    
    df = pd.read_csv(data_path)
    gyro_df = pd.read_csv(gyro_data)
    
    val_df = pd.read_csv(val_path)
    gyro_val_df = pd.read_csv(gyro_val_path)
    
    for col in df.columns:
        df[col] = df[col].astype(float)
        
    col_names = df.columns[28:]
        
    print("Training df size:")
    print(df.shape)
    print("Validation df size:")
    print(val_df.shape)
    
    training_dataset = dataLoader_simple_loader(df, gyro_df, config["seq_length"], config["inputs"], config["outputs"])
    validation_dataset = dataLoader_simple_loader(val_df, gyro_df, config["seq_length"], config["inputs"], config["outputs"])
    training_dataset_plot = dataLoader_simple_loader(df, gyro_val_df, config["seq_length"], config["inputs"], config["outputs"])
    
    
    training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    training_dataloader_plot = DataLoader(training_dataset_plot, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    # Load Model
    # model = utility.ToDevice(FCNN(config["model_inputs"], 
    #                         config["model_outputs"], 
    #                         config["num_hidden_layers"],
    #                         config["num_hidden_neurons"],
    #                         0.3))
    
    model = utility.ToDevice(AutoEncoder(input_dim=70, output_dim=10, latent_dim=8, num_layers=4, hidden_dim=256, dropout_rate=0.3))
    
    # model = utility.ToDevice(TCNModel(28,10,[128,64,32,16,8]))
    
    # model = utility.ToDevice(QuaternionNet(input=7, hidden_dim=64, output=7))
    # model = utility.ToDevice(QVAE(7,8,10,128,0.15))
    
    # model = utility.ToDevice(LSTM(28,64,2,28))
    
    # Train
    training_losses, validation_losses = train_model(model=model, config=config, training_dataloader=training_dataloader, 
                                                   validation_dataloader=validation_dataloader, log_wandB=log_wandB)
    
    # Save the Model
    # model_save_location = "Saved Models/" + datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"] + ".pth"
    # torch.save(model.state_dict(), model_save_location)
    
    plot_results( training_dataloader_plot, validation_dataloader, model, col_names)
    
    # # Save the Model
    # model_save_location = "Saved Models/" + datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"] + ".pth"
    # torch.save(model.state_dict(), model_save_location)
    
    # # Plot files
    # # Option to plot loss plot
    # testing_losses = None
    # plot_save_location = "Plots/" + config["training_tag"] 
    # loss_plot_save_location = plot_save_location + "_loss_plot.png"
    # utility.loss_plot(training_losses, validation_losses, testing_losses, loss_plot_save_location)
    
    return 0
    
    
    
    

if __name__ == "__main__":
    raise SystemExit(main())
    