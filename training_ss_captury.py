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
from Models.QuaternionValueNetwork import QuaternionNet, QVNN_AutoEncoder
from Models.LSTM import LSTM
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import torch
from Library import Plotting as plot
from scipy.spatial.transform import Rotation as R
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
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
        "epochs": 30,
        "batch_size": 32,
        "num_workers": 8,
        "momentum":0.9,
        "lr": 1e-4,
        "dropout": 0.0,
        "dataset": "IHMC Senorsuit",
        "inputs": 28,
        "outputs": 28,
        "model_inputs": 42,
        "model_outputs": 42,
        "seq_length": 150,
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
    lossFn = geodesic_loss_batched
    # lossFn = nn.MSELoss()

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
            
            # Forward pass
            y_pred = model(utility.ToDevice(inputs))
        
            # y_pred = y_pred.view(-1, 7, 4)
            # outputs = outputs.view(-1, 7, 4)
            
            # Calculate the loss
            loss = lossFn(y_pred, utility.ToDevice(outputs))

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


def plot_results(treadmill_walking_val, walking_val, squats_val, random_val, training_dataloader, validation_dataloader, model):
    
    # Plot the training dataloader
    plot_dl(treadmill_walking_val, model)
    plot_dl(walking_val, model)
    plot_dl(squats_val, model)
    plot_dl(random_val, model)
    plot_dl(training_dataloader, model)
    plot_dl(validation_dataloader, model)
    
    
def plot_dl(dataloader, model, plot_save_name=None):
    model.eval()
    
    ground_truth = []
    preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            
            inputs, outputs = batch
            
            pred = model(utility.ToDevice(inputs))

            output_np = outputs.numpy()
            pred_np = utility.Item(pred).numpy()

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
    file_name = "Sensor-Suit-Captury-Data"
    project_name = "Quaternion Predictor"
    
    # Setup all the system paramters
    config = parameter_setup( file_name=file_name, project_name=project_name)
    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
        
    ## Data Setup
    # data_path = "/home/cshah/workspaces/deepPhase based work/Data/Quaternion_training_data_pelvis_frame_rel_quats.csv"
    # data_path = "/home/cshah/workspaces/deepPhase based work/Data/07_02_combined_data.csv"
    # val_path = "/home/cshah/workspaces/deepPhase based work/Data/07_02_combined_data_validate.csv"
    
    # walking_data = "/home/cshah/workspaces/deepPhase based work/Data/07_02_walking_quaternion_data.csv"
    # squats_data = "/home/cshah/workspaces/deepPhase based work/Data/07_02_squats_validate.csv"
    # random_data = "/home/cshah/workspaces/deepPhase based work/Data/07_02_random_validate.csv"
    
    data_path = "/home/cshah/workspaces/deepPhase based work/Data/07_09_Nicole/07_09_Nicole_combined_data.csv"
    val_path = "/home/cshah/workspaces/deepPhase based work/Data/07_09_Nicole/07_09_Nicole_combined_data_validate.csv"
    
    treadmill_walking_data = "/home/cshah/workspaces/deepPhase based work/Data/07_09_Nicole/07_09_Nicole_TW_validate.csv"
    walking_data = "/home/cshah/workspaces/deepPhase based work/Data/07_09_Nicole/07_09_Nicole_WA_validate.csv"
    squats_data = "/home/cshah/workspaces/deepPhase based work/Data/07_09_Nicole/07_09_Nicole_squats_validate.csv"
    random_data = "/home/cshah/workspaces/deepPhase based work/Data/07_09_Nicole/07_09_Nicole_random_validate.csv"
  
    
    df = pd.read_csv(data_path)
    val_df = pd.read_csv(val_path)
    
    treadmill_walking_df = pd.read_csv(treadmill_walking_data)     
    walking_data_df = pd.read_csv(walking_data)
    squats_data_df = pd.read_csv(squats_data)
    random_data_df = pd.read_csv(random_data)
    
  
    for col in df.columns:
        df[col] = df[col].astype(float)
        
    
    training_dataset = dataLoader_simple_loader(df, config["seq_length"], config["inputs"], config["outputs"])
    validation_dataset = dataLoader_simple_loader(val_df, config["seq_length"], config["inputs"], config["outputs"])
    
    training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    
    treadmill_walking_dataset = dataLoader_simple_loader(treadmill_walking_df, config["seq_length"], config["inputs"], config["outputs"])
    walking_dataset = dataLoader_simple_loader(walking_data_df, config["seq_length"], config["inputs"], config["outputs"])
    squats_dataset = dataLoader_simple_loader(squats_data_df, config["seq_length"], config["inputs"], config["outputs"])
    random_dataset = dataLoader_simple_loader(random_data_df, config["seq_length"], config["inputs"], config["outputs"])

    treadmill_dataloader = DataLoader(treadmill_walking_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    walking_dataloader = DataLoader(walking_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    squats_dataloader = DataLoader(squats_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    random_dataloader = DataLoader(random_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    
    
    
    
        
    # training_dataloader_plotting = DataLoader(training_dataset, batch_size=32, shuffle=False, num_workers=1)
    # validation_dataloader_plotting = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=1)
    
    
    # Load Model
    # model = utility.ToDevice(FCNN(config["model_inputs"], 
    #                         config["model_outputs"], 
    #                         config["num_hidden_layers"],
    #                         config["num_hidden_neurons"],
    #                         0.3))
    
    model = utility.ToDevice(AutoEncoder(input_dim=4200, output_dim=28, latent_dim=8, num_layers=3, hidden_dim=256, dropout_rate=0.3))
    
    # model = utility.ToDevice(QuaternionNet(input=7, hidden_dim=64, output=7))
    # model = utility.ToDevice(QVNN_AutoEncoder(in_quats=7, out_quats=7, latent_dim=32))
    
    # model = utility.ToDevice(LSTM(28,64,2,28))
    # Train
    training_losses, validation_losses = train_model(model=model, config=config, training_dataloader=training_dataloader, 
                                                   validation_dataloader=validation_dataloader, log_wandB=log_wandB)
    
    plot_results(treadmill_dataloader, walking_dataloader, squats_dataloader, random_dataloader, training_dataloader, validation_dataloader, model)
    
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
    