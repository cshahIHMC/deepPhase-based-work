######## Author - Chinmay Shah ##############
''' 
This script contains all the functions needed often to either transfer to the GPU,
plot stuff or writing or processing data, misc helper functions
'''
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
import sys


# Check if the GPU is available and put the object on the GPU
def ToDevice(x):
    return x.cuda() if torch.cuda.is_available() else x

# To detach from the GPU
def Item(value):
        return value.detach().cpu()

# This function gets the entire data frame extracts the required data and returns it as dataframe subset
def extract_data(all_data, data_keys):
    
    columns_to_extract = []
    
    for keys, cols in data_keys.items():
        
        columns_to_extract.append(cols+"_gyro_x")
        columns_to_extract.append(cols+"_gyro_y")
        columns_to_extract.append(cols+"_gyro_z")
        
    
    
    # Extract all the angles
    for key in data_keys.keys():
        
        if "pelvis" in key or "back" in key:

            continue

        columns_to_extract.append(key+"_angle_y")
        
    # Extract all the Gyro's
    subset_df = all_data[columns_to_extract].copy()    
        
    return subset_df

# Simple function to plot every col of a pandas dataframe independantly
def plot_df(df, normalized_df):
    num_cols = len(df.columns)
    print(num_cols)
    fig, axes = plt.subplots(num_cols, 1, figsize=(10, 4 * num_cols), sharex=True)

    for i, col in enumerate(df.columns):
        ax = axes[i] if num_cols > 1 else axes
        df[col].plot(ax=ax, label='1')
        # normalized_df[col].plot(ax=ax, label='2')
        ax.set_title(col)
        ax.set_ylabel("Value")

    plt.xlabel("Index")
    plt.legend()
    plt.tight_layout()

    plt.show()
    
# Simple function to plot loss plot after training
def loss_plot(training_losses, validation_losses=None, testing_losses=None, plot_name=""):
    
    epochs = range(1, len(training_losses) + 1,1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_losses, label='Training Loss', linewidth=2)
    
    if validation_losses is not None:
        plt.plot(epochs, validation_losses, label='Validation Loss', linewidth=2)
    
    if testing_losses is not None:
        plt.plot(epochs, testing_losses, label='Testing Loss', linewidth=2)

    plt.title('Training vs Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0,0.6)
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    # plt.show()
    
    
# Fancy plotting function to plot the predictions across the different datasets
def plot_predictions(training_dataloader, validation_dataloader, model, imu_joint_map, folder_name, col_names):
    
    training_file_name = folder_name + "_training_prediction.png"
    plot_one_df_prediction(training_dataloader, model, training_file_name, imu_joint_map, col_names)
    
    validation_file_name = folder_name + "_validation_prediction.png"
    plot_one_df_prediction(validation_dataloader, model, validation_file_name, imu_joint_map, col_names) 

    
    # file_name = "Plots/testing_prediction.png"
    # plot_one_df_prediction(testing_plot, model, file_name, testing_df.columns, data_keys)
    
    
    


def plot_one_df_prediction(dataloader, model, file_name, imu_joint_map, col_names):
    
    model.eval()
    fig, axs = plt.subplots(3, 8, figsize=(30,10), sharey=True)
    
    key_list = list(imu_joint_map.keys())
    
    step = 20
    end_plot_timestep = 450
    with torch.no_grad():
        
        for j, batch in enumerate(islice(dataloader, 0, end_plot_timestep, step)):
            
            start_index = j*step
            
            end_index = start_index + 301
            
            # Transpose and convert to tensor
            input_tensor = batch

            output,_,_,_ = model(input_tensor)
        
            output = output.squeeze(0)
            input = input_tensor.squeeze(0)
    
            output_df = pd.DataFrame(output.T.numpy())
            input_df = pd.DataFrame(input.T.numpy())
            
            for i in range(24):
                row = i % 3
                col = i // 3
                ax = axs[row, col]
            
                
                
                
                if end_index>=end_plot_timestep: 
                    
                    value_over = end_index - end_plot_timestep
                    stop_plot = len(output_df) - value_over
                    ax.plot(dataloader.dataset.indices[start_index:end_plot_timestep],output_df.iloc[:stop_plot,i], linewidth=1, alpha=0.75)
                    
                    # Plot the ground truth
                    # TODO - Right now it plots lines on top of each other - Only plot once
                    ax.plot(dataloader.dataset.indices[start_index:end_plot_timestep], input_df.iloc[:stop_plot, i], linewidth=1, color="black")  # Plot the i-th column
                
                else:
                    ax.plot(dataloader.dataset.indices[start_index:end_index],output_df.iloc[:,i], linewidth=1, alpha=0.75)
                    
                     # Plot the ground truth
                     # TODO - Right now it plots lines on top of each other - Only plot once
                    ax.plot(dataloader.dataset.indices[start_index:end_index], input_df.iloc[:end_index, i], linewidth=1, color="black")  # Plot the i-th column
                
                
                ax.set_ylim(-10,8)
                
                # Name of the Sub Plot 
                
                if j==0:
                    joint_name = None
                    prefix = col_names[i][:4]
                    
                    for k in key_list:
                        
                        if prefix in k:
                            joint_name = imu_joint_map[k]
                            break

                    if "_l" in joint_name:
                        joint_name = joint_name.replace("_l", "")
                        joint_name = "left " + joint_name
                    elif "_r" in joint_name:
                        joint_name = joint_name.replace("_r", "")
                        joint_name = "right " + joint_name

                    
                    name = joint_name + " (" + col_names[i] + ")"
                    ax.set_title(name)
                    ax.tick_params(labelsize=8)
            
            
    fig.suptitle(file_name)

    plt.tight_layout()
    # plt.show()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')



# Function to calculate the PAE validation loss
def cal_validation_loss(model, validation_dataloader, lossFn, lossFn_no_reduction):
    model.eval()
    
    val_loss = 0.0
    individual_losses = np.zeros(24, dtype=np.float32)
    
    # Convoluted Signal
    all_latents = []
    
    # Reconstructed signal
    all_signals = []
    
    # model_input = torch.tensor([])
    # model_prediction = torch.tensor([])
    
    with torch.no_grad():
        for batch in validation_dataloader:
            
            PAE_inputs = batch
            PAE_inputs = ToDevice(PAE_inputs)
                        
            # Predict
            outputs, latent, signal, params  = model(PAE_inputs)
                    
            # Flattening the outputs and inputs and calculating the loss
            flattened_inputs = PAE_inputs.reshape(PAE_inputs.shape[0], -1)
            flattened_outputs = outputs.reshape(outputs.shape[0], -1)
        
            # Calculate Total Loss
            loss = lossFn(flattened_inputs, flattened_outputs)
            
            # Calculate running loss
            val_loss += loss.item() * PAE_inputs.size(0)
            
            # Calculate Individual Loss, first across sequence length and then across batches
            individual_loss = lossFn_no_reduction(PAE_inputs, outputs)
            individual_loss_across_sequence_length = individual_loss.mean(2)
            individual_loss_across_batch = individual_loss_across_sequence_length.mean(0)
            
            individual_losses = individual_losses + ( Item(individual_loss_across_batch).numpy() * PAE_inputs.size(0))
            
            
            # Append all the latents and signals
            all_latents.append(Item(latent))
            all_signals.append(Item(signal))
            
        
        val_loss = val_loss / len(validation_dataloader.dataset)
        individual_losses = individual_losses / len(validation_dataloader.dataset)   
          
        latents_np = torch.cat(all_latents, dim=0).numpy()  
        signals_np = torch.cat(all_signals, dim=0).numpy()      
        
        
    return val_loss, individual_losses, latents_np, signals_np

# Function to calculate the PAE validation loss
def cal_validation_loss_future_prediction(model, PAE_model, validation_dataloader, lossFn, lossFn_no_reduction):
    model.eval()
    PAE_model.eval()
    
    val_loss = 0.0
    individual_losses = np.zeros(6, dtype=np.float32)
    
    with torch.no_grad():
        for batch in validation_dataloader:
            
            PAE_inputs, FCNN_inputs, FCNN_outputs = batch
            PAE_inputs = ToDevice(PAE_inputs)
                        
            # Predict
            _, _, _, params  = PAE_model(PAE_inputs)
            
            flattened_inputs = ToDevice(FCNN_inputs.reshape(FCNN_inputs.shape[0], -1))


            phaseInputs = torch.flatten(torch.stack(params, dim=1),1,2) 
            phaseInputs = phaseInputs[:,:,-1]
            
            FCNN_combine_inputs = torch.cat((flattened_inputs, phaseInputs), dim=1)
            
            # Forwards pass
            y_pred = model(FCNN_combine_inputs)
            
            # Calculate the loss
            # weightedLoss = weightedMSELossFunction(ypred, groundTruth, conditions)
            loss = lossFn(y_pred,ToDevice(FCNN_outputs))

            
            # Calculate running loss
            val_loss += loss.item() * PAE_inputs.size(0)
            
            # Calculate Individual Loss, first across sequence length and then across batches
            individual_loss = lossFn_no_reduction(ToDevice(FCNN_outputs), y_pred)
            individual_loss_across_batch = individual_loss.mean(0)

            individual_losses = individual_losses + ( Item(individual_loss_across_batch).numpy() * FCNN_inputs.size(0))
            

        val_loss = val_loss / len(validation_dataloader.dataset)
        individual_losses = individual_losses / len(validation_dataloader.dataset)   
   
        
        
    return val_loss, individual_losses


# Plot a numpy array with every individual component on a subplot
def plot_conv_deconv(latents_np, signals_np, config, file_name):
    
    
    fig, axes = plt.subplots(latents_np.shape[1], 1, figsize=(8, 12), sharex='col')
    
    # print(latents_np.shape)
    plot_conv_numpy_array(axes, latents_np, signals_np, config)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    # plt.show()


def plot_conv_numpy_array(axes, latents_numpy_array, signals_numpy_array, config):
    
    
    # Generate a np array of indices
    rng = np.random.default_rng()
    random_start = rng.integers(low=0, high=(latents_numpy_array.shape[0] - config["seq_length"]))
    plot_timesteps = 2000
    
    x = np.linspace(0,latents_numpy_array.shape[0],latents_numpy_array.shape[0])
    

    
    # Iterate over the entire array
    end = random_start+plot_timesteps - 301
    plotting_freq = 20
    for j in range(random_start,end, plotting_freq):
    
        for i in range(latents_numpy_array.shape[1]):    
            
     
            axes[i].plot(x[j:j+config["seq_length"]], latents_numpy_array[j,i,:], linewidth=1, color="black")
            axes[i].plot(x[j:j+config["seq_length"]], signals_numpy_array[j,i,:], linewidth=1, color="blue")
            axes[i].set_ylim(-15,15)
            
            
            if j==random_start:
                axes[i].set_title(f"Channel_{i}")
                # axes[i].legend()


## Plotting function - Heavily study the function
class PlottingWindow():
    def __init__(self, title, ax=None, min=None, max=None, cumulativeHorizon=100, drawInterval=100):
        plt.ion()
        _, self.ax = plt.subplots() if ax is None else ax
        self.Title = title
        self.CumulativeHorizon = cumulativeHorizon
        self.DrawInterval = drawInterval
        self.YMin = min
        self.YMax = max
        self.YRange = [sys.float_info.max if min==None else min, sys.float_info.min if max==None else max]
        self.Functions = {} #string->[History, Horizon]
        self.Counter = 0

    def Add(self, *args): #arg->(value, label)
        for arg in args:
            value = arg[0]
            label = arg[1]
            if label not in self.Functions:
                self.Functions[label] = ([],[])
            function = self.Functions[label]
            function[0].append(value)
            function[1].append(sum(function[0][-self.CumulativeHorizon:]) / len(function[0][-self.CumulativeHorizon:]))

            self.YRange[0] = min(self.YRange[0], value) if self.YMin==None else self.YRange[0]
            self.YRange[1] = max(self.YRange[1], value) if self.YMax==None else self.YRange[1]

        self.Counter += 1
        if self.Counter >= self.DrawInterval:
            self.Counter = 0
            self.Draw()

    def Draw(self):
        self.ax.cla()
        self.ax.set_title(self.Title)
        for label in self.Functions.keys():
            function = self.Functions[label]
            step = max(int(len(function[0])/self.DrawInterval), 1)
            self.ax.plot(function[0][::step], label=label + " (" + str(round(self.CumulativeValue(label), 3)) + ")")
            self.ax.plot(function[1][::step], c=(0,0,0))
        self.ax.set_ylim(self.YRange[0], self.YRange[1])
        self.ax.legend()
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(1e-5)

    def Value(self, label=None):
        if label==None:
            return sum(x[0][-1] for x in self.Functions.values())
        else:
            return self.Functions[label][0][-1]

    def CumulativeValue(self, label=None):
        if label==None:
            return sum(x[1][-1] for x in self.Functions.values())
        else:
            return self.Functions[label][1][-1]
