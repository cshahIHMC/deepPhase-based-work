######## Author - Chinmay Shah ##############
''' 
This script contains all the functions needed often to either transfer to the GPU,
plot stuff or writing or processing data, misc helper functions
'''
import torch
import pandas as pd
import matplotlib.pyplot as plt

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
def loss_plot(training_losses, validation_losses, testing_losses, name):
    
    epochs = range(1, len(training_losses) + 1,1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_losses, label='Training Loss', linewidth=2)
    plt.plot(epochs, validation_losses, label='Validation Loss', linewidth=2)
    # plt.plot(epochs, testing_losses, label='Testing Loss', linewidth=2)

    plt.title('Training vs Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0,0.6)
    plt.savefig(name, dpi=300, bbox_inches='tight')
    # plt.show()
    
    
# Fancy plotting function to plot the predictions across the different datasets
def plot_predictions(training_df, validation_df, model, data_keys, folder_name):
    
    ## Extract the slice of data to plot
    training_plot = training_df.iloc[2531:3431]
    validation_plot = validation_df.iloc[1990:2890]
    # testing_plot = testing_df.iloc[1025:1625]
    
    file_name = "Plots/" + folder_name + "_training_prediction.png"
    plot_one_df_prediction(training_plot, model, file_name, training_df.columns, data_keys )
    
    file_name = "Plots/" + folder_name + "_validation_prediction.png"
    plot_one_df_prediction(validation_plot, model, file_name, validation_df.columns, data_keys) 

    
    # file_name = "Plots/testing_prediction.png"
    # plot_one_df_prediction(testing_plot, model, file_name, testing_df.columns, data_keys)
    
    
    


def plot_one_df_prediction(df, model, file_name, col_names, data_keys):
    
    model.eval()
    fig, axs = plt.subplots(3, 8, figsize=(30,10))
    
    key_list = list(data_keys.keys())

    
    
    step_increments = 20
    end_plot_timestep = 450
    with torch.no_grad():
        
        for j in range(0,end_plot_timestep,step_increments):
            
            start_index = j
            end_index = j+301
        
            input = df.iloc[start_index:end_index]
        

            # Transpose and convert to tensor
            input_tensor = torch.tensor(input.T.values, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0)

            output,_,_,_ = model(input_tensor)
        
            output = output.squeeze(0)
    
            output_df = pd.DataFrame(output.T.numpy())
            
            for i in range(24):
                row = i % 3
                col = i // 3
                ax = axs[row, col]
            
                if j==0: ax.plot(df.iloc[:end_plot_timestep, i], linewidth=2, color="black")  # Plot the i-th column
                
                if end_index>=end_plot_timestep: 
                    
                    value_over = end_index - end_plot_timestep
                    stop_plot = len(output_df) - value_over
                    ax.plot(df.index[start_index:end_plot_timestep],output_df.iloc[:stop_plot,i], linewidth=1, alpha=0.75)
                else:
                    ax.plot(df.index[start_index:end_index],output_df.iloc[:,i], linewidth=1, alpha=0.75)
                
                ax.set_ylim(-10,8)
                
                # Name of the Sub Plot 
                
                if j==0:
                    joint_name = None
                    prefix = col_names[i][:4]
                    
                    for k in key_list:
                        
                        if prefix in k:
                            joint_name = data_keys[k]
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
