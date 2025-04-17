######### Author - Chinmay Shah #################


## Imports
import wandb
from Library import utility
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DataLoader.data_loader_seq_loader import dataLoader_seq_loader
from torch.utils.data import Dataset, DataLoader
from Models import PAE
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import torch



## Helper functions 

# Function to calculate the PAE validation loss
def cal_validation_loss(model, validation_dataloader, lossFn, lossFn_no_reduction):
    model.eval()
    
    val_loss = 0.0
    individual_losses = np.zeros(24, dtype=np.float32)
    
    # model_input = torch.tensor([])
    # model_prediction = torch.tensor([])
    
    with torch.no_grad():
        for batch in validation_dataloader:
            
            PAE_inputs = batch
            PAE_inputs = utility.ToDevice(PAE_inputs)
                        
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
            
            individual_losses = individual_losses + ( utility.Item(individual_loss_across_batch).numpy() * PAE_inputs.size(0))
        
        val_loss = val_loss / len(validation_dataloader.dataset)
        individual_losses = individual_losses / len(validation_dataloader.dataset)            
        
        
    return val_loss, individual_losses
            



## Flags
log_wandB = False
save_model = False

## Data Setup - Setup all the parameters for logging and training
config = {
    "training_tag": "PAE - sensor suit - 1 subject GYRO Data",
    "project_name": "PAE - Sensor suit Data",
    "epochs": 25,
    "batch_size": 32,
    "num_workers": 8,
    "momentum":0.9,
    "lr": 1e-4,
    "dropout": 0.0,
    "dataset": "IHMC Senorsuit",
    "seq_length": 151,
    "inputs": 24,
    "outputs": 24,
    "phases": 4,
    "training_window": 1.0, # How many seconds of data you are reviewing
    "data_recorded_rate": 150 # 
}



## Login to weights and biases and setup the data recording run
if log_wandB:
    wandb.login()
    project_name = config["project_name"]
    
## Path to the data file
training_csv_path = "/home/cshah/workspaces/deepPhase based work/Data/training_validation_sensor_suit_file.csv"
testing_csv_path = "/home/cshah/workspaces/deepPhase based work/Data/testing_sensor_suit_file.csv"

# Read the csv file into a pandas data frame
data = pd.read_csv(training_csv_path)


range_indices = list(range(0, 10700)) + list(range(17000, len(data)))


training_df = data.iloc[range_indices].reset_index(drop=True)
validation_df = data.iloc[10700:17000].reset_index(drop=True)
testing_df = pd.read_csv(testing_csv_path)

# Checking the Size of the data frame
print("Training DF size:", training_df.shape)
print("Validation DF size:", validation_df.shape)
print("Testing DF size:", testing_df.shape)

data_keys = {
    "back": "imu3",
    "pelvis": "imu2",
    "thigh_l": "imu1",
    "thigh_r": "imu6", 
    "shank_l": "imu4",
    "shank_r": "imu5",
    "foot_l": "L_insole",
    "foot_r": "R_insole"
}

col_keys = {
    "imu3": "back",
    "imu2": "pelvis",
    "imu1": "thigh_l",
    "imu6": "thigh_r", 
    "imu4": "shank_l",
    "imu5": "shank_r",
    "L_insole": "foot_l",
    "R_insole": "foot_r"
}

extracted_training_df = utility.extract_data(training_df, data_keys)
extracted_validation_df = utility.extract_data(validation_df, data_keys)
extracted_testing_df = utility.extract_data(testing_df, data_keys)



# Plot the extracted data frame to verify the data
# utility.plot_df(extracted_df)

## Data preprocessing
# The xsensor data is recorded in deg/sec while microstrain data is recorded in rad/sec.
# To make everything consistent we convert the xsensor data to rad/sec
col_to_modify = ["R_insole_gyro_x", "R_insole_gyro_y" , "R_insole_gyro_z",
                 "L_insole_gyro_x", "L_insole_gyro_y" , "L_insole_gyro_z"]

for col in col_to_modify:
    extracted_training_df[col] = extracted_training_df[col] * np.pi / 180
    extracted_validation_df[col] = extracted_validation_df[col] * np.pi / 180
    extracted_testing_df[col] = extracted_testing_df[col] * np.pi / 180

print("Validation DF size:", extracted_validation_df.shape)


#################### Code to plot the predictions ######################3

model_file = "/home/cshah/workspaces/deepPhase based work/Saved Models/20250416_1753_PAE - sensor suit - 1 subject GYRO Data.pth"

weights = torch.load(model_file, weights_only=True)
model = PAE.Model(
                          input_channels=config["inputs"],
                          embedding_channels=config["phases"],
                          time_range=config["seq_length"],
                          window=config["training_window"]
                         )
model.load_state_dict(weights)




utility.plot_predictions(extracted_training_df, extracted_validation_df, extracted_testing_df, model, col_keys)


####################################################################################



##################################### Training Code #################################################
## Load the data in a dataset and a dataloader
training_dataset = dataLoader_seq_loader(extracted_training_df, config["seq_length"])
training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

validation_dataset = dataLoader_seq_loader(extracted_validation_df, config["seq_length"])
validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

testing_dataset = dataLoader_seq_loader(extracted_testing_df, config["seq_length"])
testing_dataloader = DataLoader(testing_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

# utility.plot_df(training_dataset.original_df)
# print(training_dataset.mean)
# print(training_dataset.std)
# utility.plot_df(training_dataset.original_df, training_dataset.normalized_df)
# utility.plot_df(validation_dataset.original_df, validation_dataset.normalized_df)


print("Successfully Loaded the Data")

## Dimensions of the batch
for inputs in training_dataloader:
    print("Data Loader Length: ", len(training_dataloader.dataset))
    print("Inputs size:", inputs.size())
    break
#     print("Outputs size:", outputs.size())

print("Model Setup......")

## Initializing the PAE model
# PAE model
print("Initializing a PAE")
model = utility.ToDevice(PAE.Model(
                          input_channels=config["inputs"],
                          embedding_channels=config["phases"],
                          time_range=config["seq_length"],
                          window=config["training_window"]
                         ))

## Initialize wandb

if log_wandB:
    wandb.init(
        project=project_name,
        name= config["training_tag"],
        config=config
    )
    
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
    
    
    val_loss, individual_loss = cal_validation_loss(model, validation_dataloader, lossFn, lossFn_no_reduction)
    validation_losses.append(val_loss)
    individual_losses.append(individual_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
    
    test_loss, individual_test_loss = cal_validation_loss(model, testing_dataloader, lossFn, lossFn_no_reduction)
    testing_losses.append(test_loss)
    individual_test_losses.append(individual_test_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Testing Loss: {test_loss}')

    if log_wandB:
        wandb.log({"train/train_loss": train_loss,
                    "train/epoch": epoch,
                    "val/val_loss": val_loss,
                    "val/epoch":epoch,
                    "test/test_loss": test_loss,
                    "test/epoch":epoch}) 

print('Finished Training')

# Option to plot loss plot
file_name = "Plots/loss_plot.png"
utility.loss_plot(training_losses, validation_losses, testing_losses, file_name)

if log_wandB:
    wandb.finish()
    
if save_model:
    ## Running all the evaluations

    # Form file name to save the training log
    current_datetime = datetime.now()
    # Format the date and time for the filename
    formatted_datetime = current_datetime.strftime('%Y%m%d_%H%M')
        
    ## Folder to put it in
    folder = "Saved Models/"
    fileName = config["training_tag"]
        
    # Create the filename
    fileNameWithDate = folder + formatted_datetime + "_" + fileName 
    fileNameWithDate_pth = fileNameWithDate + ".pth"
    
    # Save the model in a pytorch file and a wandb files   
    torch.save(model.state_dict(), fileNameWithDate_pth)
        
