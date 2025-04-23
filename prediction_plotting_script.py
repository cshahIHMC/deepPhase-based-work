


from Library import utility
import pandas as pd
import numpy as np
from Models import PAE
import torch


## Data Setup - Setup all the parameters for logging and training
config = {
    "training_tag": "PAE - sensor suit Walking Data - 300 time horizon - 6 phase Channels Conv - 24x16xphase",
    "project_name": "PAE - Sensor suit Walking Data",
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


csv_path = "/home/cshah/workspaces/deepPhase based work/Data/04_21_2025_walking_data.csv"

# Read the csv file into a pandas data frame
data = pd.read_csv(csv_path)
print(len(data))


# range_indices = list(range(0, 10700)) + list(range(17000, len(data)))


training_df = data.iloc[0:179580].reset_index(drop=True)
validation_df = data.iloc[179580:].reset_index(drop=True)
# testing_df = pd.read_csv(testing_csv_path)
# testing_df = 

# Checking the Size of the data frame
print("Training DF size:", training_df.shape)
print("Validation DF size:", validation_df.shape)
# print("Testing DF size:", testing_df.shape)

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
# extracted_testing_df = utility.extract_data(testing_df, data_keys)



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
    # extracted_testing_df[col] = extracted_testing_df[col] * np.pi / 180

print("Validation DF size:", extracted_validation_df.shape)


#################### Code to plot the predictions ######################3

model_file = "/home/cshah/workspaces/deepPhase based work/Saved Models/20250422_1447_PAE - sensor suit Walking Data - 300 time horizon - 8 phase Channels 24X16Xphase_channels convolution .pth"

weights = torch.load(model_file, weights_only=True)
model = PAE.Model(
                          input_channels=config["inputs"],
                          embedding_channels=config["phases"],
                          intermediate_channels=config["intermediate_channels"],
                          time_range=config["seq_length"],
                          window=config["training_window"]
                         )
model.load_state_dict(weights)




utility.plot_predictions(extracted_training_df, extracted_validation_df, model, col_keys, folder_name=config["training_tag"])


####################################################################################
