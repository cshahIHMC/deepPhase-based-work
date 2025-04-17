### Author - Chinmay Shah #######

## This file is responsible for reading all the data from specified csv's and writing it to one giant csv file


import pandas as pd 


## All of the files write to a master CSV called -  
# Path where it is saved "/home/cshah/workspaces/deepPhase based work/Data/training_sensor_suit_file"
# Path where it is saved "/home/cshah/workspaces/deepPhase based work/Data/validation_sensor_suit_file"
# csv_write = "/home/cshah/workspaces/deepPhase based work/Data/training_validation_sensor_suit_file.csv"
csv_write = "/home/cshah/workspaces/deepPhase based work/Data/testing_sensor_suit_file.csv"

# CSV's to read from
csv_files =  [
# "/home/cshah/workspaces/sensorsuit/logs/04_09_2025/04_09_2025_trial_2.csv",
# "/home/cshah/workspaces/sensorsuit/logs/04_09_2025/04_09_2025_trial_4.csv"
"/home/cshah/workspaces/sensorsuit/logs/04_09_2025/04_09_2025_trial_1.csv"
# Add more files as needed
]

# Read and concatenate all CSVs
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


# Optional: sort or reset index
combined_df.reset_index(drop=True, inplace=True)


# Write to a single long CSV
combined_df.to_csv(csv_write, index=False)