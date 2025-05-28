import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

#general rotation matrices
def get_R_x(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def get_R_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def get_R_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R

# Load quaternion data from CSV
def load_quaternion_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


# Transforms to go anatomical frame to sensor frame - all fo these are extrinsic transformations going right to left
def build_transforms_2():
    transforms = {}
    
    # Anatomical 2 Sensor Frames
    transforms["Anatomical_2_pelvis"] = R.from_matrix( get_R_z(np.pi) @ get_R_y(-np.pi/2) )
    transforms["Anatomical_2_back"] = R.from_matrix( get_R_z(np.pi) @ get_R_y(-np.pi/2) )
    
    transforms["Anatomical_2_thigh_r"] = R.from_matrix( get_R_z(-np.pi/2) @ get_R_y(-np.pi/2) )
    transforms["Anatomical_2_thigh_l"] = R.from_matrix( get_R_z(np.pi/2) @ get_R_y(-np.pi/2) )
    
    transforms["Anatomical_2_shank_r"] = R.from_matrix( get_R_z(-np.pi/2) @ get_R_y(-np.pi/2) )
    transforms["Anatomical_2_shank_l"] = R.from_matrix( get_R_z(np.pi/2) @ get_R_y(-np.pi/2) )
    
    
    # Pelvis 2 Sensor Frames
    
    transforms["pelvis_2_pelvis"] = R.from_matrix( np.eye(3)) 
    
    transforms["pelvis_2_thigh_r"] = R.from_matrix( get_R_x(np.pi/2) )
    transforms["pelvis_2_thigh_l"] = R.from_matrix( get_R_x(-np.pi/2) )
    
    transforms["pelvis_2_shank_r"] = R.from_matrix( get_R_x(np.pi/2) )
    transforms["pelvis_2_shank_l"] = R.from_matrix( get_R_x(-np.pi/2) )
    
    # transforms["pelvis_2_foot"] = R.from_matrix( get_R_x(np.pi) @ get_R_y(np.pi/2) )
    transforms["pelvis_2_foot"] = R.from_matrix( get_R_y(np.pi/2) )
    
    return transforms

# Function to extract the imu and insole data and return a list with the entire data and a tpose list
def extract_data(data, joint_imu_map_microstrain, joint_imu_map_insole):
    
    # Extract t_pose_q for microstrain
    t_pose_q = {limb : np.array(eval(data[f"{imu}"].iloc[0])) for limb, imu in joint_imu_map_microstrain.items()}
    
    # Extract quat for microstrain
    quat_data = {limb: np.stack(data[f"{imu}"].apply(eval).values) for limb, imu in joint_imu_map_microstrain.items()}
    
    # Extract Insole quaternion data
    for limb, data_key in joint_imu_map_insole.items():
        
        cols = [data_key+"_qw", data_key+"_qx" , data_key+"_qy" , data_key+"_qz" ]
        
        quat_cols = data[cols]
        quat_data[limb] = quat_cols.to_numpy()
        t_pose_q[limb] = quat_cols.to_numpy()[0]
        
    return quat_data, t_pose_q
        
        
        