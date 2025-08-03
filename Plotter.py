#################### FUnction to plot all the quaternion data ################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def normalize_np_quat(q):
    # Assume preds is your (50000, 7, 4) array of quaternions
    norm = np.linalg.norm(q, axis=-1, keepdims=True)  # shape: (50000, 7, 1)

    # Avoid divide-by-zero
    norm[norm == 0] = 1.0

    # Normalize
    q_norm = q / norm
    
    return q_norm   

def joint_angle_calculator(np_array):
    
    euler_angles = np.zeros((np_array.shape[0] , np_array.shape[1] // 4, 3))
    
    for i in range(np_array.shape[0]):
        for j in range(0, np_array.shape[1], 4):
            
            q = np_array[i, j:j+4]
            r = R.from_quat(q)
            euler = r.as_euler('zyx', degrees=True)
            euler_angles[i, j//4, :] = euler
            
    return euler_angles
            
df = pd.read_csv("/home/cshah/workspaces/deepPhase based work/Data/Full Training - Gyro + Joint Angles/Quat_Sub2_train.csv")

ss_df = df.iloc[:, 0:28]
cap_df = df.iloc[:, 28:56]


ss_angles = joint_angle_calculator( normalize_np_quat(ss_df.to_numpy()) )
cap_angles = joint_angle_calculator( normalize_np_quat(cap_df.to_numpy()) )


print(ss_angles.shape)


fig, axes = plt.subplots(cap_angles.shape[2], 7, figsize=(20,12), sharex=True)

joints = ss_df.columns
print(cap_angles.shape)

for i in range(cap_angles.shape[1]):
        
        # for j in range(len(pred_euler_angles)):
        #     sum = pred_euler_angles[j,i,0] ** 2 + pred_euler_angles[j,i,1] ** 2 + pred_euler_angles[j,i,2] ** 2 + pred_euler_angles[j,i,3] ** 2
        #     sum2 = ground_truth_euler_angles[j,i,0] ** 2 + ground_truth_euler_angles[j,i,1] ** 2 + ground_truth_euler_angles[j,i,2] ** 2 + ground_truth_euler_angles[j,i,3] ** 2
            
            # print("Sum 1: ", sum)
            # print("Sum 2: ", sum2)
            
        # Combine to compute display-centered y-axis limits
        combined = np.concatenate([cap_angles, ss_angles], axis=0)  # shape (2N, 3)

        
        axes[0,i].plot( cap_angles[:,i,0], linewidth=1, color="red")
        axes[0,i].plot( ss_angles[:,i,0], linewidth=0.75, color="black")
        axes[0,i].set_title(joints[i] + "_X ")
    
        
        axes[1,i].plot( cap_angles[:,i,1], linewidth=1, color="green")
        axes[1,i].plot( ss_angles[:,i,1], linewidth=0.75, color="black")
        axes[1,i].set_title(joints[i] + "_Y ")
        
        axes[2,i].plot( cap_angles[:,i,2], linewidth=1, color="blue")
        axes[2,i].plot( ss_angles[:,i,2], linewidth=0.75, color="black")
        axes[2,i].set_title(joints[i] + "_Z ")
        
        # if pred_euler_angles.shape[2] >= 4:
        #     for k in range(3, pred_euler_angles.shape[2]):
        #         axes[k,i].plot( pred_euler_angles[:,i,k], linewidth=1, color="blue")
        #         axes[k,i].plot( ground_truth_euler_angles[:,i,k], linewidth=0.75, color="black")
                
                
        #         axes[k,i].set_title(joints[i] + "_W ")
        
        
        # axes[0,i].set_ylim(-1.5,1.5)
        # axes[1,i].set_ylim(-1.5,1.5)
        # axes[2,i].set_ylim(-1.5,1.5)

plt.tight_layout()
    # plt.savefig(plot_save_name, dpi=300, bbox_inches='tight')
    # plt.close()
plt.show()
    