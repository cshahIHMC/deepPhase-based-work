import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.opengl as gl
import sys
import MotionAnalyzer_utility as Utility

class IMUMotionAnalyzer:
    """
    Class to load IMU and force data, transform quaternions to an anatomical frame,
    compute joint angles, and plot the results.
    """
    
    def __init__(self, csv_path: str):
        # File path for the sensor data
        self.csv_path = csv_path
        # Build transforms between frames
        self.body_transforms = Utility.build_transforms_2()
        # Limb structure and joint hierarchy mappings
        self.limb_structure, self.joint_hierarchy = self._set_limb_structure()
        # IMU-to-joint mappings
        (
            self.joint_imu_map,
            self.joint_imu_map_microstrain,
            self.joint_imu_map_insole
        ) = self._get_joint_imu_map()
        
    @staticmethod
    def _set_limb_structure():
        limb_structure = {
            "pelvis_l": ("pelvis", "pelvis_l"),
            "pelvis_r": ("pelvis", "pelvis_r"),
            "thigh_l": ("pelvis_l", "thigh_l"),
            "thigh_r": ("pelvis_r", "thigh_r"),
            "shank_l": ("thigh_l", "shank_l"),
            "shank_r": ("thigh_r", "shank_r"),
            "foot_l": ("shank_l", "foot_l"),
            "foot_r": ("shank_r", "foot_r"),
            "back": ("pelvis", "back"),
        }
        joint_hierarchy = {
            "pelvis": "pelvis",
            "thigh_r": "pelvis",
            "thigh_l": "pelvis",
            "shank_r": "thigh_r",
            "shank_l": "thigh_l",
            "foot_r": "shank_r",
            "foot_l": "shank_l",
        }
        return limb_structure, joint_hierarchy

    @staticmethod
    def _get_joint_imu_map():
        joint_imu_map = {
            "pelvis": "imu2_quat",
            "thigh_r": "imu5_quat",
            "shank_r": "imu6_quat",
            "thigh_l": "imu1_quat",
            "shank_l": "imu4_quat",
            "foot_r": "R_insole",
            "foot_l": "L_insole",
        }
        joint_imu_map_microstrain = {
            k: v for k, v in joint_imu_map.items() if "insole" not in v
        }
        joint_imu_map_insole = {
            k: v for k, v in joint_imu_map.items() if "insole" in v
        }
        return joint_imu_map, joint_imu_map_microstrain, joint_imu_map_insole
    
    def load_data(self):
        # Load full dataframe
        data = Utility.load_quaternion_data(csv_path=self.csv_path)
        # Extract force and quaternion channels
        force_cols = ["L_insole_force", "R_insole_force"]
        self.force_data = data[force_cols]
        # Extract IMU quaternions and T-pose quats
        self.quaternion_data, self.t_pose_quat = Utility.extract_data(
            data=data,
            joint_imu_map_microstrain=self.joint_imu_map_microstrain,
            joint_imu_map_insole=self.joint_imu_map_insole
        )
        
    def transform_quaternions(self):
        
        t_pose_q_norm = { imu: R.from_quat(q / np.linalg.norm(q), scalar_first=True) for imu, q in self.t_pose_quat.items() }
        
        transformed = {}
        
        for imu, raw in self.quaternion_data.items():
            # Normalize each sample
            mag = np.linalg.norm(raw, axis=1, keepdims=True)
            q_norm = R.from_quat(raw / mag, scalar_first=True)
            
            if "foot" in imu:
                # Zero relative to pelvis then to anatomical frame
                zeroed = (
                    t_pose_q_norm["pelvis"]
                    * self.body_transforms["pelvis_2_foot"]
                    * q_norm
                    * t_pose_q_norm[imu].inv()
                    * self.body_transforms["pelvis_2_foot"].inv()
                )
                rel_rot = (
                    self.body_transforms["Anatomical_2_pelvis"]
                    * t_pose_q_norm["pelvis"].inv()
                    * zeroed
                    * self.body_transforms["Anatomical_2_pelvis"].inv()
                )
            else:
                frame = "Anatomical_2_" + imu
                rel_rot = (
                    self.body_transforms[frame]
                    * t_pose_q_norm[imu].inv()
                    * q_norm
                    * self.body_transforms[frame].inv()
                )
            transformed[imu] = rel_rot
        self.transformed_quats = transformed

    def calculate_joint_angles(self):
        joint_angles = {}
        for child, parent in self.joint_hierarchy.items():
            if child == "pelvis":
                joint_q = self.transformed_quats[child]
            else:
                joint_q = (
                    self.transformed_quats[parent].inv()
                    * self.transformed_quats[child]
                )
            rotvec = joint_q.as_rotvec()
            angle_rad = np.linalg.norm(rotvec)
            angle_deg = np.degrees(angle_rad)
            if angle_rad > 1e-8:
                axis = rotvec / angle_rad
                angle_x = angle_deg * axis[0]
                angle_y = angle_deg * axis[1]
                angle_z = angle_deg * axis[2]
            else:
                axis = np.zeros(3)
                angle_x = angle_y = angle_z = 0.0
            joint_angles[child] = {
                "angle_deg": angle_deg,
                "axis": axis,
                "angle_x": angle_x,
                "angle_y": angle_y,
                "angle_z": angle_z,
            }
        self.joint_angles = joint_angles
        
    def get_joint_angles(self):
        pass
        
        
        
    def analyze(self):
        self.load_data()
        self.transform_quaternions()
        self.calculate_joint_angles()
        
        return get_joint_angles()
        