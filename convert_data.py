#!/usr/bin/env python3
"""
Script to convert IMU and LiDAR data files to the format expected by the SLAM ICP algorithm.

Usage: python convert_data.py <imu_input.csv> <lidar_input.csv>

This will create <imu_input_converted.csv> and <lidar_input_converted.csv> in the same directory.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to path for rotations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rotations import Quaternion

def convert_imu(imu_file):
    """Convert IMU file from Euler angles to quaternions."""
    df = pd.read_csv(imu_file)
    # Assume roll, pitch, yaw are in radians
    quats = []
    for i, row in df.iterrows():
        q = Quaternion(euler=[row['roll'], row['pitch'], row['yaw']])
        quats.append([q.x, q.y, q.z, q.w])

    quats = np.array(quats)
    df['field.orientation.x'] = quats[:,0]
    df['field.orientation.y'] = quats[:,1]
    df['field.orientation.z'] = quats[:,2]
    df['field.orientation.w'] = quats[:,3]
    df['field.angular_velocity.x'] = df['gyro_x']
    df['field.angular_velocity.y'] = df['gyro_y']
    df['field.angular_velocity.z'] = df['gyro_z']
    df['field.linear_acceleration.x'] = df['accel_x']
    df['field.linear_acceleration.y'] = df['accel_y']
    df['field.linear_acceleration.z'] = df['accel_z']
    # Rename timestamp to field.header.stamp
    df.rename(columns={'timestamp': 'field.header.stamp'}, inplace=True)
    # Keep only the needed columns
    df = df[['field.header.stamp', 'field.orientation.x', 'field.orientation.y', 'field.orientation.z', 'field.orientation.w', 'field.angular_velocity.x', 'field.angular_velocity.y', 'field.angular_velocity.z', 'field.linear_acceleration.x', 'field.linear_acceleration.y', 'field.linear_acceleration.z']]
    return df

def convert_lidar(lidar_file):
    """Convert LiDAR file to expected format for 360-degree scans."""
    df = pd.read_csv(lidar_file)
    df.rename(columns={'timestamp': 'field.header.stamp'}, inplace=True)

    # Extract the 360 distance values
    dist_cols = [f'dist_{i}.0' for i in range(360)]
    
    # Create new df with interpolated data for each timestamp
    new_df = pd.DataFrame()
    new_df['%time'] = df['field.header.stamp']  # dummy
    new_df['field.header.seq'] = 0
    new_df['field.header.stamp'] = df['field.header.stamp']
    for i in range(8):
        new_df[f'dummy_{i}'] = 0

    # Interpolate each row from 360 points to 682 points (both covering full 360 degrees)
    num_ranges = 682
    interpolated_data = []
    
    for idx, row in df.iterrows():
        # Get the original 360 range values in meters
        ranges_original = row[dist_cols].values / 1000.0
        
        # Original angles span full 360 degrees (0 to 2*pi)
        angles_original = np.linspace(0, 2*np.pi, 360, endpoint=False)
        
        # Target angles span full 360 degrees with 682 points
        angles_target = np.linspace(0, 2*np.pi, num_ranges, endpoint=False)
        
        # Interpolate to denser set of angles
        ranges_interpolated = np.interp(angles_target, angles_original, ranges_original, period=2*np.pi)
        interpolated_data.append(ranges_interpolated)
    
    interpolated_data = np.array(interpolated_data)
    
    # Add interpolated range columns
    for i in range(num_ranges):
        new_df[f'range_{i}'] = interpolated_data[:, i]
    return new_df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_data.py <imu_input.csv> <lidar_input.csv>")
        sys.exit(1)
    
    imu_file = sys.argv[1]
    lidar_file = sys.argv[2]
    
    if not os.path.exists(imu_file):
        print(f"IMU file {imu_file} does not exist.")
        sys.exit(1)
    if not os.path.exists(lidar_file):
        print(f"LiDAR file {lidar_file} does not exist.")
        sys.exit(1)
    
    convert_imu(imu_file)
    convert_lidar(lidar_file)