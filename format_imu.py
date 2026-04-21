import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (radians) to quaternion [x, y, z, w]"""
    r = R.from_euler('xyz', [roll, pitch, yaw])
    return r.as_quat()  # Returns [x, y, z, w]

def format_imu_data(input_file, output_file):
    """
    Format imu_out-2.csv to match exp1_imu.csv format
    """
    # Read the input file
    df = pd.read_csv(input_file)
    
    # Create output dataframe
    output_rows = []
    
    for idx, row in df.iterrows():
        # Convert Euler angles to quaternion
        quat = euler_to_quaternion(row['roll'], row['pitch'], row['yaw'])
        qx, qy, qz, qw = quat
        
        # Create ROS IMU message format row
        output_row = {
            '%time': int(row['timestamp'] * 1e9),  # Convert to nanoseconds
            'field.header.seq': idx,
            'field.header.stamp': int(row['timestamp'] * 1e9),
            'field.header.frame_id': 'base_link',
            'field.orientation.x': qx,
            'field.orientation.y': qy,
            'field.orientation.z': qz,
            'field.orientation.w': qw,
            'field.orientation_covariance0': 1.0,
            'field.orientation_covariance1': 0.0,
            'field.orientation_covariance2': 0.0,
            'field.orientation_covariance3': 0.0,
            'field.orientation_covariance4': 1.0,
            'field.orientation_covariance5': 0.0,
            'field.orientation_covariance6': 0.0,
            'field.orientation_covariance7': 0.0,
            'field.orientation_covariance8': 1.0,
            'field.angular_velocity.x': row['gyro_x'],
            'field.angular_velocity.y': row['gyro_y'],
            'field.angular_velocity.z': row['gyro_z'],
            'field.angular_velocity_covariance0': 1.21846967915e-07,
            'field.angular_velocity_covariance1': 0.0,
            'field.angular_velocity_covariance2': 0.0,
            'field.angular_velocity_covariance3': 0.0,
            'field.angular_velocity_covariance4': 1.21846967915e-07,
            'field.angular_velocity_covariance5': 0.0,
            'field.angular_velocity_covariance6': 0.0,
            'field.angular_velocity_covariance7': 0.0,
            'field.angular_velocity_covariance8': 1.21846967915e-07,
            'field.linear_acceleration.x': row['accel_x'],
            'field.linear_acceleration.y': row['accel_y'],
            'field.linear_acceleration.z': row['accel_z'],
            'field.linear_acceleration_covariance0': 9e-08,
            'field.linear_acceleration_covariance1': 0.0,
            'field.linear_acceleration_covariance2': 0.0,
            'field.linear_acceleration_covariance3': 0.0,
            'field.linear_acceleration_covariance4': 9e-08,
            'field.linear_acceleration_covariance5': 0.0,
            'field.linear_acceleration_covariance6': 0.0,
            'field.linear_acceleration_covariance7': 0.0,
            'field.linear_acceleration_covariance8': 9e-08,
        }
        output_rows.append(output_row)
    
    # Create output dataframe with proper column order
    output_df = pd.DataFrame(output_rows)
    
    # Reorder columns to match exp1_imu.csv
    columns_order = [
        '%time', 'field.header.seq', 'field.header.stamp', 'field.header.frame_id',
        'field.orientation.x', 'field.orientation.y', 'field.orientation.z', 'field.orientation.w',
        'field.orientation_covariance0', 'field.orientation_covariance1', 'field.orientation_covariance2',
        'field.orientation_covariance3', 'field.orientation_covariance4', 'field.orientation_covariance5',
        'field.orientation_covariance6', 'field.orientation_covariance7', 'field.orientation_covariance8',
        'field.angular_velocity.x', 'field.angular_velocity.y', 'field.angular_velocity.z',
        'field.angular_velocity_covariance0', 'field.angular_velocity_covariance1', 'field.angular_velocity_covariance2',
        'field.angular_velocity_covariance3', 'field.angular_velocity_covariance4', 'field.angular_velocity_covariance5',
        'field.angular_velocity_covariance6', 'field.angular_velocity_covariance7', 'field.angular_velocity_covariance8',
        'field.linear_acceleration.x', 'field.linear_acceleration.y', 'field.linear_acceleration.z',
        'field.linear_acceleration_covariance0', 'field.linear_acceleration_covariance1', 'field.linear_acceleration_covariance2',
        'field.linear_acceleration_covariance3', 'field.linear_acceleration_covariance4', 'field.linear_acceleration_covariance5',
        'field.linear_acceleration_covariance6', 'field.linear_acceleration_covariance7', 'field.linear_acceleration_covariance8',
    ]
    
    output_df = output_df[columns_order]
    
    # Write to output file
    output_df.to_csv(output_file, index=False)
    print(f"Successfully formatted {input_file} -> {output_file}")
    print(f"Rows converted: {len(output_df)}")

if __name__ == '__main__':
    input_file = 'sensordata/imu_out-2.csv'
    output_file = 'sensordata/imu_out-2_formatted.csv'
    format_imu_data(input_file, output_file)
