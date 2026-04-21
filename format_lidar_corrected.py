#!/usr/bin/env python
# Format LIDAR data to match ROS LaserScan format with proper scaling

import pandas as pd
import numpy as np

def format_lidar_data(input_file, output_file):
    """
    Convert LIDAR distance data to ROS LaserScan format with proper scaling.
    
    The raw LIDAR distances are scaled linearly to match the expected range
    for the ICP algorithm (0.1-1.5 meters typical).
    """
    
    # Read input file
    df = pd.read_csv(input_file)
    
    # Extract timestamp and distance columns (360 distance measurements)
    timestamps = df['timestamp'].values
    distances = df.iloc[:, 1:361].values  # 360 distance columns
    
    # Determine scaling factor based on data statistics
    # Raw data ranges from ~32 to ~2376, needs to map to ~0.1-1.5
    # Use linear scaling: (value - min) / (max - min) * (max_target - min_target) + min_target
    min_raw = 32.0
    max_raw = 2376.0
    min_target = 0.1
    max_target = 1.5
    
    # Apply scaling to all distance values
    distances_scaled = (distances - min_raw) / (max_raw - min_raw) * (max_target - min_target) + min_target
    
    # Clip to valid range
    distances_scaled = np.clip(distances_scaled, min_target, max_target)
    
    print(f"Scaled distance range: [{distances_scaled.min():.4f}, {distances_scaled.max():.4f}]")
    print(f"Scaled distance mean: {distances_scaled.mean():.4f}")
    
    # Interpolate from 360 to 682 points per scan
    x_old = np.arange(360)
    x_new = np.linspace(0, 359, 682)
    
    # Create output dataframe
    output_data = []
    
    for i, timestamp in enumerate(timestamps):
        row_dict = {
            '%time': int(timestamp * 1e9),
            'field.header.seq': i,
            'field.header.stamp': int(timestamp * 1e9),
            'field.header.frame_id': 'laser',
            'field.angle_min': -2.08621382713,
            'field.angle_max': 2.09234976768,
            'field.angle_increment': 0.00613592332229,
            'field.time_increment': 9.76562732831e-05,
            'field.scan_time': 0.10000000149,
            'field.range_min': 0.019999999553,
            'field.range_max': 5.59999990463,
        }
        
        # Interpolate the 360 distances to 682 points
        distances_interpolated = np.interp(x_new, x_old, distances_scaled[i])
        
        # Add each range measurement
        for j in range(682):
            row_dict[f'field.ranges{j}'] = distances_interpolated[j]
        
        output_data.append(row_dict)
    
    # Create output dataframe with proper column order
    output_df = pd.DataFrame(output_data)
    
    # Reorder columns: metadata first, then ranges
    column_order = [
        '%time', 'field.header.seq', 'field.header.stamp', 'field.header.frame_id',
        'field.angle_min', 'field.angle_max', 'field.angle_increment',
        'field.time_increment', 'field.scan_time', 'field.range_min', 'field.range_max'
    ]
    range_columns = [f'field.ranges{i}' for i in range(682)]
    column_order.extend(range_columns)
    
    output_df = output_df[column_order]
    
    # Write to CSV
    output_df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully formatted {input_file} -> {output_file}")
    print(f"Rows converted: {len(output_df)}")
    print(f"Range measurements per scan: {len(range_columns)}")


if __name__ == "__main__":
    format_lidar_data("sensordata/lidar_out-2.csv", "sensordata/lidar_out-2_formatted_corrected.csv")
