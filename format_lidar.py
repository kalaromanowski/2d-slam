import pandas as pd
import numpy as np

def format_lidar_data(input_file, output_file):
    """
    Format lidar_out-2.csv to match exp1_lidar.csv format
    
    Input: 360 distance + 360 confidence columns
    Output: 682 range columns in ROS LaserScan format
    """
    # Read the input file
    df = pd.read_csv(input_file)
    
    # Extract distance and confidence columns
    dist_cols = [col for col in df.columns if col.startswith('dist_')]
    conf_cols = [col for col in df.columns if col.startswith('conf_')]
    
    # Sort to ensure proper order
    dist_cols.sort(key=lambda x: float(x.split('_')[1]))
    conf_cols.sort(key=lambda x: float(x.split('_')[1]))
    
    # ROS LaserScan parameters (from exp1_lidar.csv)
    angle_min = -2.08621382713
    angle_max = 2.09234976768
    angle_increment = 0.00613592332229
    range_min = 0.019999999553
    range_max = 5.59999990463
    n_ranges = 682  # Number of range measurements in ROS format
    
    # Create output dataframe
    output_rows = []
    
    for idx, row in df.iterrows():
        # Get the distance values (360 points at 1 deg intervals)
        distances = row[dist_cols].values
        
        # Interpolate to 682 points
        # Create mapping from 360 points to 682 points
        x_old = np.linspace(0, 359, 360)
        x_new = np.linspace(0, 359, n_ranges)
        ranges_interp = np.interp(x_new, x_old, distances)
        
        # Create output row with ROS LaserScan format
        output_row = {
            '%time': int(row['timestamp'] * 1e9),  # Convert to nanoseconds
            'field.header.seq': idx,
            'field.header.stamp': int(row['timestamp'] * 1e9),
            'field.header.frame_id': 'laser',
            'field.angle_min': angle_min,
            'field.angle_max': angle_max,
            'field.angle_increment': angle_increment,
            'field.time_increment': 9.76562732831e-05,
            'field.scan_time': 0.10000000149,
            'field.range_min': range_min,
            'field.range_max': range_max,
        }
        
        # Add range measurements
        for i in range(n_ranges):
            output_row[f'field.ranges{i}'] = ranges_interp[i]
        
        output_rows.append(output_row)
    
    # Create output dataframe
    output_df = pd.DataFrame(output_rows)
    
    # Build column order
    columns_order = [
        '%time', 'field.header.seq', 'field.header.stamp', 'field.header.frame_id',
        'field.angle_min', 'field.angle_max', 'field.angle_increment', 
        'field.time_increment', 'field.scan_time', 'field.range_min', 'field.range_max'
    ]
    
    # Add all range columns
    for i in range(n_ranges):
        columns_order.append(f'field.ranges{i}')
    
    output_df = output_df[columns_order]
    
    # Write to output file
    output_df.to_csv(output_file, index=False)
    print(f"Successfully formatted {input_file} -> {output_file}")
    print(f"Rows converted: {len(output_df)}")
    print(f"Range measurements per scan: {n_ranges}")

if __name__ == '__main__':
    input_file = 'sensordata/lidar_out-2.csv'
    output_file = 'sensordata/lidar_out-2_formatted.csv'
    format_lidar_data(input_file, output_file)
