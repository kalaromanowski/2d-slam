import pandas as pd
import numpy as np

df = pd.read_csv('static_lidar1.csv')
df.rename(columns={'timestamp': 'field.header.stamp'}, inplace=True)

# Create new df
new_df = pd.DataFrame()
new_df['%time'] = df['field.header.stamp']  # dummy
new_df['field.header.seq'] = 0
new_df['field.header.stamp'] = df['field.header.stamp']
for i in range(8):
    new_df[f'dummy_{i}'] = 0

# Now add 682 range columns
num_ranges = 682
dist_cols = [f'dist_{i}.0' for i in range(360)]
for i in range(num_ranges):
    if i < 360:
        # Assume distances are in mm, convert to m
        new_df[f'range_{i}'] = df[dist_cols[i]] / 1000.0
    else:
        new_df[f'range_{i}'] = 0  # pad with 0

new_df.to_csv('static_lidar1_converted.csv', index=False)
print('Converted lidar file saved as static_lidar1_converted.csv')