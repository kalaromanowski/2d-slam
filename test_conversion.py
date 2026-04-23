import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from rotations import Quaternion

# Test convert_imu
df = pd.read_csv('sensordata/static_imu1.csv')
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
df.rename(columns={'timestamp': 'field.header.stamp'}, inplace=True)
df = df[['field.header.stamp', 'field.orientation.x', 'field.orientation.y', 'field.orientation.z', 'field.orientation.w', 'field.angular_velocity.x', 'field.angular_velocity.y', 'field.angular_velocity.z', 'field.linear_acceleration.x', 'field.linear_acceleration.y', 'field.linear_acceleration.z']]
df.to_csv('sensordata/static_imu1_converted.csv', index=False)

# Test convert_lidar with interpolation
df = pd.read_csv('sensordata/static_lidar1.csv')
df.rename(columns={'timestamp': 'field.header.stamp'}, inplace=True)

dist_cols = [f'dist_{i}.0' for i in range(360)]
num_ranges = 682
new_df = pd.DataFrame()
new_df['%time'] = df['field.header.stamp']
new_df['field.header.seq'] = 0
new_df['field.header.stamp'] = df['field.header.stamp']
for i in range(8):
    new_df[f'dummy_{i}'] = 0

interpolated_data = []
for idx, row in df.iterrows():
    ranges_original = row[dist_cols].values / 1000.0
    angles_original = np.linspace(-2*np.pi/3, 2*np.pi/3, 360)
    angles_target = np.linspace(0, 2*np.pi, num_ranges, endpoint=False)
    angles_extended = np.concatenate([angles_original - 2*np.pi, angles_original, angles_original + 2*np.pi])
    ranges_extended = np.concatenate([ranges_original, ranges_original, ranges_original])
    ranges_interpolated = np.interp(angles_target, angles_extended, ranges_extended)
    interpolated_data.append(ranges_interpolated)

interpolated_data = np.array(interpolated_data)
for i in range(num_ranges):
    new_df[f'range_{i}'] = interpolated_data[:, i]

new_df.to_csv('sensordata/static_lidar1_converted.csv', index=False)
print('Conversion complete. Checking column counts...')
imu_cols = len(pd.read_csv('sensordata/static_imu1_converted.csv').columns)
lidar_cols = len(pd.read_csv('sensordata/static_lidar1_converted.csv').columns)
print(f'IMU columns: {imu_cols}')
print(f'LiDAR columns: {lidar_cols}')
