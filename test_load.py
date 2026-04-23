import sys
sys.path.append('.')
import pandas as pd
from slam import SLAM

slam = SLAM('icp', 'sensordata/static_imu1_converted.csv', 'sensordata/static_lidar1_converted.csv')
try:
    slam.get_imu_data()
    print("IMU data loaded successfully")
except Exception as e:
    print("Error in get_imu_data:", e)

try:
    slam.get_lidar_data()
    print("Lidar data loaded successfully")
except Exception as e:
    print("Error in get_lidar_data:", e)