import sys
from slam import SLAM

if __name__ == "__main__":
    # Verify user input arguments and initialize input data
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    
    
    try:
        assert len(args) == 3
        algorithm = args[0].lower()
        imu_path = args[1]
        lid_path = args[2]
    except AssertionError:
        n_args = len(args)
        print("\nExpected number of arguments: 3. Received: {0}. Correct usage of function:".format(n_args))
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data]´´´\n")
        print("Options:")

        sys.exit()

    try:
        assert algorithm in ["feature", "icp"]
    except AssertionError:
        print("\nInvalid algorithm choice, choose between 'feature' or 'icp'. Correct usage of function:")
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data]´´´\n")
        sys.exit()

    slam = SLAM(algorithm, imu_path, lid_path)
    slam.get_imu_data()
    slam.get_lidar_data()

    try:
        slam = SLAM(algorithm, imu_path, lid_path)
        slam.get_imu_data()
        slam.get_lidar_data()
    except:
        print("\nInvalid filename(s). Correct usage of function:")
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data]´´´\n")
        sys.exit()
    
    # Processing data
    slam.set_params()
    slam.initialize_arrays()
    slam.ekf()
    slam.postprocess()

    # Displaying results
    print("Plotting results\n" + 50*"=")
    slam.plot_results()
