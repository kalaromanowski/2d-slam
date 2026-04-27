import sys
from slam import SLAM

if __name__ == "__main__":
    # Verify user input arguments and initialize input data
    # Check for optional --no-gt flag
    use_gt = True
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    
    if '--no-gt' in flags:
        use_gt = False
    
    try:
        assert len(args) == 3
        algorithm = args[0].lower()
        imu_path = args[1]
        lid_path = args[2]
    except AssertionError:
        n_args = len(args)
        print("\nExpected number of arguments: 3. Received: {0}. Correct usage of function:".format(n_args))
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data] [--no-gt]´´´\n")
        print("Options:")
        print("  --no-gt    Skip ground truth plotting and RMSE calculations")
        sys.exit()

    try:
        assert algorithm in ["feature", "icp"]
    except AssertionError:
        print("\nInvalid algorithm choice, choose between 'feature' or 'icp'. Correct usage of function:")
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data] [--no-gt]´´´\n")
        sys.exit()

    try:
        slam = SLAM(algorithm, imu_path, lid_path, use_gt=use_gt)
        slam.get_imu_data()
        slam.get_lidar_data()
    except:
        print("\nInvalid filename(s). Correct usage of function:")
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data] [--no-gt]´´´\n")
        sys.exit()

    # Visualize ground truth (optional)
    if use_gt:
        print("Plotting ground truth\n" + 50*"=")
        slam.plot_ground_truth()

    # Processing data
    slam.set_params()
    slam.initialize_arrays()
    slam.ekf()
    slam.postprocess()

    # Displaying results
    print("Plotting results\n" + 50*"=")
    slam.plot_results()
    
    if use_gt:
        print("Deviation errors\n" + 50*"=")
        print("RMSE for trajectory estimate:", slam.RMSE_traj, "m")
        if algorithm == "icp":
            print("RMSE for mapping estimate:", slam.RMSE_wall_icp, "m")
        if algorithm == "feature":
            print("RMSE for mapping estimate:", slam.RMSE_wall_feature, "m")
        print("Plotting trajectory error\n" + 50*"=")
        slam.plot_traj_error()
    else:
        print("Ground truth comparison disabled (--no-gt mode)")
