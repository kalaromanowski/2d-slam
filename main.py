import sys
from slam import SLAM

if __name__ == "__main__":
    # Verify user input arguments and initialize input data
    use_ground_truth = True
    
    try:
        # Check if --no-ground-truth flag is provided
        if "--no-ground-truth" in sys.argv:
            use_ground_truth = False
            sys.argv.remove("--no-ground-truth")
        
        assert len(sys.argv) == 4
        algorithm = sys.argv[1].lower()
        imu_path = sys.argv[2]
        lid_path = sys.argv[3]
    except AssertionError:
        n_args = len(sys.argv) - 1
        print("\nExpected number of arguments: 3. Received: {0}. Correct usage of function:".format(n_args))
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data]´´´")
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data] --no-ground-truth´´´\n")
        sys.exit()

    try:
        assert algorithm in ["feature", "icp"]
    except AssertionError:
        print("\nInvalid algorithm choice, choose between 'feature' or 'icp'. Correct usage of function:")
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data]´´´\n")
        sys.exit()

    try:
        slam = SLAM(algorithm, imu_path, lid_path, use_ground_truth=use_ground_truth)
        slam.get_imu_data()
        slam.get_lidar_data()
    except:
        print("\nInvalid filename(s). Correct usage of function:")
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data]´´´\n")
        sys.exit()

    # Visualize ground truth if available
    if use_ground_truth:
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
    
    # Print deviation errors if ground truth is available
    if use_ground_truth:
        print("Deviation errors\n" + 50*"=")
        rmse_traj = slam.RMSE_traj
        if rmse_traj is not None:
            print("RMSE for trajectory estimate:", rmse_traj, "m")
        if algorithm == "icp":
            rmse_wall = slam.RMSE_wall_icp
            if rmse_wall is not None:
                print("RMSE for mapping estimate:", rmse_wall, "m")
        if algorithm == "feature":
            rmse_wall = slam.RMSE_wall_feature
            if rmse_wall is not None:
                print("RMSE for mapping estimate:", rmse_wall, "m")
        print("Plotting trajectory error\n" + 50*"=")
        slam.plot_traj_error()
