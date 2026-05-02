# Class file for SLAM based on IMU and LiDAR
# Accuracy Estimation of 2D SLAM Using Sensor Fusion of LiDAR and INS
# Author: Jan Xu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from rotations import Quaternion, skew_symmetric
from feature_scan import ScanFeature, update_state, update_map
from icp import GlobalFrame, wraptopi
import time
from convert_data import convert_imu, convert_lidar

class SLAM:

    ###################
    ### Constructor ###
    ###################

    def __init__(self, algorithm, imu_file, lid_file):
        """
        Instantiates a SLAM object based on IMU and LiDAR input data, specifying
        a LiDAR scan matching algorithm.

        Args:
            algorithm (str):
                String describing the scan matching algorithm. Choose between
                ´icp´ or ´feature´.
            imu_file (str):
                Path string to IMU dataset (.csv-file).
            lid_file (str):
                Path string to LiDAR dataset (.csv-file).
        """

        self.algorithm = algorithm
        self.imu_file = imu_file
        self.lid_file = lid_file


    ########################
    ### Instance methods ###
    ########################

    def get_imu_data(self, stby=700, offset_quat=True, gravity_bias_est=False):
        """
        Loads, cleans and assigns IMU data to object attributes.

        Args:
            stby (int):
                Number of initial IMU readings (standby readings)
                before first movement. Default: 700.
            offset_quat (bool):
                If True, the quaternion readings will be offset such that
                initial heading is zero. Default: True.
            gravity_bias_est (bool):
                If True, the initial bias estimation, including estimating
                gravity component in the z-direction, will be done by averaging
                across the first ´stby´ number of IMU readings. Default: False
        """
        imu = convert_imu(self.imu_file)
        imu = imu.loc[:,["field.header.stamp",
                                                  "field.orientation.x",
                                                  "field.orientation.y",
                                                  "field.orientation.z",
                                                  "field.orientation.w",
                                                  "field.angular_velocity.x",
                                                  "field.angular_velocity.y",
                                                  "field.angular_velocity.z",
                                                  "field.linear_acceleration.x",
                                                  "field.linear_acceleration.y",
                                                  "field.linear_acceleration.z"]]
        # imu = pd.read_csv(self.imu_file, usecols=["field.header.stamp",
        #                                           "field.orientation.x",
        #                                           "field.orientation.y",
        #                                           "field.orientation.z",
        #                                           "field.orientation.w",
        #                                           "field.angular_velocity.x",
        #                                           "field.angular_velocity.y",
        #                                           "field.angular_velocity.z",
        #                                           "field.linear_acceleration.x",
        #                                           "field.linear_acceleration.y",
        #                                           "field.linear_acceleration.z"])
        imu.columns = ["timestamp", "q_x", "q_y", "q_z", "q_w", "om_x", "om_y", "om_z", "a_x", "a_y", "a_z"]

        # Assign arrays for timestamp, linear acceleration and rotational velocity
        self.imu_t = np.round(imu.timestamp.values / 1e9, 3) # converting to seconds with three decimal places
        self.imu_f = imu[["a_x", "a_y", "a_z"]].values
        self.imu_w = imu[["om_x", "om_y", "om_z"]].values
        self.imu_q = imu[["q_w", "q_x", "q_y", "q_z"]].values
        self.n = self.imu_f.shape[0] # number of IMU readings

        if offset_quat:
            # Transform quaternions to Euler angles
            phi = np.ndarray((self.n, 3))
            for i in range(self.n):
                phi[i,:] = Quaternion(*self.imu_q[i,:]).normalize().to_euler()

            # Shift yaw angle such that the inital yaw equals zero
            inityaw = np.mean(phi[:stby,2])       # Estimated initial yaw
            tf_yaw = wraptopi(phi[:,2] - inityaw) # Transformed yaw

            # Transform Euler angles back to quaternions
            phi[:,2] = tf_yaw
            for i in range(self.n):
                self.imu_q[i,:] = Quaternion(euler=phi[i,:]).normalize().to_numpy()

        if gravity_bias_est:
            imu_f_trans = np.ndarray((stby, 3))
            for i in range(stby):
                C_ns = Quaternion(*self.imu_q[i,:]).normalize().to_mat()
                imu_f_trans[i] = C_ns.dot(self.imu_f[i,:])

                self.g = np.mean(imu_f_trans,0) # bias + gravity
        else:
            self.g = np.array([0, 0, 9.80665]) # gravity


    def get_lidar_data(self):
        """
        Loads and assigns LiDAR data to object attributes.

        """

        #lid = pd.read_csv(self.lid_file, usecols=[2] + list(range(11, 693)))
        lid = convert_lidar(self.lid_file)
        lid = lid.iloc[1:,[2] + list(range(11, 693))]

        # Assign arrays for timestamp and range data
        self.lid_t = np.round(lid["field.header.stamp"].values / 1e9, 3) # converting to seconds with three decimal places
        self.lid_r = lid.iloc[:,1:].values


    def set_params(self, imu_f=0.05, imu_w=10*np.pi/180,
                   imu_bias_f=0.0005, imu_bias_w=0.01*np.pi/180,
                   lid_p=0.01, lid_r=0.5*np.pi/180):
        """
        Sets standard deviation parameters and constructs sensor covariance
        matrices Q_imu (process noise covariance) and R_lid (measurement noise
        covariance).

        Args:
            imu_f (float):
                Standard deviation associated with uncertainty of inertial
                accelerometer sensor in [m/s^2]. Default: 0.05 m/s^2.
            imu_w (float):
                Standard deviation associated with uncertainty of inertial
                gyroscope sensor in [rad/s]. Default: 1.745e-1 rad/s ≈ 10°/s.
            imu_bias_f (float):
                Standard deviation associated with uncertainty of the bias in
                inertial accelerometer sensor in [m/s^2]. Default: 0.0005 m/s^2.
            imu_bias_w (float):
                Standard deviation associated with uncertainty of the bias in
                inertial gyroscope sensor in [rad/s].
                Default: 1.745e-4 rad/s ≈ 0.01°/s.
            lid_p (float):
                Standard deviation associated with uncertainty of position
                estimate from LiDAR scan matching algorithm in [m].
                Default: 0.01 m.
            lid_r (float):
                Standard deviation associated with uncertainty of orientation
                estimate from LiDAR scan matching algorithm in [rad].
                Default: 8.727e-3 rad ≈ 0.5°.
        """

        # Process noise attributed to odometry
        std_imu_f = 0.05                # [m/s^2]
        std_imu_w = 10*np.pi/180        # [rad/s]
        std_imu_bias_f = 0.0005         # [m/s^2]
        std_imu_bias_w = 0.01*np.pi/180 # [rad/s]

        # Measurement noise attributed to LiDAR scans
        std_lid_p = 0.01          # [m] x and y position
        std_lid_r = 0.5*np.pi/180 # [rad] yaw angle

        # Constructs covariance arrays
        self.Q_imu = np.diag(np.array([std_imu_f]*3 + [std_imu_w]*3 + [std_imu_bias_f]*3 + [std_imu_bias_w]*3))**2
        self.R_lid = np.diag([std_lid_p, std_lid_p, std_lid_r])**2



    def initialize_arrays(self):
        """
        Initializes state and covariance arrays. Initial covariance standard
        deviations for the system state variables are pre-established here;
        however, feel free to modify these directly in the procedure.

        """

        self.p_est = np.zeros([self.n, 3])      # position estimates
        self.v_est = np.zeros([self.n, 3])      # velocity estimates
        self.q_est = np.zeros([self.n, 4])      # orientation estimates as quaternions
        self.del_u_est = np.zeros([self.n, 6])  # sensor bias estimates
        self.p_cov = np.zeros([self.n, 15, 15]) # covariance array at each timestep
        self.lidar_states = np.zeros([self.n, 3]) # position and heading state associated with each LiDAR scan

        # Set initial values
        self.p_est[0] = np.zeros((1,3))
        self.v_est[0] = np.zeros((1,3))
        self.q_est[0] = self.imu_q[0,:]
        self.del_u_est[0] = np.zeros((1,6))
        self.lidar_states[0] = np.zeros((1,3))

        # Initial uncertainties in standard deviations
        p_cov_p = 0.02*np.ones(3,)                # [m] position
        p_cov_v = 0.01*np.ones(3,)                # [m/s] velocity
        p_cov_q = np.array([1, 1, 20])*np.pi/180  # [rad] roll, pitch and yaw angles
        p_cov_bias_f = 0.02*np.ones(3,)           # [m/s^2] accelerometer biases
        p_cov_bias_w = 0.05*np.ones(3,)*np.pi/180 # [rad/s] gyroscope biases

        self.p_cov[0] = np.diag(np.hstack([p_cov_p, p_cov_v, p_cov_q, p_cov_bias_f, p_cov_bias_w]))**2


    def initialize_lidar(self):
        """
        Initializes global navigation frame with the point cloud from the first
        LiDAR scan, and also initializes the index of the next LiDAR scan and
        the position and heading state estimate associated with each LiDAR scan
        (important for sensor fusion -- see ´measurement_update´.)

        Returns:
            lid_i (int):
                Index of the next LiDAR scan to be processed.
            lid_state [3x0 (1D) Numpy array]:
                Initial position and heading state estimate associated with
                LiDAR scans.
        """

        # Initialize global navigation frame with first LiDAR scan
        if self.algorithm == "icp":
            self.gf = GlobalFrame(self.lid_r)
        if self.algorithm == "feature":
            self.gf = ScanFeature(self.lid_r[0,:], frame="global")

        lid_i = 1                                    # index of next LiDAR scan
        lid_state = np.hstack([self.p_est[0,:2], 0]) # current position and heading state estimate

        return lid_i, lid_state


    def ekf(self):
        """
        Main loop of the extended Kalman filter sensor fusion procedure.

        """

        lid_i, lid_state = self.initialize_lidar()

        t = self.imu_t[0]  # initialize time step
        st = round(time.time()) # initialize runtime

        print("Initiating Kalman filter loop...")
        for k in range(1, self.n):

            # Progress report
            if k % 100 == 0:
                tm_new = round(time.time())
                tm_el = tm_new - st
                tm_rm = (self.n-k)*tm_el//k
                print("Iteration {0}/{1}: Time elapsed {2}m{3}sec - Est. time remaining {4}m{5}sec".format(k, self.n, tm_el//60, tm_el%60, tm_rm//60, tm_rm%60))

            delta_t = (self.imu_t[k] - self.imu_t[k-1])

            # Assign previous state
            p_km = self.p_est[k-1].reshape(3,1)
            v_km = self.v_est[k-1].reshape(3,1)
            q = self.imu_q[k,:].reshape(4,1)
            del_u_km = self.del_u_est[k-1].reshape(6,1)
            quat = Quaternion(*q).normalize()
            C_ns = quat.to_mat()

            # Assign control inputs and calibrate with estimated sensor error
            f_km = self.imu_f[k-1].reshape(3,1) - del_u_km[:3]
            w_km = self.imu_w[k-1].reshape(3,1) - del_u_km[3:]

            # Update state
            p_check = p_km + delta_t * v_km + delta_t**2/2 * (C_ns.dot(f_km) - self.g.reshape(3,1))
            v_check = v_km + delta_t * (C_ns.dot(f_km) - self.g.reshape(3,1))
            q_check = q
            del_u_check = del_u_km

            # Linearize motion model
            F, L = SLAM.state_space_model(f_km, C_ns, delta_t)

            # Propagate uncertainty
            p_cov_km = self.p_cov[k-1,:,:]
            p_cov_check = F.dot(p_cov_km).dot(F.T) + L.dot(self.Q_imu).dot(L.T)

            # Check availability of LIDAR measurements
            if lid_i != len(self.lid_r) and t >= self.lid_t[lid_i]:

                if self.algorithm == "icp":
                    y_k = SLAM.icp_state(self.gf, self.lid_r[lid_i], lid_state)
                if self.algorithm == "feature":
                    y_k = SLAM.feature_state(self.lid_r[lid_i-1], self.lid_r[lid_i], lid_state)

                new_state = SLAM.measurement_update(y_k, p_check, v_check,
                                                    q_check, del_u_check,
                                                    p_cov_check, self.R_lid)
                p_check, v_check, q_check, p_cov_check, del_u_check, lid_state = new_state

                if self.algorithm == "feature":
                    x_hat = p_check[0,0]
                    y_hat = p_check[1,0]
                    yaw_hat = Quaternion(*q_check).normalize().to_euler()[2,0]
                    update_map(self.gf, self.lid_r[lid_i], x_hat, y_hat, yaw_hat)

                lid_i += 1

            # Store into state and uncertainty arrays
            self.p_est[k,:] = p_check.reshape(3,)
            self.v_est[k,:] = v_check.reshape(3,)
            self.q_est[k,:] = q_check.reshape(4,)
            self.del_u_est[k,:] = del_u_check.reshape(6,)
            self.p_cov[k,:,:] = p_cov_check
            self.lidar_states[k,:] = y_k.reshape(3,)

            # Increment time
            t += delta_t

        # Report finished progress
        tm_el = round(time.time()) - st
        print("Finished iterating after {0}m{1}sec".format(tm_el//60, tm_el%60))


    def postprocess(self):
        """
        Post-process the SLAM results.

        """

        if self.algorithm == "icp":
            self.pc_t = np.array([-self.gf.y_pc , self.gf.x_pc ]).T

        if self.algorithm == "feature":
            # Remove unsufficiently matched line segments
            rem_lines = []
            for i, line in enumerate(self.gf.lines):
                if line.it < 100:
                    rem_lines.append(line)

            for line in rem_lines:
                self.gf.lines.remove(line)

            # Transform LiDAR scans in global reference frame
            for line in self.gf.lines:
                line.x_start, line.y_start = -line.y_start , line.x_start
                line.x_end, line.y_end = -line.y_end , line.x_end

        # Transform final state estimates
        self.p_est[:,:2] = (np.array([[0,-1],[1,0]]).dot(self.p_est[:,:2].T) ).T
        self.v_est[:,:2] = np.array([[0,-1],[1,0]]).dot(self.v_est[:,:2].T).T


    def plot_results(self):
        """
        Plots the final SLAM results against the ground truth.

        """

        fig, ax = plt.subplots(figsize=(10,8))

        if self.algorithm == "icp":
            # Plot all of the point cloud for complete visualization
            x_plot = self.pc_t[:,0]*-1
            y_plot = self.pc_t[:,1]
            ax.plot(x_plot, y_plot, 'm.', markersize=1, label="Point cloud of walls")

        if self.algorithm == "feature":
            for i, line in enumerate(self.gf.lines):
                if i == 0:
                    ax.plot([line.x_start, line.x_end], [line.y_start, line.y_end], 'm--', linewidth=3, label="Estimated walls")
                else:
                    ax.plot([line.x_start, line.x_end], [line.y_start, line.y_end], 'm--', linewidth=3)


        reconstructed_path = np.cumsum(self.gf.T, axis=0)*[1,-1]
        ax.plot(reconstructed_path[:,1], reconstructed_path[:,0]*-1, 'b-', lw=3, label="Estimated Trajectory", zorder=5)
        
        # Auto-scale axis limits to include all data
        all_x = np.concatenate([x_plot if self.algorithm == "icp" else [0], reconstructed_path[:,1]])
        all_y = np.concatenate([y_plot if self.algorithm == "icp" else [0], reconstructed_path[:,0]*-1])
        
        x_margin = (np.max(all_x) - np.min(all_x)) * 0.1
        y_margin = (np.max(all_y) - np.min(all_y)) * 0.1
        ax.set_xlim(np.min(all_x) - x_margin, np.max(all_x) + x_margin)
        ax.set_ylim(np.min(all_y) - y_margin, np.max(all_y) + y_margin)
        
        plt.legend(loc="upper right", fontsize=9)
        plt.grid(alpha=0.5)
        graphname = "Map 3, Trial 4"
        ax.set_title("SLAM Results for "+ graphname)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.show()

        fig, ax = plt.subplots(figsize=(10,8))
        avg_error = np.sum(self.gf.d)/self.gf.d.shape[0]
        ax.plot(self.gf.d, 'r-', lw=3, label="Sum of squared distances between matched points")
        ax.set_title("ICP Matching Error for "+ graphname + " (Average error: {:.2f} m^2)".format(avg_error))
        ax.set_xlabel("LiDAR scan index")
        ax.set_ylabel("Sum of Squared Distances")
        plt.show()


    #####################
    ### Class methods ###
    #####################

    def feature_state(r_prev, r_new, prev_lid_state):
        """
        Obtains the position and orientation state of the system based on the
        feature-based (line) scan matching algorithm and LiDAR scans.

        Args:
            r_prev [1D Numpy array]:
                Range readings of the point cloud from the LiDAR scan in the
                previous time step.
            r_new [1D Numpy array]:
                Range readings of the point cloud from the LiDAR scan in the
                current time step.
            prev_lid_state [3x0 (1D) Numpy array]:
                1D Numpy array consisting of the x and y position and heading
                of the previous measurement update.

        Returns:
            y_k [1x3 Numpy array]:
                Position (x and y-coordinate) and orientation state of the
                system.
        """

        # Unpack state from previous LiDAR scan
        x_prev = prev_lid_state[0]
        y_prev = prev_lid_state[1]
        head_prev = prev_lid_state[2]

        # Calculate estimated state from LiDAR scans using line feature algorithm
        x_lid, y_lid, head_lid = update_state(r_prev, r_new, x_prev, y_prev, head_prev)
        y_k = np.array([x_lid, y_lid, head_lid]).reshape(3,1)
        return y_k


    def icp_state(gf, r_new, prev_lid_state):
        """
        Obtains the position and orientation state of the system based on the
        ICP algorithm and LiDAR scans.

        Args:
            gf <GlobalFrame object>:
                Instance of GlobalFrame (specified in `icp`) representing the
                global navigation frame of the SLAM problem.
            r_new [1D Numpy array]:
                Range readings of the point cloud from the LiDAR scan in the
                current time step.
            prev_lid_state [3x0 (1D) Numpy array]:
                1D Numpy array consisting of the x and y position and heading
                of the previous measurement update.

        Returns:
            y_k [1x3 Numpy array]:
                Position (x and y-coordinate) and orientation state of the
                system.
        """

        # Unpack state from previous LiDAR scan
        prev_lid_pose = prev_lid_state[:2]
        head_prev = prev_lid_state[2]

        # Calculate estimated state from LiDAR scans using ICP algorithm
        x_lid, y_lid, head_lid = gf.next_scan(r_new, prev_lid_pose, head_prev)
        y_k = np.array([x_lid, y_lid, head_lid]).reshape(3,1)
        return y_k

    def measurement_update(y_k, p_check, v_check, q_check, del_u_check,
                           p_cov_check, sensor_var):
        """
        Updates the system state with measurement correction from LiDAR scan
        matching procedure.

        Args:
            y_k [1x3 Numpy array]:
                Position (x and y-coordinate) and orientation state of the
                system.
            p_check [3x1 Numpy array]:
                Estimated position state of the current time step.
            v_check [3x1 Numpy array]:
                Estimated velocity state of the current time step.
            q_check [4x1 Numpy array]:
                Estimated orientation state (in quaternion representation) of
                the current time step.
            del_u_check [6x1 Numpy array]:
                Estimated bias state of accelerometer and gyroscope of the
                current time step.
            p_cov_check [15x15 Numpy array]:
                Estimated state covariance matrix of the current time step.
            sensor_var [3x3 Numpy array]:
                Sensor covariance matrix of the LiDAR sensor.

        Returns:
            p_hat [3x1 Numpy array]:
                Corrected (sensor fused) position state.
            v_hat [3x1 Numpy array]:
                Corrected (sensor fused) velocity state.
            q_hat [4x1 Numpy array]:
                Corrected (sensor fused) orientation state (in quaternion
                representation).
            p_cov_check [15x15 Numpy array]:
                Corrected (sensor fused) state covariance matrix.
            del_u_hat [6x1 Numpy array]:
                Corrected (sensor fused) bias state of accelerometer and
                gyroscope.
            new_lid_state [3x0 (1D) Numpy array]:
                1D Numpy array consisting of the x and y position and heading
                of the new measurement update.
        """

        # Load measurement model Jacobian matrix H
        H = SLAM.H()

        # Compute Kalman gain
        K_k = p_cov_check.dot(H.T).dot(inv(H.dot(p_cov_check).dot(H.T) + sensor_var))

        # Bundle up state
        phi_check = Quaternion(*q_check).normalize().to_euler() # convert to Euler angles
        X_check = np.vstack((p_check[:2], phi_check[2])) # bundle up state into one single array

        # Calculate change of state
        dx_k = K_k.dot(y_k - X_check)

        # Unpack change of state into components
        dp_k = dx_k[:3]
        dv_k = dx_k[3:6]
        dphi_k = dx_k[6:9]
        d_u_check = dx_k[9:]

        # Correct predicted state
        p_hat = p_check + dp_k
        v_hat = v_check + dv_k
        q_hat = Quaternion(euler=dphi_k).normalize().quat_mult(q_check)
        del_u_hat = del_u_check + d_u_check

        # Compute corrected covariance
        p_cov_hat = (np.eye(15) - K_k.dot(H)).dot(p_cov_check)

        # Collate new LiDAR state estimate
        yaw_hat = phi_check[2] + dphi_k[2]
        new_lid_state = np.hstack([p_hat[:2].reshape(2,), yaw_hat])

        return p_hat, v_hat, q_hat, p_cov_hat, del_u_hat, new_lid_state


    def state_space_model(imu_f, C_ns, delta_t):
        """
        Given state inputs and time step increment, this function returns the
        state transition matrix F and the noise gain matrix L required for the
        linearized state representation.

        Args:
            imu_f [3x1 Numpy array]:
                Original specific force vector from inertial accelerometer in
                [m/s^2] of the current time step.
            C_ns [3x3 Numpy array]:
                Direction cosine matrix that resolves the current orientation
                state to the navigation frame.
            delta_t (float):
                Time increment in [s] between the current time step and the
                previous one.

        Returns:
            F [15x15 Numpy array]:
                State transition matrix of the current time step.
            L [15x12 Numpy array]:
                Noise gain matrix of the current time step.
        """

        F = np.eye(15)
        F[:3,3:6] = np.eye(3)*delta_t
        F[3:6,6:9] = skew_symmetric(C_ns.dot(imu_f))*delta_t
        F[3:6,9:12] = C_ns*delta_t
        F[6:9,12:] = -C_ns*delta_t

        # L: noise gain matrix
        L = np.zeros([15,12])
        L[3:6,:3] = C_ns*delta_t
        L[6:9,3:6] = C_ns*delta_t
        L[9:,6:] = np.eye(6)

        return F, L


    def H():
        """
        Returns measurement model Jacobian matrix H.

        Returns:
            H [3x15 array]: measurement model Jacobian matrix.
        """

        H = np.zeros([3, 15])
        H[:3,:3] = np.eye(3)
        return H
