import numpy as np
import matplotlib.pyplot as plt
import csv

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = [np.eye(6) for _ in range(10)]  # Filter state covariance matrices for multiple targets
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time, sig_r, sig_a, sig_e_sqr):
        # Initialize state and covariance matrices
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Filtered_Time = time
        
        # Initialize covariance matrix R
        self.R = np.array([[sig_r**2 * np.cos(sig_e_sqr) * np.sin(sig_a)**2 + sig_r**2 * np.cos(sig_e_sqr) * np.cos(sig_a)**2 + sig_a**2 + sig_r**2 * np.sin(sig_e_sqr)**2 * np.sin(sig_a)**2 * sig_e_sqr, 0, 0],
                           [0, sig_r**2 * np.cos(sig_e_sqr) * np.cos(sig_a)**2 + sig_r**2 * np.cos(sig_e_sqr) * np.sin(sig_a)**2 + sig_a**2 + sig_r**2 * np.sin(sig_e_sqr)**2 * np.cos(sig_a)**2 * sig_e_sqr, 0],
                           [0, 0, sig_r**2 * np.cos(sig_e_sqr) * np.sin(sig_a)**2 + sig_r**2 * np.cos(sig_e_sqr) * np.cos(sig_a)**2 + sig_a**2 + sig_r**2 * np.sin(sig_e_sqr)**2 * np.sin(sig_a)**2 * sig_e_sqr]])
        
        # Assign R to each element of pf
        for i in range(10):
            self.pf[i] = self.R

    def predict_state_covariance(self, delt, plant_noise):
        # Predict state covariance for each target
        predicted_covariances = []
        for cov_matrix in self.pf:
            Phi = np.eye(6)
            Phi[0, 3] = delt
            Phi[1, 4] = delt
            Phi[2, 5] = delt
            Pp = np.dot(np.dot(Phi, cov_matrix), Phi.T) + plant_noise
            predicted_covariances.append(Pp)
        return predicted_covariances

    def Filter_state_covariance(self, measurements_sets):
        # Initialize lists to store filtered states and innovations
        filtered_states_list = []
        innovations_list = []
        
        # Process measurements and get filtered states and innovations at each time step
        for measurements in measurements_sets:
            Sp, Pp, predicted_Time = self.predict_state_covariance(measurements[0][3] - self.Meas_Time, np.eye(6))
            filtered_states, innovations = self.filter_states(measurements, Sp, Pp)
            filtered_states_list.append(filtered_states)
            innovations_list.append(innovations)
        
        return filtered_states_list, innovations_list

    def filter_states(self, measurements, Sp, Pp):
        # Initialize lists to store filtered states and innovations for multiple targets
        filtered_states = []
        innovations = []
        
        # Iterate over each potential measurement set
        for measurement in measurements:
            # Calculate Kalman gain for each target
            S = self.R + np.dot(np.dot(np.eye(3, 6), Pp), np.eye(3, 6).T)
            K = np.dot(np.dot(Pp, np.eye(3, 6).T), np.linalg.inv(S))

            # Update state for each target
            Inn = measurement[0:3] - np.dot(np.eye(3, 6), Sp)
            Sf = Sp + np.dot(K, Inn)
            Pf = Pp - np.dot(np.dot(K, np.eye(3, 6)), Pp)

            # Store filtered state and innovation for each target
            filtered_states.append(Sf)
            innovations.append(Inn)
        
        return filtered_states, innovations

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            rng1 = float(row[10])  # Measurement range
            az = float(row[11])    # Measurement azimuth
            el = float(row[12])    # Measurement elevation
            time = float(row[13]) # Measurement time
            measurements.append((rng1, az, el, time))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define initial state estimates
x = 0  # Initial x position
y = 0  # Initial y position
z = 0  # Initial z position
vx = 0  # Initial velocity in x direction
vy = 0  # Initial velocity in y direction
vz = 0  # Initial velocity in z direction
initial_time = 0  # Initial time

# Define noise parameters
sig_r = 30
sig_a = 0.005  # 5 milliradians
sig_e_sqr = 0.005  # 5 milliradians

# Initialize the filter with initial state estimates and noise parameters
kalman_filter.Initialize_Filter_state_covariance(x, y, z, vx, vy, vz, initial_time, sig_r, sig_a, sig_e_sqr)

# Define the path to your CSV file containing measurements
csv_file_path = 'data_57.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Split measurements into sets (assuming each set contains measurements for a specific time)
measurements_sets = []
current_time = measurements[0][3]
current_set = []
for measurement in measurements:
    if measurement[3] == current_time:
        current_set.append(measurement)
    else:
        measurements_sets.append(current_set)
        current_time = measurement[3]
        current_set = [measurement]
measurements_sets.append(current_set)

# Filter state covariance for each set of measurements
filtered_states_list, innovations_list = kalman_filter.Filter_state_covariance(measurements_sets)

# Print filtered states and innovations
for i, (filtered_states, innovations) in enumerate(zip(filtered_states_list, innovations_list)):
    print(f"Set {i+1}:")
    for j, (state, innovation) in enumerate(zip(filtered_states, innovations)):
        print(f"Target {j+1} - Filtered State: {state}, Innovation: {innovation}")

# Plotting measured vs predicted values
plt.figure(figsize=(12, 8))

for measurement_set in measurements_sets:
    measured_range = [measurement[0] for measurement in measurement_set]
    measured_azimuth = [measurement[1] for measurement in measurement_set]
    measured_elevation = [measurement[2] for measurement in measurement_set]
    measurement_times = [measurement[3] for measurement in measurement_set]
    plt.subplot(3, 1, 1)
    plt.plot(measurement_times, measured_range, label='Measured Range')
    plt.xlabel('Measurement Time')
    plt.ylabel('Range')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(measurement_times, measured_azimuth, label='Measured Azimuth')
    plt.xlabel('Measurement Time')
    plt.ylabel('Azimuth')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(measurement_times, measured_elevation, label='Measured Elevation')
    plt.xlabel('Measurement Time')
    plt.ylabel('Elevation')
    plt.legend()

plt.tight_layout()
plt.show()
