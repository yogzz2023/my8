import numpy as np
import matplotlib.pyplot as plt
import csv

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
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
        
        # Initialize pf matrix
        self.pf = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                k = i % 3
                l = j % 3
                self.pf[i, j] = self.R[k, l]

    def predict_state_covariance(self, delt, plant_noise):
        # Predict state covariance
        Phi = np.eye(6)
        Phi[0, 3] = delt
        Phi[1, 4] = delt
        Phi[2, 5] = delt
        Sp = np.dot(Phi, self.Sf)
        predicted_Time = self.Filtered_Time + delt

        T_3 = (delt * delt * delt) / 3.0
        T_2 = (delt * delt) / 2.0
        Q = np.array([[T_3, 0, 0, T_2, 0, 0],
                      [0, T_3, 0, 0, T_2, 0],
                      [0, 0, T_3, 0, 0, T_2],
                      [T_2, 0, 0, delt, 0, 0],
                      [0, T_2, 0, 0, delt, 0],
                      [0, 0, T_2, 0, 0, delt]])
        Q = np.dot(Q, plant_noise)
        Pp = np.dot(np.dot(Phi, self.pf), Phi.T) + Q
        return Sp, Pp, predicted_Time

    def Filter_state_covariance(self, H, Z, Pp):
        # Filter state covariance
        S = self.R + np.dot(np.dot(H, Pp), H.T)
        K = np.dot(np.dot(Pp, H.T), np.linalg.inv(S))
        Inn = Z - np.dot(H, self.Sf)  # Use self.Sf instead of self.Sp
        self.Sf = self.Sf + np.dot(K, Inn)
        self.pf = Pp - np.dot(np.dot(K, H), Pp)
        self.Filtered_Time = self.Meas_Time
        return self.Sf, Inn

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

# Lists to store measured and predicted values
measured_range = []
measured_azimuth = []
measured_elevation = []
predicted_range = []
predicted_azimuth = []
predicted_elevation = []
measurement_times = []

# Process measurements and get predicted state estimates at each time step
for measurement in measurements:
    Sp, Pp, predicted_Time = kalman_filter.predict_state_covariance(measurement[3] - kalman_filter.Meas_Time, np.eye(6))
    filtered_state, most_likely_measurement = kalman_filter.Filter_state_covariance(np.eye(3, 6), measurement, Pp)
    
    # Append measured and predicted values to lists
    measured_range.append(measurement[0])
    measured_azimuth.append(measurement[1])
    measured_elevation.append(measurement[2])
    predicted_range.append(filtered_state[0][0])
    predicted_azimuth.append(filtered_state[1][0])
    predicted_elevation.append(filtered_state[2][0])
    measurement_times.append(measurement[3])
    
    # Print the predicted azimuth value for each measurement
    print("Predicted Azimuth:", filtered_state[1][0])

# Plotting measured vs predicted values
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(measurement_times, measured_range, label='Measured Range')
plt.plot(measurement_times, predicted_range, label='Predicted Range')
plt.xlabel('Measurement Time')
plt.ylabel('Range')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(measurement_times, measured_azimuth, label='Measured Azimuth')
plt.plot(measurement_times, predicted_azimuth, label='Predicted Azimuth')
plt.xlabel('Measurement Time')
plt.ylabel('Azimuth')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(measurement_times, measured_elevation, label='Measured Elevation')
plt.plot(measurement_times, predicted_elevation, label='Predicted Elevation')
plt.xlabel('Measurement Time')
plt.ylabel('Elevation')
plt.legend()

plt.tight_layout()
plt.show()
