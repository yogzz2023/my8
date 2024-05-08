#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

class CVFilter {
private:
    Matrix<float, 6, 6> Sf; // State covariance matrix
    Matrix<float, 6, 6> Pf; // Process covariance matrix
    Matrix<float, 6, 6> Phi; // State transition matrix
    Matrix<float, 6, 1> Sp; // Predicted state vector
    Matrix<float, 6, 6> Q; // Process noise covariance matrix
    Matrix<float, 6, 6> Pp; // Predicted process covariance matrix
    Matrix<float, 3, 6> H; // Measurement matrix
    Matrix<float, 3, 1> Z; // Measurement vector
    Matrix<float, 3, 3> R; // Measurement noise covariance matrix
    Matrix<float, 6, 3> K; // Kalman gain
    Matrix<float, 3, 1> Inn; // Innovation vector
    float filteredTime; // Filtered time
    float predictedTime; // Predicted time
    float measurementTime; // Measurement time

public:
    void initializeFilterStateCovariance(float x, float y, float z, float vx, float vy, float vz, float time) {
        Sf << x, 0, 0, 0, 0, 0,
              0, y, 0, 0, 0, 0,
              0, 0, z, 0, 0, 0,
              0, 0, 0, vx, 0, 0,
              0, 0, 0, 0, vy, 0,
              0, 0, 0, 0, 0, vz;

        filteredTime = time;
    }

    void predictStateCovariance(float delt, float plantNoise) {
        Phi << 1, 0, 0, delt, 0, 0,
               0, 1, 0, 0, delt, 0,
               0, 0, 1, 0, 0, delt,
               0, 0, 0, 1, 0, 0,
               0, 0, 0, 0, 1, 0,
               0, 0, 0, 0, 0, 1;

        Sp = Phi * Sf;
        predictedTime = filteredTime + delt;

        float T_3 = (delt * delt * delt) / 3.0;
        float T_2 = (delt * delt) / 2.0;
        Q << T_3, 0, 0, T_2, 0, 0,
             0, T_3, 0, 0, T_2, 0,
             0, 0, T_3, 0, 0, T_2,
             T_2, 0, 0, delt, 0, 0,
             0, T_2, 0, 0, delt, 0,
             0, 0, T_2, 0, 0, delt;
        Q = Q * plantNoise;

        Pp = Phi * Pf * Phi.transpose() + Q;
    }

    void filterStateCovariance() {
        Matrix<float, 6, 6> prevSf = Sf;
        float prevFilteredTime = filteredTime;

        Matrix<float, 3, 1> Inn = Z - H * Sp;
        Matrix<float, 6, 6> S = R + H * Pp * H.transpose();
        K = Pp * H.transpose() * S.inverse();

        Sf = Sp + K * Inn;
        Pf = (Matrix<float, 6, 6>::Identity() - K * H) * Pp;

        filteredTime = measurementTime;
    }
};

int main() {
    // Initialize Kalman Filter
    CVFilter kalmanFilter;

    // Initialize state covariance matrix
    kalmanFilter.initializeFilterStateCovariance(1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.0);

    // Predict state covariance matrix
    kalmanFilter.predictStateCovariance(0.1, 0.01);

    // Assuming measurement vector and noise
    kalmanFilter.Z << 10.0, 20.0, 30.0;
    kalmanFilter.R << 0.1, 0, 0,
                      0, 0.2, 0,
                      0, 0, 0.3;

    // Measurement update
    kalmanFilter.filterStateCovariance();

    // Output filtered state covariance matrix
    std::cout << "Filtered State Covariance Matrix:\n" << kalmanFilter.getSf() << std::endl;

    return 0;
}
