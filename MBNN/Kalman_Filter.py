# Reference : https://github.com/ChadFulton/tsa-notebooks/blob/master/code_state_space.ipynb 
import numpy as np

def kalman_filter(y, Z, H, T, Q, a_0, P_0):
    # Dimensions
    k_endog, nobs = y.shape
    k_states = T.shape[0]

    # Allocate memory for variabless
    filtered_state = np.zeros((k_states, nobs))
    filtered_state_cov = np.zeros((k_states, k_states, nobs))
    predicted_state = np.zeros((k_states, nobs+1))
    predicted_state_cov = np.zeros((k_states, k_states, nobs+1))
    forecast = np.zeros((k_endog, nobs))
    forecast_error = np.zeros((k_endog, nobs))
    forecast_error_cov = np.zeros((k_endog, k_endog, nobs))
    loglikelihood = np.zeros((nobs+1,))

    # Copy initial values to predicted
    predicted_state[:, 0] = a_0
    predicted_state_cov[:, :, 0] = P_0

    # Kalman filter iterations
    for t in range(nobs):

        # Forecast for time t
        forecast[:, t] = np.dot(Z, predicted_state[:, t])

        # Forecast error for time t
        forecast_error[:, t] = y[:, t] - forecast[:, t]

        # Forecast error covariance matrix and inverse for time t
        tmp1 = np.dot(predicted_state_cov[:, :, t], Z.T)
        forecast_error_cov[:, :, t] = (
            np.dot(Z, tmp1) + H
        )
        forecast_error_cov_inv = np.linalg.inv(forecast_error_cov[:, :, t])
        determinant = np.linalg.det(forecast_error_cov[:, :, t])

        # Filtered state for time t
        tmp2 = np.dot(forecast_error_cov_inv, forecast_error[:,t])
        filtered_state[:, t] = (
            predicted_state[:, t] +
            np.dot(tmp1, tmp2)
        )

        # Filtered state covariance for time t
        tmp3 = np.dot(forecast_error_cov_inv, Z)
        filtered_state_cov[:, :, t] = (
            predicted_state_cov[:, :, t] -
            np.dot(
                np.dot(tmp1, tmp3),
                predicted_state_cov[:, :, t]
            )
        )

        # Loglikelihood
        loglikelihood[t] = -0.5 * (
            np.log((2*np.pi)**k_endog * determinant) +
            np.dot(forecast_error[:, t], tmp2)
        )

        # Predicted state for time t+1
        predicted_state[:, t+1] = np.dot(T, filtered_state[:, t])

        # Predicted state covariance matrix for time t+1
        tmp4 = np.dot(T, filtered_state_cov[:, :, t])
        predicted_state_cov[:, :, t+1] = np.dot(tmp4, T.T) + Q
        
        predicted_state_cov[:, :, t+1] = (
            predicted_state_cov[:, :, t+1] + predicted_state_cov[:, :, t+1].T
        ) / 2

    return (
        filtered_state, filtered_state_cov,
        predicted_state, predicted_state_cov,
        forecast, forecast_error, forecast_error_cov,
        loglikelihood
    )