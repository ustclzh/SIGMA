
import numpy as np
import matplotlib.pyplot as plt


class SDE_OU:
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.sde_dim = 1
        self.theta_dim = 3
        self.dim = 4
        self.x_init = np.array([0, 1, 1, 0.2]) # Initial guesses for X, θ₁, θ₂, sigma_S
        self.P_init = np.eye(4) * 10 
        self.Q = np.diag([dt, dt, dt, dt])  # Process noise
        self.R = np.array([[0.]])  # Measurement noise

    def f(self, x):
        X, theta1, theta2, sigma_S = x
        return np.array([X + theta1 * (theta2 - X) * self.dt, theta1, theta2, sigma_S])

    def h(self, x):
        return x[0]

    def F_jacobian(self, x):
        X, theta1, theta2, sigma_S = x
        return np.array([
            [1 - theta1 * self.dt, (theta2 - X) * self.dt, theta1 * self.dt, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def G(self, x):
        X, theta1, theta2, sigma_S = x
        return np.array([[sigma_S, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0], ])

    def H_jacobian(self, x):
        return np.array([[1, 0, 0, 0]])
    def generate_synthetic_data(self, theta, N, dt = None):
        if dt is None: dt = self.dt
        #true_theta1, true_theta2, true_sigma_S = 1.2, 0.8, 0.5
        x_true = np.zeros((N,1))
        x_true[0] = self.x_init[0]
        for i in range(1, N):
            x_true[i] = x_true[i - 1] + theta[0] * (theta[1] - x_true[i - 1]) * dt + theta[2] * np.sqrt(dt) * np.random.randn()
        measurements = x_true
        return (measurements)

# Bessel_Process class
class Bessel_Process:
    def __init__(self, dt):
        self.dt = dt
        self.sde_dim = 1
        self.theta_dim = 2
        self.dim = 3
        self.x_init = np.array([1, 1, 0.5])  # Initial guesses for X, μ_S, σ_S
        self.P_init = np.eye(3) * 10 
        self.Q = np.diag([dt, dt, dt])  # Process noise
        self.R = np.array([[0.1]])  # Measurement noise

    def f(self, x):
        X, mu_S, sigma_S = x
        return np.array([X + (mu_S / X) * self.dt, mu_S, sigma_S])

    def h(self, x):
        return x[0]

    def F_jacobian(self, x):
        X, mu_S, sigma_S = x
        return np.array([
            [1 + mu_S * self.dt / (X**2), self.dt / X, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    def G(self, x):
        X, mu_S, sigma_S = x
        return np.array([[sigma_S, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],])

    def H_jacobian(self, x):
        return np.array([[1, 0, 0]])

    def generate_synthetic_data(self, theta, N, dt = None):
        if dt is None: dt = self.dt
        X = np.zeros((N,1))
        X[0] = self.x_init[0]
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            X[i] = X[i - 1] + theta[0] / X[i - 1] * dt + theta[1] * dW
        return X

class LotkaVolterraModel:
    def __init__(self, dt):
        self.dt = dt
        self.sde_dim = 2
        self.theta_dim = 6
        self.dim = 8
        self.x_init = np.array([40, 9, 0.1, 0.01, 0.2, 0.01, 0.5, 0.5])  # Initial guesses for x1, x2, a, b, c, d, sigma1, sigma2
        self.P_init = np.eye(8) * 10 
        self.Q = np.diag([dt, dt, dt, dt, dt, dt, dt, dt])  # Process noise
        self.R = np.diag([1,1])  # Measurement noise
    def f(self, x):
        x1, x2, a, b, c, d, sigma1, sigma2 = x
        dx1 = x1 * (a - b * x2) * self.dt
        dx2 = x2 * (c * x1 - d) * self.dt
        return x + np.array([dx1, dx2, 0, 0, 0, 0, 0, 0])
    def h(self, x):
        return x[:2]
    def F_jacobian(self, x):
        x1, x2, a, b, c, d, sigma1, sigma2 = x
        return np.array([
            [1 + self.dt * (a - b * x2), -self.dt * b * x1, self.dt * x1, -self.dt * x1 * x2, 0, 0, 0, 0],
            [self.dt * c * x2, 1 + self.dt * (c * x1 - d), 0, 0, self.dt * x2, -self.dt * x2, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
    def G(self, x):
        return np.array([
            [x[-2], 0, 0, 0, 0, 0, 0, 0],
            [0, x[-1], 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
    def H_jacobian(self, x):
        return np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0]
        ])
    def generate_synthetic_data(self, theta, N, dt = None):
        if dt is None: dt = self.dt
        x = np.zeros((N, 2))
        x[0] = self.x_init[0:2]
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt), 2)
            dx1 = x[i - 1, 0] * (theta[0] - theta[1] * x[i - 1, 1]) * dt + theta[4] * dW[0]
            dx2 = x[i - 1, 1] * (theta[2] * x[i - 1, 0] - theta[3]) * dt + theta[5] * dW[1]
            x[i] = x[i - 1] + np.array([dx1, dx2])
        return x


class LotkaVolterraModel2:
    def __init__(self, dt):
        self.dt = dt
        self.sde_dim = 2
        self.theta_dim = 5
        self.dim = 7
        self.x_init = np.array([40, 9, 0.1, 0.01, 0.2, 0.1, 0.1])  # Initial guesses for x1, x2, a, b, c, d, sigma1, sigma2
        self.P_init = np.eye(7) * 10 
        self.Q = np.diag([dt, dt, dt, dt, dt, dt, dt])  # Process noise
        self.R = np.diag([1,1])  # Measurement noise
    def f(self, x):
        x1, x2, a, b, d, sigma1, sigma2 = x
        dx1 = x1 * (a - b * x2) * self.dt
        dx2 = x2 * (b * x1 - d) * self.dt
        return x + np.array([dx1, dx2, 0, 0, 0, 0, 0])
    def h(self, x):
        return x[:2]
    def F_jacobian(self, x):
        x1, x2, a, b, d, sigma1, sigma2 = x
        return np.array([
            [1 + self.dt * (a - b * x2), -self.dt * b * x1, self.dt * x1, -self.dt * x1 * x2, 0, 0, 0],
            [self.dt * b * x2, 1 + self.dt * (b * x1 - d), 0, self.dt * x2, -self.dt * x2, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
    def G(self, x):
        return np.array([
            [x[-2], 0, 0, 0, 0, 0, 0],
            [0, x[-1], 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
    def H_jacobian(self, x):
        return np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0]
        ])
    def generate_synthetic_data(self, theta, N, dt = None):
        if dt is None: dt = self.dt
        x = np.zeros((N, 2))
        x[0] = self.x_init[0:2]
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt), 2)
            dx1 = x[i - 1, 0] * (theta[0] - theta[1] * x[i - 1, 1]) * dt + theta[3] * dW[0]
            dx2 = x[i - 1, 1] * (theta[1] * x[i - 1, 0] - theta[2]) * dt + theta[4] * dW[1]
            x[i] = x[i - 1] + np.array([dx1, dx2])
        return x


def extended_kalman_filter(model, measurements):
    x_pred = model.x_init
    P_pred = model.P_init
    inferred_params = []
    for y_obs in measurements:
        # Prediction step
        x_pred = model.f(x_pred)
        F = model.F_jacobian(x_pred)
        G = model.G(x_pred)
        P_pred = F @ P_pred @ F.T + G @ model.Q @ G.T
        # Update step
        H = model.H_jacobian(x_pred)
        S = H @ P_pred @ H.T + model.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_update = x_pred + K @  (y_obs - model.h(x_pred))
        P_update = (np.eye(model.dim) - K @ H) @ P_pred
        inferred_params.append(x_update.copy())
        x_pred = x_update
        P_pred = P_update
    return np.array(inferred_params)

