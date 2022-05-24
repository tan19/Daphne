import numpy as np
from scipy.linalg import inv

__all__ = ['GaussianProcessRegression']

class GaussianProcessRegression():
    def __init__(self, xs, ys, x, kernel_f=None):
        self.xs = xs
        self.ys = ys
        self.x = x
        self.kernel_f = kernel_f if kernel_f else (lambda x_i, x_j : np.exp(- 0.1 * (x_i - x_j) ** 2))

        self.len_xs = len(self.xs)
        self.len_x = len(self.x)

    @property
    def mean(self):
        return np.zeros(len(self.xs) + self.len_x)
        
    @property
    def covariance_matrix(self):
        sigma_11 = np.zeros((len(self.xs), len(self.xs)))
        for i in range(len(self.xs)):
            for j in range(i, len(self.xs)):
                sigma_11[i, j] = self.kernel_f(self.xs[i], self.xs[j])
                sigma_11[j, i] = sigma_11[i, j]
        sigma_11 += np.diag([0.01] * len(self.xs))

        sigma_12 = np.zeros((len(self.xs), self.len_x))
        for i in range(len(self.xs)):
            for j in range(self.len_x):
                sigma_12[i, j] = self.kernel_f(self.xs[i], self.x[j])                
        sigma_21 = sigma_12.T

        sigma_22 = np.zeros((self.len_x, self.len_x))
        for i in range(self.len_x):
            for j in range(i, self.len_x):
                sigma_22[i, j] = self.kernel_f(self.x[i], self.x[j])
                sigma_22[j, i] = sigma_22[i, j]

        return np.block([[sigma_11, sigma_12], [sigma_21, sigma_22]])

    @property
    def precision_matrix(self):
        return inv(self.covariance_matrix)

    @property
    def conditional_mean(self):
        mu_1 = self.mean[:self.len_xs]
        mu_2 = self.mean[-self.len_x:]
        sigma_21 = self.covariance_matrix[self.len_xs:, :self.len_xs]
        sigma_11 = self.covariance_matrix[:self.len_xs, :self.len_xs]
                
        return mu_2 + sigma_21 @ inv(sigma_11) @ (self.ys - mu_1)        

    @property
    def conditional_var(self):        
        sigma_21 = self.covariance_matrix[-self.len_x:, :-self.len_x]
        sigma_11 = self.covariance_matrix[:-self.len_x, :-self.len_x]
        sigma_22 = self.covariance_matrix[-self.len_x:, -self.len_x:]

        return sigma_22 - sigma_21 @ inv(sigma_11) @ sigma_21.T