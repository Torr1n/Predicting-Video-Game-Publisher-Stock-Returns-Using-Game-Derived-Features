import numpy as np


class GaussianRBFKernel:
    """
    Custom implementation of Gaussian RBF kernel for interpretability and control.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X1, X2):
        n1, d = X1.shape
        n2, d = X2.shape
        K = np.zeros((n1, n2))

        # Calculate gram matrix
        for i in range(n1):
            for j in range(n2):
                diff = X1[i, :] - X2[j, :]
                norm = diff.T @ diff
                K[i, j] = np.exp((-1 * norm) / (2 * self.sigma**2))
        return K
