import numpy as np
from numpy.linalg import svd

# def unit_ball()
from scipy.stats import multivariate_normal


class unit_hypercube(object):
    def __init__(self, dim):
        self.dim = dim

    def rvs(self, size):
        return np.random.uniform(-1, 1, (size, self.dim))

    def logpdf(self, x):
        return np.where(np.abs(x) > 1, -np.inf, 0)

    def pdf(self, x):
        return np.where(np.abs(x) > 1, 0, 1)


class unit_hyperball(object):
    def __init__(self, dim, scale=1.0, loc=0.0):
        self.dim = dim
        self.scale = scale
        self.loc = loc

    def rvs(self, size):
        x = np.random.randn(size, self.dim)
        r = np.random.rand(size) ** (1 / self.dim)
        return (
            x / np.linalg.norm(x, axis=1)[:, None] * r[:, None] * self.scale + self.loc
        )

    def logpdf(self, x):
        r = np.linalg.norm(x, axis=1)
        return np.where(r > 1, -np.inf, (self.dim - 1) * np.log(r))

    def pdf(self, x):
        r = np.linalg.norm(x, axis=1)
        return np.where(r > 1, 0, (self.dim - 1) / 2 * r ** (self.dim - 1))


class ellipse(object):
    def __init__(self, points):
        self.points = np.asarray([xi.x for xi in points])
        # self.points = points
        self.dim = self.points.shape[1]
        self.mean = np.mean(self.points, axis=0)
        # svd(self.points)
        # self.cov = svd(self.points-self.mean)[2]
        self.cov = np.cov(self.points, rowvar=False)
        self.svd = svd(self.cov)
        _, singular_values, singular_vectors = np.linalg.svd(self.points - self.mean)
        self.transform = singular_vectors.T @ np.diag(singular_values)

    # def rvs(self, size):
    #     x = np.random.randn(size, self.dim)
    #     return x @ self.cov.T + self.mean

    def rvs(self, size):
        x = np.random.randn(size, self.dim)
        r = np.random.rand(size) ** (1 / self.dim)
        return (x / np.linalg.norm(x, axis=1)[:, None] * r[:, None]) @ np.linalg.inv(
            self.cov.T
        ) + self.mean
        # Scale and rotate the points
        # eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
        # points = points @ (eigenvectors * np.sqrt(eigenvalues))

        # Shift the points to have the desired mean
        # return points + self.mean

    def logpdf(self, x):
        return multivariate_normal(self.mean, self.cov).logpdf(x)
