import numpy as np
from jax import numpy as jnp
from jax import random
from ott.geometry import pointcloud
from ott.solvers import linear

nprng = np.random.default_rng()


class OTMap(object):
    def __init__(self, x0, x1):
        x0, x1
        # geom = pointcloud.PointCloud(x0, x1)
        # self.M = linear.solve(geom).matrix.flatten()
        # self.N= np.asarray(linear.solve(geom).matrix)
        # self.N = self.N / self.N.sum()
        # # self.M = self.M / self.M.sum()
        # self.M = np.asarray(self.M)
        # self.M = self.M / self.M.sum()

    def sample(self, rng, batch_size=128):
        gumbel_noise = random.gumbel(rng, shape=(batch_size,) + self.M.shape)
        idx = random.categorical(rng, self.M.flatten(), shape=(batch_size,))
        idx = jnp.divmod(idx, batch_size)
        return idx[0], idx[1]

    def sample_np(self, rng, batch_size=128):
        idx = np.random.choice(
            self.M.shape[0], size=batch_size, p=self.M, replace=False
        )
        idx = np.divmod(idx, batch_size)
        return idx[0], idx[1]

    def sample_flat(self, rng, data_size, batch_size=128):
        idx = nprng.choice(data_size, size=(batch_size, 2), replace=True)
        # idx = np.random.choice(self.M.shape[0], size=batch_size, replace=False)
        # idx = np.divmod(idx, batch_size)
        return idx[..., 0], idx[..., 1]

    def sample_matrix(self, rng, batch_size=128):
        # idx = np.random.choice(self.N.shape[0], size=batch_size, replace=False)
        idx_1 = np.random.choice(self.N.shape[0], size=batch_size, replace=False)
        idx_0 = np.argmax(self.N[..., idx_1], axis=0)
        # idx = np.divmod(idx, batch_size)
        return idx_0, idx_1

    # def sample_matrix(self,rng,batch_size=128):
    #     # idx = np.random.choice(self.N.shape[0], size=batch_size, replace=False)
    #     idx_0=np.random.choice(self.N.shape[0], size=batch_size, replace=False)
    #     idx_1= np.argmax(self.N[idx_0,...],axis=-1)
    #     # idx = np.divmod(idx, batch_size)
    #     return idx_0, idx_1
