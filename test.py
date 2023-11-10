from functools import partial

import anesthetic as ns
import numpy as np
from anesthetic.examples.perfect_ns import gaussian

# from biff.biff import DiffusionModelBase
from jax import grad, jit
from jax import numpy as jnp
from jax import random, vmap
from matplotlib import pyplot as plt

from fusions import DiffusionModel

# from lsbi import LinearMixtureModel
model = DiffusionModel()
chains = gaussian(50, 2)
data_1 = np.random.normal(1, 0.5, (100, 2))
data_2 = np.random.normal(-1, 0.5, (100, 2))
data = np.concatenate([data_1, data_2])
# plt.scatter(data[...,0],data[...,1])
model.chains = chains.set_beta(1.0)
model.ndims = 2


def sample_sphere(J):
    """
    2 dimensional sample

    N_samples: Number of samples
    Returns a (N_samples, 2) array of samples
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1 / J), J)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    mf = jnp.stack([xs, ys], axis=1)
    return mf


# _data = sample_sphere(8)

# nabla_log_hat_pt = jit(vmap(grad(model.log_hat_pt), in_axes=(0, 0), out_axes=(0)))

# def plot_score(score, t, area_min=-1, area_max=1):
#     #this helper function is here so that we can jit it.
#     #We can not jit the whole function since plt.quiver cannot
#     #be jitted
#     # @partial(jit, static_argnums=[0,])
#     def helper(score, t, area_min, area_max):
#         x = jnp.linspace(area_min, area_max, 16)
#         x, y = jnp.meshgrid(x, x)
#         grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
#         t = jnp.ones((grid.shape[0], 1)) * t
#         scores = score(grid, t)
#         return grid, scores
#     grid, scores = helper(score, t, area_min, area_max)
#     plt.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 1])

# plot_score(nabla_log_hat_pt, 0.001, -2, 2)

# x0 = np.random.normal(0, 1, (1000, 2))
model.train(data, batch_size=128, n_epochs=30000)
x0 = np.random.normal(0, 1, (1000, model.ndims))
# x0=np.random.uniform(-1,1,(1000,model.ndims))
x1, x1_t = model.predict(x0)

f, a = ns.make_2d_axes(np.arange(model.ndims), upper=False, diagonal=False)

a = ns.MCMCSamples(x1).plot_2d(a, kinds={"lower": "scatter_2d"})
# ns.MCMCSamples(x0).plot_2d()
f.savefig("test.pdf")

f, a = plt.subplots()
a.plot(x1_t[..., 0])
f.savefig("params.pdf")
print("Done")
