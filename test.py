from functools import partial

import anesthetic as ns
import numpy as np
from anesthetic.examples.perfect_ns import gaussian
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

model.train(data, batch_size=128, n_epochs=1000)
x0 = model.prior.rvs((1000, model.ndims))
# x0 = np.random.normal(0, 1, (1000, model.ndims))
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
