import numpy as np

# from biff.biff import DiffusionModelBase
from jax import numpy as jnp

from fusions import DiffusionModel

model = DiffusionModel()


model.read_chains("data/gaussian_2d")


x0 = np.random.normal(0, 1, (1000, 2))
model.train()
print("Done")
