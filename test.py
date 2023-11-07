from fusions import DiffusionModel

# from biff.biff import DiffusionModelBase
from jax import numpy as jnp
import numpy as np

model = DiffusionModel()
model.read_chains("data/gaussian_2d")

x0 = np.random.normal(0, 1, (1000, 2))
model.train()
print("Done")
