# fusions

[![tests](https://github.com/yallup/fusions/actions/workflows/tests.yml/badge.svg)](https://github.com/yallup/fusions/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/fusions.svg)](https://badge.fury.io/py/fusions)

Diffusion meets (nested) sampling

A miniminal implementation of generative diffusion models in JAX (Flax). Tuned for usage in building emulators for scientific models, particularly where MCMC sampling is tractable and used.


```python
from fusions.cfm import CFM
from lsbi.model import LinearMixtureModel
from anesthetic import MCMCSamples
import matplotlib.pyplot as plt
import numpy as np


dims = 5
Model = LinearMixtureModel(
    M=np.stack([np.eye(dims), -np.eye(dims)]),
    C=np.eye(dims)*0.1,
)

data = Model.evidence().rvs(1)

diffusion = CFM(Model.prior())
# diffusion = CFM(dims)

diffusion.train(Model.posterior(data).rvs(1000))

a = MCMCSamples(Model.posterior(data).rvs(500)).plot_2d(np.arange(dims))
MCMCSamples(diffusion.rvs(500)).plot_2d(a)
plt.show()
```