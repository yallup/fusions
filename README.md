# fusions

[![tests](https://github.com/yallup/fusions/actions/workflows/tests.yml/badge.svg)](https://github.com/yallup/fusions/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/fusions.svg)](https://badge.fury.io/py/fusions)

Diffusion meets (nested) sampling

A miniminal implementation of diffusion models in JAX (Flax). Tuned for usage in building emulators for scientific models, particularly where MCMC sampling is tractable and used.


## Quickstart

Install `fusions` and `lsbi` from pypi
```
pip install lsbi fusions
```

create a 5D sampling problem then train a flow matched model to approximate the posterior

```python
from fusions.cfm import CFM
from lsbi.model import MixtureModel
from anesthetic import MCMCSamples
import matplotlib.pyplot as plt
import numpy as np


dims = 5
Model = MixtureModel(
    M=np.stack([np.eye(dims), -np.eye(dims)]),
    C=np.eye(dims)*0.1,
)

data = Model.evidence().rvs()

diffusion = CFM(Model.prior())
# diffusion = CFM(dims)

diffusion.train(Model.posterior(data).rvs(1000))

a = MCMCSamples(Model.posterior(data).rvs(500)).plot_2d(np.arange(dims))
MCMCSamples(diffusion.rvs(500)).plot_2d(a)
plt.show()
```
