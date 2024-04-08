import os

import matplotlib.pyplot as plt
import numpy as np

# import jax.random as rng
from lsbi.model import LinearModel, MixtureModel
from scipy.stats import multivariate_normal, norm, uniform

import anesthetic as ns
from fusions.cfm import CFM
from fusions.diffusion import Diffusion
from fusions.integrate import NestedDiffusion, SequentialDiffusion

os.makedirs("plots", exist_ok=True)

dims = 5
data_dims = dims * 2
# v hard
np.random.seed(123456)
# np.random.seed(1)

mixtures = 10
A = np.random.randn(mixtures, data_dims, dims)
TargetModel = MixtureModel(
    # M=np.stack([np.eye(dims), -np.eye(dims)]),
    M=A,
    mu=np.zeros(dims),
    Sigma=np.eye(dims),
    m=np.zeros(data_dims),
    C=np.ones(data_dims) * 0.01**2,
)

TargetModel = LinearModel(M=np.random.randn(data_dims, dims), C=0.1)

np.random.seed(123456)
data_dims = dims
A = np.random.rand(mixtures, data_dims, dims)
# A /= np.linalg.norm(A, axis=2)[:, :, None]
# A *=.008
TargetModel = MixtureModel(
    # M=np.stack([np.eye(dims), -np.eye(dims)]),
    M=A,
    mu=np.zeros(dims),
    Sigma=np.eye(dims),
    m=np.zeros(data_dims),
    C=np.ones(data_dims) * 0.05**2,
)

# os.mkdir("nessai", exist_ok=True)


data = TargetModel.evidence().rvs()
logz = TargetModel.evidence().logpdf(data)
print(logz)


class likelihood(object):
    def logpdf(self, x):
        return TargetModel.likelihood(x).logpdf(data)

    def __call__(self, x):
        return TargetModel.likelihood(x).logpdf(data)


# diffuser = SequentialDiffusion(
#     prior=Model.prior(), likelihood=likelihood()# , schedule =np.geomspace
# )
diffuser = NestedDiffusion(
    prior=TargetModel.prior(), likelihood=likelihood(), model=CFM
)
diffuser.settings.target_eff = 1.0
diffuser.settings.epoch_factor = 10.0
diffuser.settings.n = 1000
diffuser.settings.noise = 1e-3
diffuser.settings.prior_boost = 1
diffuser.settings.eps = 1e-3
diffuser.settings.batch_size = 128
diffuser.settings.restart = False
diffuser.settings.lr = 1e-3
# diffuser.run(steps=10, n=500)
diffuser.run()
samples = diffuser.samples()
diffuser.write("diffusion")
print(f"analytic: {logz:.2f}")
zs = samples.logZ(50)
print(f"numerical estimation: {zs.mean():.2f} +- {zs.std():.2f}")
print(samples.logZ())
print(samples.logZ(30).std())

total_samples = len(samples.compress(200))
print(f"total samples: {total_samples}")
size = total_samples * 2
theta = TargetModel.prior().rvs(size)
P = TargetModel.posterior(data).rvs(size * 3)
a = ns.MCMCSamples(theta).plot_2d(np.arange(dims))

f, a = ns.make_2d_axes(np.arange(dims))
# ns.MCMCSamples(theta).plot_2d(a, alpha=0.3, label="Prior")
ns.MCMCSamples(P).plot_2d(a, alpha=0.3, label="Analytic")
samples.plot_2d(a, label="NestedDiffusion")
plt.legend()
f.savefig("plots/ns.pdf")
