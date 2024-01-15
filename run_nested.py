import os

import anesthetic as ns
import matplotlib.pyplot as plt
import numpy as np

# import jax.random as rng
from lsbi.model import LinearMixtureModel, LinearModel
from margarine.clustered import clusterMAF
from margarine.maf import MAF
from scipy.stats import multivariate_normal, uniform

from fusions.cfm import CFM
from fusions.diffusion import Diffusion
from fusions.integrate import (
    NestedDiffusion,
    NestedSequentialDiffusion,
    SequentialDiffusion,
    SimpleNestedDiffusion,
)

os.makedirs("plots", exist_ok=True)

dims = 5
data_dims = dims
# true_theta = np.ones(dims)
# np.random.seed(1)
# Model = LinearModel(
#     mu=np.zeros(dims),
#     sigma=np.eye(dims),
#     m=np.zeros(data_dims),
#     C=np.eye(data_dims)*0.01,
# )
np.random.seed(12)

mixtures = 10
A = np.random.randn(mixtures, data_dims, dims)
Model = LinearMixtureModel(
    # M=np.stack([np.eye(dims), -np.eye(dims)]),
    M=A,
    mu=np.zeros(dims),
    sigma=np.eye(dims),
    m=np.zeros(data_dims),
    C=np.eye(data_dims) * 0.1,
)

# 1 prior samples
# theta = Model.prior().rvs(200)
true_theta = Model.prior().rvs()
# evaluate the likelihood
# Model.likelihood(true_theta).logpdf(theta)
# Analytic posterior samples
# data= np.ones(data_dims)
data = Model.evidence().rvs()
# P = Model.posterior(true_theta).rvs(200)
# Evidence value
logz = Model.evidence().logpdf(data)
# print(logz)

# prior = multivariate_normal(mean=np.zeros(dims), cov=np.eye(dims))
# diffuser = SequentialDiffusion(
#     prior=Model.prior(), likelihood=Model.likelihood(true_theta)
# )


class likelihood(object):
    def logpdf(self, x):
        return np.asarray([Model.likelihood(y).logpdf(data) for y in x])


# diffuser = SequentialDiffusion(
#     prior=Model.prior(), likelihood=likelihood()# , schedule =np.geomspace
# )
diffuser = SimpleNestedDiffusion(prior=Model.prior(), likelihood=likelihood())

# diffuser = NestedSequentialDiffusion(
#     prior=Model.prior(), likelihood=likelihood())
# from pypolychord import run_polychord
# from pypolychord.settings import PolyChordSettings
# from pypolychord.priors import GaussianPrior
# def poly_like(theta):
#     return float(Model.likelihood(theta).logpdf(data)), []

# settings = PolyChordSettings(dims, 0)
# run_polychord(poly_like,dims,0,prior=GaussianPrior(0,1), settings=settings)

diffuser.run(steps=10, n=500, target_eff=0.1)
# samples = ns.read_chains("chains/test")
samples = diffuser.samples()
# samples.gui()
# logz_diff = diffuser.importance_integrate(diffuser.dist,10000)

print(f"analytic: {logz:.2f}")
zs = samples.logZ(50)
print(f"numerical estimation: {zs.mean():.2f} +- {zs.std():.2f}")
print(samples.logZ())
print(samples.logZ(30).std())

total_samples = len(samples.compress())
size = total_samples * 2
theta = Model.prior().rvs(size)
P = Model.posterior(data).rvs(size)
a = ns.MCMCSamples(theta).plot_2d(np.arange(dims))

f, a = ns.make_2d_axes(np.arange(dims))
ns.MCMCSamples(theta).plot_2d(a, alpha=0.3, label="Prior")
ns.MCMCSamples(P).plot_2d(a, alpha=0.3, label="Analytic")
samples.plot_2d(a, label="NestedDiffusion")
# samples.set_beta(0.01).plot_2d(a, label="NestedDiffusion 2", alpha=0.5)
plt.legend()
f.savefig("plots/ns.pdf")
plt.close()


flow = Diffusion(Model.prior())
flow.train(Model.posterior(true_theta).rvs(10000), n_epochs=10000)
P_flow = flow.sample_posterior(1000)

from margarine.clustered import clusterMAF
from margarine.maf import MAF

mflow = MAF(Model.posterior(true_theta).rvs(10000))
mflow.train()
P_mflow = mflow.sample(500)
cmflow = clusterMAF(Model.posterior(true_theta).rvs(10000))
cmflow.train()
P_cmflow = cmflow.sample(500)


f, a = ns.make_2d_axes(np.arange(dims))
# a = samples.compress(size).plot_2d(a)
ns.MCMCSamples(theta).plot_2d(a, alpha=0.3, label="Prior")
ns.MCMCSamples(P).plot_2d(a, alpha=0.3, label="Analytic")
ns.MCMCSamples(np.asarray(P_flow)).plot_2d(a, label="Diffusion")
ns.MCMCSamples(np.asarray(P_mflow)).plot_2d(a, alpha=0.8, label="MAF")
ns.MCMCSamples(np.asarray(P_cmflow)).plot_2d(a, alpha=0.8, label="ClusterMAF")
plt.legend()
f.savefig("plots/emulator.pdf")
print("Done")
