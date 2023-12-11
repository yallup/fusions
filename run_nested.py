import anesthetic as ns
import matplotlib.pyplot as plt
import numpy as np

# import jax.random as rng
from lsbi.model import LinearMixtureModel, LinearModel
from scipy.stats import multivariate_normal, uniform

from fusions.nested import NestedDiffusion

dims = 5
true_theta = np.ones(dims) + 1
Model = LinearModel(
    mu=np.zeros(dims), sigma=np.eye(dims), m=np.zeros(dims), C=np.eye(dims) * 0.1
)
Model = LinearMixtureModel(
    M=np.stack([np.eye(dims), -np.eye(dims)]),
    mu=np.zeros(dims),
    sigma=np.eye(dims),
    m=np.zeros(dims),
    C=np.eye(dims) * 0.1,
)
# 1 prior samples
theta = Model.prior().rvs(200)
# evaluate the likelihood
Model.likelihood(true_theta).logpdf(theta)
# Analytic posterior samples
P = Model.posterior(true_theta).rvs(200)
# Evidence value
logz = Model.evidence().logpdf(true_theta)
print(logz)

diffuser = NestedDiffusion(prior=Model.prior(), likelihood=Model.likelihood(true_theta))
diffuser.run(steps=20, n=500)
samples = diffuser.samples()


print(logz)
print(samples.logZ())
total_samples = len(samples.compress())
size = 500
theta = Model.prior().rvs(size)
P = Model.posterior(true_theta).rvs(size)
a = ns.MCMCSamples(theta).plot_2d(np.arange(dims))

a = samples.compress(size).plot_2d(a)
ns.MCMCSamples(P).plot_2d(a, alpha=0.3)
plt.savefig("fusion_sampler.pdf")
print("Done")
