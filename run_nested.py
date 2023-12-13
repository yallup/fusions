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
    mu=np.zeros(dims),
    sigma=np.eye(dims),
    m=np.zeros(dims),
    C=np.eye(dims) * 0.1,
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
size = 1000
theta = Model.prior().rvs(size)
P = Model.posterior(true_theta).rvs(size)
a = ns.MCMCSamples(theta).plot_2d(np.arange(dims))

f, a = ns.make_2d_axes(np.arange(dims))
ns.MCMCSamples(theta).plot_2d(a, alpha=0.3, label="Prior")
ns.MCMCSamples(P).plot_2d(a, alpha=0.3, label="Analytic")
samples.plot_2d(a, label="NestedDiffusion")
plt.legend()
f.savefig("plots/ns.pdf")
plt.close()

from fusions.cfm import CFM
from fusions.diffusion import Diffusion

flow = Diffusion(Model.prior())
flow.train(Model.posterior(true_theta).rvs(10000))
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
