import os

import anesthetic as ns
import matplotlib.pyplot as plt
import numpy as np

# import jax.random as rng
from lsbi.model import LinearModel, MixtureModel
from margarine.clustered import clusterMAF
from margarine.maf import MAF
from scipy.stats import multivariate_normal, uniform

from fusions.cfm import CFM, VPCFM
from fusions.diffusion import Diffusion
from fusions.integrate import NestedDiffusion, SequentialDiffusion

os.makedirs("plots", exist_ok=True)

dims = 5
data_dims = dims
np.random.seed(12)

mixtures = 10
A = np.random.randn(mixtures, data_dims, dims)
Model = MixtureModel(
    # M=np.stack([np.eye(dims), -np.eye(dims)]),
    M=A,
    mu=np.zeros(dims),
    Sigma=np.eye(dims),
    m=np.zeros(data_dims),
    C=np.eye(data_dims) * 0.1,
)

data = Model.evidence().rvs()
logz = Model.evidence().logpdf(data)


class likelihood(object):
    def logpdf(self, x):
        return Model.likelihood(x).logpdf(data)


# diffuser = SequentialDiffusion(
#     prior=Model.prior(), likelihood=likelihood()# , schedule =np.geomspace
# )
diffuser = NestedDiffusion(prior=Model.prior(), likelihood=likelihood(), model=CFM)


diffuser.run(steps=12, n=500, target_eff=0.1)
samples = diffuser.samples()

print(f"analytic: {logz:.2f}")
zs = samples.logZ(50)
print(f"numerical estimation: {zs.mean():.2f} +- {zs.std():.2f}")
# print(samples.logZ())
# print(samples.logZ(30).std())

total_samples = len(samples.compress())
size = total_samples * 2
theta = Model.prior().rvs(size)
P = Model.posterior(data).rvs(size)
a = ns.MCMCSamples(theta).plot_2d(np.arange(dims))

f, a = ns.make_2d_axes(np.arange(dims))
ns.MCMCSamples(theta).plot_2d(a, alpha=0.3, label="Prior")
ns.MCMCSamples(P).plot_2d(a, alpha=0.3, label="Analytic")
samples.plot_2d(a, label="NestedDiffusion")
plt.legend()
f.savefig("plots/ns.pdf")
