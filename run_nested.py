import os

import anesthetic as ns
import matplotlib.pyplot as plt
import numpy as np

# import jax.random as rng
from lsbi.model import LinearModel, MixtureModel
from margarine.clustered import clusterMAF
from margarine.maf import MAF
from scipy.stats import multivariate_normal, norm, uniform

from fusions.cfm import CFM
from fusions.diffusion import Diffusion
from fusions.integrate import NestedDiffusion, SequentialDiffusion

os.makedirs("plots", exist_ok=True)

dims = 5
data_dims = dims * 2
# v hard
np.random.seed(123)
# np.random.seed(1)

mixtures = 5
A = np.random.randn(mixtures, data_dims, dims)
TargetModel = MixtureModel(
    # M=np.stack([np.eye(dims), -np.eye(dims)]),
    M=A,
    mu=np.zeros(dims),
    Sigma=np.eye(dims),
    m=np.zeros(data_dims),
    C=np.ones(data_dims) * 0.1**2,
)

# os.mkdir("nessai", exist_ok=True)


data = TargetModel.evidence().rvs()
logz = TargetModel.evidence().logpdf(data)


class likelihood(object):
    def logpdf(self, x):
        return TargetModel.likelihood(x).logpdf(data)

    def __call__(self, x):
        return TargetModel.likelihood(x).logpdf(data)


from nessai.livepoint import dict_to_live_points
from nessai.model import Model


class GaussianModel(Model):
    """A simple two-dimensional Gaussian likelihood."""

    def __init__(self):
        # Names of parameters to sample
        self.names = ["0", "1", "2", "3", "4"]
        # Prior bounds for each parameter
        self.bounds = {
            "0": [-10, 10],
            "1": [-10, 10],
            "2": [-10, 10],
            "3": [-10, 10],
            "4": [-10, 10],
        }
        # self.bounds = {"0": [0,1], "1": [0,1], "2": [0,1], "3": [0,1], "4": [0,1]}

    def log_prior(self, x):
        """
        Returns the log-prior.

        Checks if the points are in bounds.
        """
        log_p = np.log(self.in_bounds(x), dtype="float")
        p = norm(scale=1)
        # log_p += p.logpdf(p.ppf(x["0"]))
        # log_p += p.logpdf(p.ppf(x["1"]))
        # log_p += p.logpdf(p.ppf(x["2"]))
        # log_p += p.logpdf(p.ppf(x["3"]))
        # log_p += p.logpdf(p.ppf(x["4"]))
        log_p += p.logpdf(x["0"])
        log_p += p.logpdf(x["1"])
        log_p += p.logpdf(x["2"])
        log_p += p.logpdf(x["3"])
        log_p += p.logpdf(x["4"])

        return log_p

        # log_p += norm(scale=1).logpdf(x["1"])
        # log_p += norm(scale=1).logpdf(x["2"])
        # log_p += norm(scale=1).logpdf(x["3"])
        # log_p += norm(scale=1).logpdf(x["4"])
        # return log_p

    def new_point(self, N=1):
        """Draw n points.

        This is used for the initial sampling. Points do not need to be drawn
        from the exact prior but algorithm will be more efficient if they are.
        """
        # There are various ways to create live points in nessai, such as
        # from dictionaries and numpy arrays. See nessai.livepoint for options
        d = {
            "0": norm(scale=1).rvs(size=N),
            "1": norm(scale=1).rvs(size=N),
            "2": norm(scale=1).rvs(size=N),
            "3": norm(scale=1).rvs(size=N),
            "4": norm(scale=1).rvs(size=N),
        }
        return dict_to_live_points(d)

    def log_likelihood(self, x):
        """
        Returns log likelihood of given live point assuming a Gaussian
        likelihood.
        """
        # print(x[0])
        log_l = np.zeros(x.size)
        t = []
        for i in self.names:
            t.append(x[i])
        # print(TargetModel.likelihood(np.asarray(t)).logpdf(data))
        log_l = TargetModel.likelihood(np.asarray(t).T.squeeze()).logpdf(data)
        return log_l


import ultranest
from nessai.flowsampler import FlowSampler

# Initialise sampler with the model
# sampler = FlowSampler(GaussianModel(), output='./nessai', nlive=1000)
# # Run the sampler
# sampler.run()
# import sys
# sys.exit()


# sampler = ultranest.ReactiveNestedSampler(["0","1","2","3","4"], likelihood(), norm.ppf,
#     log_dir="myanalysis", resume="overwrite")
# result = sampler.run()
# sampler.print_results()

# import sys
# sys.exit()

# diffuser = SequentialDiffusion(
#     prior=Model.prior(), likelihood=likelihood()# , schedule =np.geomspace
# )
diffuser = NestedDiffusion(
    prior=TargetModel.prior(), likelihood=likelihood(), model=CFM
)


diffuser.run(steps=10, n=500)
samples = diffuser.samples()

print(f"analytic: {logz:.2f}")
zs = samples.logZ(50)
print(f"numerical estimation: {zs.mean():.2f} +- {zs.std():.2f}")
# print(samples.logZ())
# print(samples.logZ(30).std())

total_samples = len(samples.compress(200))
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
