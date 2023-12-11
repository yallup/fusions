import anesthetic
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
from scipy.stats import multivariate_normal, uniform

from fusions.cfm import CFMBase

dims = 2
rng = default_rng(0)
np.random.seed(4)
prior = multivariate_normal(mean=rng.normal(size=dims))
# prior = multivariate_normal(mean=np.zeros(dims), cov=np.eye(dims) * [1,2,3])
# prior =


class uniform_prior(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def rvs(self, n):
        return uniform.rvs(self.low, self.high, size=(n, dims))


# prior = uniform_circle(0.5,0.1)
data_1 = multivariate_normal(mean=np.ones(dims) + 1.5, cov=np.eye(dims) * 0.025).rvs(
    2000
)
data_2 = multivariate_normal(mean=np.ones(dims), cov=np.eye(dims) * 0.05).rvs(2000)
data = np.concatenate([data_1, data_2])

# data = data_2
# data= uniform_circle(0.5,0.51).rvs(2000)
# data = multivariate_normal(mean=np.zeros(dims) + 0.2, cov = np.eye(dims) * 0.5).rvs(1000)


# plt.scatter(data[:,0],data[:,1])
# ns.MCMCSamples(data).plot_2d()
# chains = anesthetic.read_chains("gaussian")
model = CFMBase(prior)
# model.ndims = 5

model.train(data, n_epochs=5000, batch_size=2024, lr=1e-3)
import timeit

start = timeit.default_timer()
x, xt = model.sample_posterior(1000, history=True)
end = timeit.default_timer()
print(end - start)

f, a = plt.subplots()
# a.plot(*xt[:,:,0], *xt[:,:,1], alpha=0.8, color="k")
# Plot each trajectory separately
for i in range(xt.shape[0]):
    a.plot(xt[i, :, 0], xt[i, :, 1], alpha=0.2, color="k")

a.scatter(xt[:, 0, 0], xt[:, 0, 1], alpha=0.5, color="C0")
a.scatter(xt[:, -1, 0], xt[:, -1, 1], alpha=0.5, color="C1")
f.savefig("trajectory.pdf")

x0 = model.sample_prior(1000)
f, a = anesthetic.make_2d_axes(np.arange(dims))
a = anesthetic.MCMCSamples(x).plot_2d(a)
a = anesthetic.MCMCSamples(x0).plot_2d(a, alpha=0.2)
a = anesthetic.MCMCSamples(data).plot_2d(a, alpha=0.5)
f.savefig("test.pdf")
print("Done")
