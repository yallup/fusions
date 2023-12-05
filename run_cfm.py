from fusions.cfm import CFMBase
import anesthetic
import numpy as np
from scipy.stats import multivariate_normal
from numpy.random import default_rng
dims = 5
rng = default_rng(0)
np.random.seed(5)
# prior = multivariate_normal(mean=rng.normal(size=dims))
prior = multivariate_normal(mean=np.zeros(dims), cov = np.eye(dims) * 1)

data_1 = multivariate_normal(
    mean=np.random.rand(dims)*4-3, cov=np.eye(dims) * 0.01
).rvs(1000)
data_2 = multivariate_normal(
    mean=np.random.rand(dims)*4-1, cov=np.eye(dims) * 0.01
).rvs(1000)
data = np.concatenate([data_1, data_2])

# data = data_2
# plt.scatter(data[:,0],data[:,1])
# ns.MCMCSamples(data).plot_2d()
# chains = anesthetic.read_chains("gaussian")
model = CFMBase(prior)
# model.ndims = 5

model.train(data, n_epochs=2000,batch_size=256,lr=1e-3)
xt=model.sample_posterior(2000)
x0=model.sample_prior(1000)
f,a=anesthetic.make_2d_axes(np.arange(dims))
a=anesthetic.MCMCSamples(xt).plot_2d(a)
a=anesthetic.MCMCSamples(x0).plot_2d(a,alpha=0.2)
a= anesthetic.MCMCSamples(data).plot_2d(a)
f.savefig("test.pdf")
print("Done")
