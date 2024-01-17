import warnings

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from fusions.cfm import CFM
from fusions.diffusion import Diffusion

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.mark.parametrize("dim", [2, 5])
@pytest.mark.parametrize("use_prior", [False, True])
class TestDiffusion(object):
    CLS = Diffusion

    @pytest.fixture
    def prior(self, dim):
        return multivariate_normal(np.zeros(dim))

    @pytest.fixture
    def model(self, prior, use_prior, dim):
        if use_prior:
            return self.CLS(prior=prior)
        else:
            return self.CLS(n=dim)

    @pytest.fixture
    def batch(self, rng, dim):
        return jax.random.normal(rng, (10, dim))

    def test_sample_prior(self, model, dim):
        xs = model.sample_prior(1000)
        assert xs.shape == (1000, dim)
        assert xs.mean() == pytest.approx(0, abs=0.1)

    @pytest.fixture
    def batch(self, rng, dim):
        return jax.random.normal(rng, (10, dim))

    @pytest.fixture
    def train_opts(self):
        return {"batch_size": 10, "lr": 1e-3, "n_epochs": 1}

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(2022)

    def test_init_state(self, model, batch):
        model.ndims = batch.shape[-1]
        model._init_state()
        assert model.state is not None

    def test_loss(self, model, rng, batch):
        model.ndims = batch.shape[-1]
        model._init_state()
        batch_prior = model.prior.rvs(batch.shape[0])
        loss = model.loss(
            model.state.params,
            batch,
            batch_prior,
            model.state.batch_stats,
            rng,
        )
        assert loss[0].dtype == np.float32

    def test_train_step(self, model, batch, train_opts):
        model.ndims = batch.shape[-1]
        model._init_state()
        model._train(batch, **train_opts)
        assert model.state.step == 1

    def test_train(self, model, batch, train_opts):
        model.ndims = batch.shape[-1]
        model._init_state()
        model.train(batch, **train_opts)
        assert model.state.step == 1
        x1 = model.predict(batch)
        assert (batch != x1).all()

    def test_reverse_process(self, model, batch):
        model.ndims = batch.shape[-1]
        model._init_state()

        model._predict = lambda x, t: model.state.apply_fn(
            {
                "params": model.state.params,
                "batch_stats": model.state.batch_stats,
            },
            x,
            t,
            train=False,
        )
        model.rng = jax.random.PRNGKey(0)
        x1, x1t = model.reverse_process(batch, model._predict, model.rng)
        x2, x2t = model.predict(batch, history=True)

        assert_allclose(x1, x2)
        assert_allclose(x1t, x2t)

        x3 = model.sample_posterior(batch.shape[0])
        x4 = model.sample_posterior(batch.shape[0])

        assert not (x3 == x4).all()


class TestCFM(TestDiffusion):
    CLS = CFM
