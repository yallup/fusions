import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from fusions.model import DiffusionModel, DiffusionModelBase

test_path = os.path.dirname(os.path.realpath(__file__))


class TestDiffusionBase(object):
    CLS = DiffusionModelBase

    @pytest.fixture
    def model(self):
        return self.CLS()

    @pytest.fixture
    def t(self):
        return jnp.linspace(0, 1, 100)

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(2022)

    @pytest.fixture
    def batch(self, rng):
        return jax.random.normal(rng, (10, 2))

    def test_read_chains(self, model):
        model.read_chains(os.path.join(test_path, "data/data"))
        assert model.chains is not None

    def test_beta_t(self, model, t):
        assert_allclose(
            model.beta_t(t),
            model.beta_min + t * (model.beta_max - model.beta_min),
        )

    def test_alpha_t(self, model, t):
        assert_allclose(
            model.alpha_t(t),
            t * model.beta_min + 0.5 * t**2 * (model.beta_max - model.beta_min),
        )

    def test_mean_factor(self, model, t):
        assert_allclose(model.mean_factor(t), jnp.exp(-0.5 * model.alpha_t(t)))

    def test_var(self, model, t):
        assert_allclose(model.var(t), 1 - np.exp(-model.alpha_t(t)), atol=1e-5)

    def test_reverse_sde(self, model):
        rng, step_rng = jax.random.split(model.rng)
        initial_samples = jax.random.normal(step_rng, (10, 2))
        samples, _ = model.reverse_sde(initial_samples, lambda x, y: x * y)
        assert (initial_samples != samples).all()


@pytest.mark.parametrize("dim", [2, 5])
@pytest.mark.parametrize("use_prior", [False, True])
class TestDiffusion(TestDiffusionBase):
    CLS = DiffusionModel

    @pytest.fixture
    def prior(self, dim):
        return multivariate_normal(np.zeros(dim))

    @pytest.fixture
    def model(self, prior, use_prior):
        if use_prior:
            return self.CLS(prior=prior)
        else:
            return self.CLS()

    @pytest.fixture
    def batch(self, rng, dim):
        return jax.random.normal(rng, (10, dim))

    @pytest.fixture
    def train_opts(self):
        return {"batch_size": 10, "lr": 1e-3, "n_epochs": 1}

    def test_init_state(self, model, batch):
        model.ndims = batch.shape[-1]
        model._init_state()
        assert model.state is not None

    def test_loss(self, model, rng, batch):
        model.ndims = batch.shape[-1]
        model._init_state()
        if model.prior is not None:
            batch_prior = model.prior.rvs(batch.shape[0])
        else:
            batch_prior = jnp.zeros(batch.shape)
        loss = model.loss(
            model.state.params, batch, batch_prior, model.state.batch_stats, rng
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
