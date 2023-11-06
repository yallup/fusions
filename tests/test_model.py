from biff.biff import DiffusionModelBase

import numpy as np
import jax.numpy as jnp
import jax
import jax.random as random
import scipy.stats
from numpy.testing import assert_allclose
import pytest
import os

test_path = os.path.dirname(os.path.realpath(__file__))


class TestDiffusionBase(object):
    CLS = DiffusionModelBase

    @pytest.fixture
    def model(self):
        return self.CLS()

    @pytest.fixture
    def t(self):
        return jnp.linspace(0, 1, 100)

    def test_read_chains(self, model):
        model.read_chains(os.path.join(test_path, "data/data"))
        assert model.chains is not None

    def test_beta_t(self, model, t):
        assert_allclose(model.beta_t(t), 0.001 + t * (1 - 0.001))

    def test_alpha_t(self, model, t):
        assert_allclose(
            model.alpha_t(t), t * 0.001 + 0.5 * t**2 * (1 - 0.001)
        )

    def test_mean_factor(self, model, t):
        assert_allclose(model.mean_factor(t), jnp.exp(-0.5 * model.alpha_t(t)))

    def test_var(self, model, t):
        assert_allclose(model.var(t), 1 - np.exp(-model.alpha_t(t)), atol=1e-5)

    def test_reverse_sde(self, model):
        rng, step_rng = jax.random.split(model.rng)
        initial_samples = jax.random.normal(step_rng, (10, 2))
        samples = model.reverse_sde(initial_samples, lambda x, y: x * y)
        assert (initial_samples != samples).all()
