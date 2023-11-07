from functools import partial
from typing import Any

import anesthetic as ns
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import optax
from flax.training import train_state
from jax import grad, jit, vmap
from jax.lax import scan
from jax.scipy.special import logsumexp
from tqdm import tqdm

from fusions.network import ScoreApprox


class DiffusionModelBase(object):
    def __init__(self, **kwargs) -> None:
        self.chains = None
        self.steps = kwargs.get("steps", 100)
        # beta_t = jnp.linspace(0.001, 1, self.steps)
        self.beta_min = 1e-3
        self.beta_max = 1
        self.rng = random.PRNGKey(2022)
        self.train_ts = jnp.linspace(self.beta_min, self.beta_max, self.steps * 10)
        self.ndims = None

    def read_chains(self, path: str, ndims: int = None) -> None:
        self.chains = ns.read_chains(path)
        if not ndims:
            self.ndims = self.chains.to_numpy()[..., :-3].shape[-1]
        else:
            self.ndims = ndims

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_t(self, t):
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def mean_factor(self, t):
        return jnp.exp(-0.5 * self.alpha_t(t))

    def var(self, t):
        return 1 - jnp.exp(-self.alpha_t(t))

    def drift(self, x, t):
        return -0.5 * self.beta_t(t) * x

    def dispersion(self, t):
        return jnp.sqrt(self.beta_t(t))

    @partial(jit, static_argnums=[0, 2])
    def reverse_sde(self, initial_samples, score):
        def f(carry, params):
            t, dt = params
            x, rng = carry
            rng, step_rng = jax.random.split(rng)
            disp = self.dispersion(1 - t)
            t = jnp.ones((x.shape[0], 1)) * t
            drift = -self.drift(x, 1 - t) + disp**2 * score(x, 1 - t)
            noise = random.normal(step_rng, x.shape)
            x = x + dt * drift + jnp.sqrt(dt) * disp * noise
            return (x, rng), ()

        rng, step_rng = random.split(self.rng)
        # initial = random.normal(step_rng, (self.n_samples, self.N))
        dts = self.train_ts[1:] - self.train_ts[:-1]
        params = jnp.stack([self.train_ts[:-1], dts], axis=1)
        (x, _), _ = scan(f, (initial_samples, rng), params)
        return x


class DiffusionModel(DiffusionModelBase):
    def score_model(self):
        return ScoreApprox()

    @partial(jit, static_argnums=[0])
    def loss(self, params, batch, batch_stats):
        rng, step_rng = random.split(self.rng)
        N_batch = batch.shape[0]
        t = random.randint(step_rng, (N_batch, 1), 1, self.steps) / (self.steps - 1)
        mean_coeff = self.mean_factor(t)
        # is it right to have the square root here for the loss?
        vs = self.var(t)
        stds = jnp.sqrt(vs)
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, batch.shape)
        xt = batch * mean_coeff + noise * stds
        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            xt,
            t,
            train=True,
            mutable=["batch_stats"],
        )
        # output, updates = self.score_model().apply(
        #     {"params": params, "batch_stats": batch_stats},
        #     xt,
        #     t,
        #     train=True,
        #     mutable=["batch_stats"],
        # )

        loss = jnp.mean((noise + output * stds) ** 2)
        return loss, updates

    def _train(self, data, batch_size=128, N_epochs=1000):
        dummy_x = jnp.zeros(self.ndims * batch_size).reshape(
            (batch_size, data.shape[-1])
        )
        dummy_t = jnp.ones((batch_size, 1))

        _params = self.score_model().init(self.rng, dummy_x, dummy_t, train=False)
        optimizer = optax.adam(1e-3)
        params = _params["params"]
        batch_stats = _params["batch_stats"]

        class TrainState(train_state.TrainState):
            batch_stats: Any

        self.state = TrainState.create(
            apply_fn=self.score_model().apply,
            params=params,
            batch_stats=batch_stats,
            tx=optimizer,
        )
        # opt_state = optimizer.init(params)

        @jit
        def update_step(state, batch):
            (val, updates), grads = jax.value_and_grad(self.loss, has_aux=True)(
                state.params, batch, state.batch_stats
            )
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates["batch_stats"])
            # updates, opt_state = optimizer.update(grads, state)
            # params = optax.apply_updates(params, updates)
            return val, state

        train_size = data.shape[0]
        steps_per_epoch = train_size // batch_size
        losses = []
        for k in tqdm(range(N_epochs)):
            rng, step_rng = random.split(self.rng)
            perms = jax.random.permutation(step_rng, train_size)
            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = data[perm, :]
                rng, step_rng = random.split(rng)
                loss, self.state = update_step(self.state, batch)
                losses.append(loss)
            if (k + 1) % 100 == 0:
                mean_loss = jnp.mean(jnp.array(losses))
                print("Epoch %d \t, Loss %f " % (k + 1, mean_loss))
                losses = []

        return self.state.params

    def train(self):
        trained_params = self._train(self.chains.to_numpy()[..., :-3])

        self._predict = lambda x, t: self.state.apply_fn(
            {"params": trained_params, "batch_stats": self.state.batch_stats},
            x,
            t,
            train=False,
        )

    def predict(self, initial_samples):
        return self.reverse_sde(initial_samples, self._predict)
