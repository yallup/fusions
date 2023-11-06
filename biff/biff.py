import anesthetic as ns
from jax import grad, jit, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from functools import partial
from jax.lax import scan
import jax.random as random
import jax
from tqdm import tqdm
from biff.network import ScoreApprox


import optax



class DiffusionModelBase(object):
    def __init__(self, **kwargs) -> None:
        self.chains = None
        self.steps = kwargs.get("steps", 100)
        beta_t = jnp.linspace(0.001, 1, self.steps)
        self.beta_min = 1e-3
        self.beta_max = 1
        self.rng = random.PRNGKey(2022)
        self.train_ts = jnp.linspace(self.beta_min, self.beta_max, self.steps)

    def read_chains(self, path: str) -> None:
        self.chains = ns.read_chains(path)

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_t(self, t):
        return t * self.beta_min + 0.5 * t**2 * (
            self.beta_max - self.beta_min
        )

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
    def loss(self, params, batch):
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
        output = self.score_model().apply(params, xt, t)
        loss = jnp.mean((noise + output * stds) ** 2)
        return loss

    def _train(self, data, batch_size=128, N_epochs=1000):

        dummy_x = jnp.zeros(2 * batch_size).reshape((batch_size, data.shape[-1]))
        dummy_t = jnp.ones((batch_size, 1))
        
        params = self.score_model().init(self.rng, dummy_x, dummy_t)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        
        @jit
        def update_step(params, batch, opt_state):
            val, grads = jax.value_and_grad(self.loss)(params,batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return val, params, opt_state

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
                loss, params, opt_state = update_step(params, batch, opt_state)
                losses.append(loss)
            if (k + 1) % 100 == 0:
                mean_loss = jnp.mean(jnp.array(losses))
                print("Epoch %d \t, Loss %f " % (k + 1, mean_loss))
                losses = []

        return params

    def train(self):
        self._train(self.chains.to_numpy()[...,:-3])