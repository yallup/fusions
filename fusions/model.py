from functools import partial

import anesthetic as ns
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from jax import grad, jit, vmap
from jax.lax import scan
from scipy.stats import norm
from tqdm import tqdm

from fusions.network import ScoreApprox, TrainState


class DiffusionModelBase(object):
    """Base class for the diffusion model.

    Implements the core (non-neural) functionality."""

    def __init__(self, prior=None, **kwargs) -> None:
        """Initialise the diffusion model.

        Args:
            prior (scipy.stats.rv_continuous): Prior distribution to use. Defaults to None.

        Keyword Args:
            steps (int): Number of steps to use in the diffusion model. Defaults to 1000.
        """

        self.chains = None
        self.steps = kwargs.get("steps", 1000)
        # beta_t = jnp.linspace(0.001, 1, self.steps)
        self.beta_min = 1e-3
        self.beta_max = 3
        self.rng = random.PRNGKey(2022)
        R = self.steps
        self.train_ts = jnp.arange(1, R) / (R - 1)
        # self.prior=norm(0,1)
        self.prior = prior
        self.ndims = None

    # def prior(self):

    def _read_chains(self, path: str, ndims: int = None) -> None:
        """Read chains from a file."""
        self.chains = ns.read_chains(path)
        if not ndims:
            self.ndims = self.chains.to_numpy()[..., :-3].shape[-1]
        else:
            self.ndims = ndims
        # beta = jnp.logspace(-5, 0, 1001)
        # D_KL = self.chains.D_KL(beta=beta)
        # new_ds = jnp.linspace(D_KL.min(), D_KL.max(), 100)
        # self.train_ts = jnp.interp(new_ds, D_KL, beta)

    def beta_t(self, t):
        """Beta function of the diffusion model."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_t(self, t):
        """Alpha function of the diffusion model."""
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def mean_factor(self, t):
        """Mean factor of the diffusion model."""
        return jnp.exp(-0.5 * self.alpha_t(t))

    def var(self, t):
        """Variance of the diffusion model."""
        return 1 - jnp.exp(-self.alpha_t(t))

    def drift(self, x, t):
        """Drift of the diffusion model."""
        return -0.5 * self.beta_t(t) * x

    def dispersion(self, t):
        """Dispersion of the diffusion model."""
        return jnp.sqrt(self.beta_t(t))

    @partial(jit, static_argnums=[0, 2])
    def reverse_sde(self, initial_samples, score):
        """Run the reverse SDE.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.
            score (callable): Score function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Samples from the posterior distribution. and the history of the process.
        """

        def f(carry, params):
            t, dt = params
            x, rng = carry
            rng, step_rng = jax.random.split(rng)
            disp = self.dispersion(1 - t)
            t = jnp.ones((x.shape[0], 1)) * t
            drift = -self.drift(x, 1 - t) + disp**2 * score(x, 1 - t)
            noise = random.normal(step_rng, x.shape)
            x = x + dt * drift + jnp.sqrt(dt) * disp * noise
            return (x, rng), (carry)

        rng, step_rng = random.split(self.rng)
        dts = self.train_ts[1:] - self.train_ts[:-1]
        params = jnp.stack([self.train_ts[:-1], dts], axis=1)
        (x, _), (x_t, _) = scan(f, (initial_samples, rng), params)
        return x, x_t

    def sample_prior(self, n):
        """Sample from the prior distribution.

        Args:
            n (int): Number of samples to draw.

        Returns:
            jnp.ndarray: Samples from the prior distribution.
        """
        if self.prior:
            return self.prior.rvs(n)
        else:
            self.rng, step_rng = random.split(self.rng)
            return random.normal(step_rng, (n, self.ndims))

    def sample_posterior(self, n, **kwargs):
        raise NotImplementedError

    # def log_hat_pt(self, data, x, t):
    #     means = data * self.mean_factor(t)
    #     v = self.var(t)
    #     potentials = jnp.sum(-((x - means) ** 2) / (2 * v), axis=1)
    #     return logsumexp(potentials, axis=0, b=1 / self.ndims)


class DiffusionModel(DiffusionModelBase):
    """Extends the base diffusion model to include neural networks to approximate the score."""

    def __init__(self, *args, **kwargs):
        """Initialise the diffusion model."""
        super(DiffusionModel, self).__init__(*args, **kwargs)
        self.state = None

    def score_model(self):
        """Score model for training the diffusion model.

        nb: Due to idosyncrocies in flax relating to batchnorm this can be replaced by any flax.linen.nn,
        but it must have BatchNorm layers (even if they are not used).
        """
        return ScoreApprox()

    @partial(jit, static_argnums=[0])
    def loss(self, params, batch, batch_prior, batch_stats, rng):
        """Loss function for training the diffusion model."""
        rng, step_rng = random.split(rng)
        N_batch = batch.shape[0]
        t = random.randint(step_rng, (N_batch, 1), 1, self.steps) / (self.steps - 1)
        mean_coeff = self.mean_factor(t)
        vs = self.var(t)
        stds = jnp.sqrt(vs)
        rng, step_rng = random.split(rng)

        noise = batch_prior + random.normal(step_rng, batch.shape)
        xt = batch * mean_coeff + noise * stds
        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            xt,
            t,
            train=True,
            mutable=["batch_stats"],
        )

        loss = jnp.mean((noise + output * stds) ** 2)
        return loss, updates

    def _train(self, data, **kwargs):
        """Internal wrapping of training loop."""
        batch_size = kwargs.get("batch_size", 128)
        n_epochs = kwargs.get("n_epochs", 1000)

        @jit
        def update_step(state, batch, batch_prior, rng):
            (val, updates), grads = jax.value_and_grad(self.loss, has_aux=True)(
                state.params, batch, batch_prior, state.batch_stats, rng
            )
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates["batch_stats"])
            return val, state

        train_size = data.shape[0]

        if self.prior:
            prior_samples = jnp.array(self.prior.rvs(train_size))
        else:
            prior_samples = jnp.zeros_like(data)

        batch_size = min(batch_size, train_size)

        steps_per_epoch = train_size // batch_size
        losses = []
        tepochs = tqdm(range(n_epochs))
        for k in tepochs:
            self.rng, step_rng = random.split(self.rng)
            perms = jax.random.permutation(step_rng, train_size)
            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = data[perm, :]

                batch_prior = prior_samples[perm, :]
                self.rng, step_rng = random.split(self.rng)
                loss, self.state = update_step(self.state, batch, batch_prior, step_rng)
                losses.append(loss)
            if (k + 1) % 100 == 0:
                mean_loss = jnp.mean(jnp.array(losses))
                self.state.losses.append((mean_loss, k))
                tepochs.set_postfix(loss=mean_loss)

    def _init_state(self, **kwargs):
        """Initialise the state of the training."""
        dummy_x = jnp.zeros((1, self.ndims))
        dummy_t = jnp.ones((1, 1))

        _params = self.score_model().init(self.rng, dummy_x, dummy_t, train=False)
        lr = kwargs.get("lr", 1e-3)
        optimizer = optax.adam(lr)
        params = _params["params"]
        batch_stats = _params["batch_stats"]

        self.state = TrainState.create(
            apply_fn=self.score_model().apply,
            params=params,
            batch_stats=batch_stats,
            tx=optimizer,
            losses=[],
        )

    def train(self, data, **kwargs):
        """Train the diffusion model on the provided data.

        Args:
            data (jnp.ndarray): Data to train on.

        Keyword Args:
            restart (bool): If True, reinitialise the model before training. Defaults to False.
            batch_size (int): Size of the training batches. Defaults to 128.
            n_epochs (int): Number of training epochs. Defaults to 1000.
            lr (float): Learning rate. Defaults to 1e-3.
        """
        restart = kwargs.get("restart", False)
        self.ndims = data.shape[-1]
        # data = self.chains.sample(200).to_numpy()[..., :-3]
        if (not self.state) | restart:
            self._init_state(**kwargs)

        self._train(data, **kwargs)
        self._predict = lambda x, t: self.state.apply_fn(
            {
                "params": self.state.params,
                "batch_stats": self.state.batch_stats,
            },
            x,
            t,
            train=False,
        )

    def predict(self, initial_samples, **kwargs):
        """Run the diffusion model on user-provided samples.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.

        Keyword Args:
            history (bool): If True, return the history of the process as well as the outpute (tuple).
                Defaults to False.

        Returns:
            jnp.ndarray: Samples from the posterior distribution.
        """
        hist = kwargs.get("history", False)
        x, x_t = self.reverse_sde(initial_samples, self._predict)
        if hist:
            return x, x_t
        else:
            return x

    def sample_posterior(self, n, **kwargs):
        """Draw samples from the posterior distribution.

        Args:
            n (int): Number of samples to draw.

        Keyword Args:
            history (bool):  return the history of the process as well as the outpute (tuple).
                Defaults to False.

        Returns:
            jnp.ndarray: Samples from the posterior distribution.
        """
        return self.predict(self.sample_prior(n), **kwargs)

    def rvs(self, n):
        """Alias for sample_posterior.

        Args:
            n (int): Number of samples to draw.

        Returns:
            jnp.ndarray: Samples from the posterior distribution.
        """
        return self.sample_posterior(n)
