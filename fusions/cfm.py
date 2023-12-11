from functools import partial

import anesthetic as ns
import diffrax as dfx
import distrax
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from diffrax.saveat import SaveAt
from jax import grad, jit, vmap
from jax.lax import scan
from ott.geometry import pointcloud
from ott.solvers import linear, was_solver
from scipy.stats import norm
from tqdm import tqdm

from fusions.network import ScoreApprox, ScoreApproxCNN, ScorePriorApprox, TrainState
from fusions.optimal_transport import OTMap


class CFMBase(object):
    """Base class for the continuous flow model.

    Implements the core (non-neural) functionality."""

    def __init__(self, prior=None, **kwargs) -> None:
        """Initialise the flow model.

        Args:
            prior (scipy.stats.rv_continuous): Prior distribution to use. Defaults to None.

        Keyword Args:
            steps (int): Number of steps to use in the diffusion model. Defaults to 1000.
        """

        self.chains = None
        # self.prior=norm(0,1)
        self.prior = prior
        self.ndims = None
        self.rng = random.PRNGKey(2022)
        self.state = None
        self.sigma = 0.1

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
        x, xt = self.reverse_sde(initial_samples, self._predict)
        if hist:
            return x, xt
        else:
            return x

    def sample_posterior(self, n, **kwargs):
        return self.predict(self.sample_prior(n), **kwargs)

    def score_model(self):
        """Score model for training the diffusion model.

        nb: Due to idosyncrocies in flax relating to batchnorm this can be replaced by any flax.linen.nn,
        but it must have BatchNorm layers (even if they are not used).
        """
        return ScoreApprox()
        # return ScoreApproxCNN()

    @partial(jit, static_argnums=[0, 2])
    def reverse_sde(self, initial_samples, score):
        """Run the reverse SDE.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.
            score (callable): Score function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Samples from the posterior distribution. and the history of the process.
        """
        rng, step_rng = random.split(self.rng)
        t0, t1, dt0 = 0.0, 1.0, 1e-3
        ts = jnp.linspace(t0, t1, 10)

        def f(x):
            # return score(x, jnp.atleast_1d(t))
            def score_args(ti, xi, args):
                return score(xi, jnp.atleast_1d(ti))

            term = dfx.ODETerm(score_args)
            solver = dfx.Heun()
            # solver = dfx.Dopri5()
            sol = dfx.diffeqsolve(
                term, solver, t0, t1, dt0, x, saveat=SaveAt(t1=True, ts=ts)
            )
            return sol.ys
            # return y

        yt = vmap(f)(initial_samples)
        # scan(f, initial_samples)
        return yt[:, -1, :], yt

    def _ot_map(self, x0, x1):
        geom = pointcloud.PointCloud(x0, x1)

    # @partial(jit, static_argnums=[0])
    def loss(self, params, batch, batch_prior, batch_stats, rng):
        """Loss function for training the diffusion model."""
        rng, step_rng = random.split(rng)
        sigma_min = 0.0  # 1e-2
        N_batch = batch.shape[0]
        t = random.uniform(step_rng, (N_batch, 1))

        # geom = pointcloud.PointCloud(batch_prior, batch)
        # A = linear.solve(geom)
        # idx = jnp.argmax(A.matrix, axis=-1)
        # x0 = batch_prior[idx]

        x0 = batch_prior
        x1 = batch
        # batch_prior = random.normal(step_rng, (N_batch, self.ndims))
        noise = random.normal(step_rng, (N_batch, self.ndims))
        # x0 = x0+ 1e-3 *noise
        noise_1 = random.normal(step_rng, (N_batch, self.ndims))
        # x1 = x1+ 1e-3 *noise_1
        psi_0 = (1 - (1 - sigma_min) * t) * x1 + (x0) * t + 0.1 * noise
        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            psi_0,
            t,
            train=True,
            mutable=["batch_stats"],
        )
        psi = (1 - sigma_min) * x1 - (x0)  # +  self.sigma * noise)
        loss = jnp.mean((output - psi) ** 2)
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
        ot_map = OTMap(prior_samples, data)
        tepochs = tqdm(range(n_epochs))
        for k in tepochs:
            self.rng, step_rng = random.split(self.rng)
            perm_prior, perm = ot_map.sample_flat(step_rng, train_size, batch_size)
            # perm_prior, perm = ot_map.sample_np(step_rng, batch_size)
            # perm_prior, perm = ot_map.sample_matrix(step_rng, batch_size)
            # if (k + 1) % 1 == 0:
            #     perm_prior, perm = ot_map.sample_flat(step_rng, batch_size)
            # else:
            #     perm_prior, perm = ot_map.sample_np(step_rng,batch_size=batch_size)
            batch = data[perm, :]
            batch_prior = prior_samples[perm_prior, :]
            # batch_prior = jnp.array(self.prior.rvs(batch_size))
            loss, self.state = update_step(self.state, batch, batch_prior, step_rng)
            losses.append(loss)
            # perms = jax.random.permutation(step_rng, train_size)
            # perms = perms[
            #     : steps_per_epoch * batch_size
            # ]  # skip incomplete batch
            # perms = perms.reshape((steps_per_epoch, batch_size))

            # for perm in perms:
            #     batch = data[perm, :]

            #     batch_prior = prior_samples[perm, :]
            #     self.rng, step_rng = random.split(self.rng)
            #     loss, self.state = update_step(
            #         self.state, batch, batch_prior, step_rng
            #     )
            # losses.append(loss)
            # if (k + 1) % 100 == 0:
            #     mean_loss = jnp.mean(jnp.array(losses))
            #     self.state.losses.append((mean_loss, k))
            #     tepochs.set_postfix(loss=mean_loss)

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

    def rvs(self, n):
        """Alias for sample_posterior.

        Args:
            n (int): Number of samples to draw.

        Returns:
            jnp.ndarray: Samples from the posterior distribution.
        """
        return self.sample_posterior(n)
