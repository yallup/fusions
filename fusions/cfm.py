import os
from functools import partial
from multiprocessing import Pool

import anesthetic as ns
import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from diffrax.saveat import SaveAt
from jax import grad, jit, pmap, vmap
from ott.geometry import pointcloud
from ott.solvers import linear

from fusions.model import Model


class CFM(Model):
    """Continuous Flows Model."""

    @partial(jit, static_argnums=[0, 2])
    def reverse_process(self, initial_samples, score, rng):
        """Run the reverse ODE.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.
            score (callable): Score function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Samples from the posterior distribution. and the history of the process.
        """
        t0, t1, dt0 = 0.0, 1.0, 1e-3
        ts = jnp.linspace(t0, t1, 100)

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

        yt = vmap(f)(initial_samples)
        return yt[:, -1, :], jnp.moveaxis(yt, 0, 1)

    @partial(jit, static_argnums=[0])
    def loss(self, params, batch, batch_prior, batch_stats, rng):
        """Loss function for training the CFM score."""
        sigma_min = 1e-3
        rng, step_rng = random.split(rng)
        N_batch = batch.shape[0]

        t = random.uniform(step_rng, (N_batch, 1))
        # t = (t) ** (1 / alpha)
        # top_n = 5
        # t = random.uniform(step_rng, (N_batch * top_n, 1))

        # minibatch ot
        # geom = pointcloud.PointCloud(batch_prior, batch)
        # A = linear.solve(geom)
        # _, idx = jax.lax.top_k(A.matrix, top_n)

        # idx = jnp.argmax(A.matrix, axis=-1)
        # x0 = batch_prior[idx].reshape(-1, batch.shape[-1])
        # t = t.reshape(-1, 1)
        x0 = batch_prior
        x1 = batch
        # x1 = jnp.stack([x1 for i in range(top_n)]).reshape(-1, batch.shape[-1])
        # batch_prior = random.normal(step_rng, (N_batch, self.ndims))
        # noise = random.normal(step_rng, (N_batch * top_n, self.ndims))
        # x0 = x0+ 1e-3 *noise
        # noise_1 = random.normal(step_rng, (N_batch, self.ndims))
        # x1 = x1+ 1e-3 *noise_1
        # psi_0 = (1 - (1 - sigma_min) * t) * x1 + (x0) * t #+ 0.1 * noise
        psi_0 = (1 - (1 - sigma_min) * t) * x1 + (x0) * t  # + sigma_min * noise
        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            psi_0,
            t,
            train=True,
            mutable=["batch_stats"],
        )
        psi = (1 - sigma_min) * x1 - (x0)  # +  self.sigma * noise)
        # psi = ((1 - sigma_min) * x1 - (x0)).reshape(-1, batch.shape[-1])
        loss = jnp.mean((output - psi) ** 2)
        return loss, updates


# class StochasticCFM(CFM):
#     @partial(jit, static_argnums=[0, 2])
#     def reverse_process(self, initial_samples, score):
#         def f(carry, params):
#             t, dt = params
#             x, rng = carry
#             rng, step_rng = jax.random.split(rng)
#             disp = self.dispersion(1 - t)
#             t = jnp.ones((x.shape[0], 1)) * t
#             drift = -self.drift(x, 1 - t) + disp**2 * score(
#                 x, initial_samples, 1 - t
#             )
#             noise = random.normal(step_rng, x.shape)
#             x = x + dt * drift + jnp.sqrt(dt) * disp * noise
#             return (x, rng), (carry)

#         rng, step_rng = random.split(self.rng)
#         dts = self.train_ts[1:] - self.train_ts[:-1]
#         params = jnp.stack([self.train_ts[:-1], dts], axis=1)
#         (x, _), (x_t, _) = scan(f, (initial_samples, rng), params)
#         return x, x_t
