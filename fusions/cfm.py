from functools import partial

import diffrax as dfx
import jax.numpy as jnp
import jax.random as random
from diffrax.saveat import SaveAt
from jax import grad, jit, pmap, vmap

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
            # solver = dfx.Heun()
            solver = dfx.Dopri5()
            sol = dfx.diffeqsolve(
                term, solver, t0, t1, dt0, x, saveat=SaveAt(t1=True, ts=ts)
            )
            return sol.ys

        yt = vmap(f)(initial_samples)
        return yt[:, -1, :], jnp.moveaxis(yt, 0, 1)

    @partial(jit, static_argnums=[0])
    def loss(self, params, batch, batch_prior, batch_stats, rng):
        """Loss function for training the CFM score."""
        sigma_noise = 1e-3
        rng, step_rng = random.split(rng)
        N_batch = batch.shape[0]

        t = random.uniform(step_rng, (N_batch, 1))
        x0 = batch_prior
        x1 = batch
        noise = random.normal(step_rng, (N_batch, self.ndims))
        psi_0 = t * batch + (1 - t) * batch_prior + sigma_noise * noise
        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            psi_0,
            t,
            train=True,
            mutable=["batch_stats"],
        )
        psi = x1 - x0
        loss = jnp.mean((output - psi) ** 2)
        return loss, updates
