import math
from functools import partial

import diffrax as dfx
from diffrax.saveat import SaveAt

import jax.numpy as jnp
import jax.random as random
from fusions.model import Model
from jax import disable_jit, grad, jit, pmap, vjp, vmap


class CFM(Model):
    """Continuous Flow Matching."""

    @partial(jit, static_argnums=[0, 2, 4, 5])
    def reverse_process(self, initial_samples, score, rng, steps=0, solution="exact"):
        """Run the reverse ODE.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.
            score (callable): Score function.
            rng: Jax Random number generator key.

        Keyword Args:
            steps (int, optional) : Number of time steps to save in addition to t=1. Defaults to 0.
            solution (str, optional): Method to use for the jacobian. Defaults to "exact".
                        one of "exact", "none", "approx".

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Samples from the posterior distribution. and the history of the process.
        """
        t0, t1, dt0 = 0.0, 1.0, 1e-3
        ts = jnp.linspace(t0, t1, steps)

        def solver_none(ti, conditions, args):
            xi, null_jac = conditions
            return score(xi, jnp.atleast_1d(ti)), null_jac

        def solver_exact(ti, conditions, args):
            xi, _ = conditions
            f, vjp_f = vjp(score, xi, jnp.atleast_1d(ti))
            (size,) = xi.shape
            eye = jnp.eye(size)
            (dfdx, _) = vmap(vjp_f)(eye)
            logp = jnp.trace(dfdx)
            return f, logp

        def solver_approx(ti, conditions, args):
            eps = args
            xi, _ = conditions
            f, vjp_f = vjp(score, xi, jnp.atleast_1d(ti))
            eps_dfdx, _ = vjp_f(eps)
            logp = jnp.sum(eps_dfdx * eps)
            return f, logp

        def f(x, eps):
            jacobian = 0.0
            conditions = (x, jacobian)
            if solution == "exact":
                term = dfx.ODETerm(solver_exact)
            elif solution == "approx":
                term = dfx.ODETerm(solver_approx)
            else:
                term = dfx.ODETerm(solver_none)
            # term = dfx.ODETerm(score_args_exact)
            # solver = dfx.Heun()
            solver = dfx.Dopri5()

            solver_approx

            sol = dfx.diffeqsolve(
                term,
                solver,
                t0,
                t1,
                dt0,
                conditions,
                args=eps,
                saveat=SaveAt(t1=True, ts=ts),
            )
            return sol.ys

        # batch_rngs = random.split(rng, initial_samples.shape[0])
        eps = random.normal(rng, initial_samples.shape)
        yt, jt = vmap(f)(initial_samples, eps)
        return yt, jt

    @partial(jit, static_argnums=[0])
    def loss(self, params, batch, batch_prior, batch_stats, rng):
        """Loss function for training the CFM score.

        Args:
            params (jnp.ndarray): Parameters of the model.
            batch (jnp.ndarray): Target batch.
            batch_prior (jnp.ndarray): Prior batch.
            batch_stats (Any): Batch statistics (batchnorm running totals).
            rng: Jax Random number generator key.

        """
        # sigma_noise = 1e-3
        rng, step_rng = random.split(rng)
        N_batch = batch.shape[0]

        t = random.uniform(step_rng, (N_batch, 1))
        x0 = batch_prior
        x1 = batch
        noise = random.normal(step_rng, (N_batch, self.ndims))
        # psi_0 = t * batch + (1 - t) * batch_prior + sigma_noise * noise
        psi_0 = t * batch + (1 - t) * batch_prior + self.noise * noise
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
