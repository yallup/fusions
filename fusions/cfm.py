import math
from functools import partial

import diffrax as dfx
import flax.linen as nn
import jax.numpy as jnp
import jax.random as random
from diffrax import (
    ControlTerm,
    Euler,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)
from diffrax.saveat import SaveAt
from jax import disable_jit, grad, jit, pmap, vjp, vmap
from jax.lax import scan

from fusions.model import Model


class CFM(Model):
    """Continuous Flow Matching."""

    # @partial(jit, static_argnums=[0,2,4,5])
    def mala(self, initial_samples, score, rng, N=1e3, eps=1e-3):
        """perform n steps of a Metropolis-adjusted Langevin algorithm.

        Args:
            x (jnp.ndarray): Initial sample.
            score (callable): Score function.
            rng: Jax Random number generator key.

        Keyword Args:
            eps (float, optional): Step size. Defaults to 1e-3.

        Returns:
            jnp.ndarray: Sample from the posterior distribution.
        """

        def f(carry, params):
            accept_step = params
            x, rng = carry
            rng, step_rng = random.split(rng)
            # metropolis adjustment
            initial_score = score(x, jnp.ones(x.shape[0])[..., None])
            x_prop = (
                x
                + 0.5 * eps * initial_score
                + jnp.sqrt(eps) * random.normal(step_rng, x.shape)
            )
            score_prop = score(x_prop, jnp.ones(x.shape[0])[..., None])
            log_accept_prob_backward = (
                -1
                / (2 * eps)
                * jnp.linalg.norm(x - x_prop - 0.5 * eps * score_prop, axis=-1) ** 2
            )
            log_accept_prob_forward = (
                -1
                / (2 * eps)
                * jnp.linalg.norm(x_prop - x - 0.5 * eps * initial_score, axis=-1) ** 2
            )
            accept_prob = jnp.min(
                jnp.asarray(
                    (
                        jnp.ones(x.shape[0]),
                        jnp.exp(log_accept_prob_backward - log_accept_prob_forward),
                    )
                ),
                axis=0,
            )
            # accept = random.bernoulli(rng, accept_prob)
            accept = random.bernoulli(rng, accept_prob)
            x = jnp.where(accept[..., None], x_prop, x)
            # accept_step += jnp.where(accept, 1, 0)
            return (x, rng), (carry)

        rng, step_rng = random.split(rng)
        # initial_samples = random.normal(rng, initial_samples.shape)
        # f((initial_samples, step_rng), initial_samples)
        # ts = jnp.ones(initial_samples.shape[0])
        # rng, step_rng = random.split(step_rng)
        params = jnp.ones(int(N))
        # dts = self.train_ts[1:] - self.train_ts[:-1]
        # params = jnp.stack([self.train_ts[:-1], dts], axis=1)
        (x, _), (x_t, _) = scan(f, (initial_samples, step_rng), params)
        return x

        # N = x.shape[0]
        # eps = jnp.ones(N) * eps
        # eps = jnp.expand_dims(eps, 1)
        # noise = random.normal(rng, x.shape)
        # x = x + 0.5 * eps * score(x) + jnp.sqrt(eps) * noise
        # return x

    def reverse_stochastic_process(
        self, initial_samples, score, rng, steps=0, solution="none"
    ):
        """Run the reverse SDE. This is a stochastic version of the reverse process."""

        t0, t1 = 0, 1
        # drift = lambda t, y, args: -y
        # diffusion = lambda t, y, args: 0.1 * t
        rng, step_rng = random.split(rng)
        # brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=step_rng)
        # terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
        solver = Euler()

        def f(x, key):
            eps = 1

            def wrapped_score(t, x, args):
                return 0.5 * eps * score(x, jnp.atleast_1d(t))

            conditions = x

            drift = ODETerm(wrapped_score)
            diffusion = lambda t, y, args: eps * y
            brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=key)
            terms = MultiTerm(drift, ControlTerm(diffusion, brownian_motion))
            solver = Euler()
            solver = dfx.ItoMilstein()
            # saveat = SaveAt(dense=True)

            sol = diffeqsolve(
                terms,
                solver,
                t0,
                t1,
                dt0=0.01,
                y0=conditions,  # , saveat=saveat
            )
            return sol.ys

        # batch_rngs = random.split(rng, initial_samples.shape[0])
        # generate a batch of rng keys
        keys = random.split(rng, initial_samples.shape[0])
        # eps = random.normal(rng, initial_samples.shape)
        # f(initial_samples[0], keys[0])
        xt = vmap(f)(initial_samples, keys)
        return xt.squeeze()

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
            # solver = dfx.Dopri8()
            # solver = dfx.Tsit5()

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
    def loss(self, params, batch, batch_prior, rng):
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
        # psi_0 = t * batch + (1 - t) * batch_prior

        output, updates = self.state.apply_fn(
            {"params": params},
            psi_0,
            t,
            mutable=["batch_stats"],
        )
        psi = x1 - x0
        loss = jnp.mean((output - psi) ** 2)
        return loss, updates

    @partial(jit, static_argnums=[0, 2, 3, 5, 6, 7])
    def guided_reverse_process(
        self,
        initial_samples,
        score,
        guide_score,
        rng,
        steps=0,
        solution="none",
        guidance_strength=1.0,
    ):
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
            combined_score = score(
                xi, jnp.atleast_1d(ti)
            ) + guidance_strength * guide_score(xi, jnp.array(0))
            return combined_score, null_jac

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
            # solver = dfx.Dopri8()
            # solver = dfx.Tsit5()

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
        return yt.squeeze(), jt
