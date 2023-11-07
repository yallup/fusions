from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax import jit
from tqdm import tqdm

# rng = random.PRNGKey(2022)


class ScoreApprox(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    n_initial: int = 256
    n_hidden: int = 16
    act = nn.leaky_relu

    @nn.compact
    def __call__(self, x, t, train: bool):
        in_size = x.shape[1]
        # act = nn.relu
        # t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)],axis=1)
        t = jnp.concatenate(
            [
                t - 0.5,
                jnp.cos(2 * jnp.pi * t),
                jnp.sin(2 * jnp.pi * t),
                -jnp.cos(4 * jnp.pi * t),
            ],
            axis=1,
        )
        x = jnp.concatenate([x, t], axis=1)
        x = nn.Dense(self.n_initial)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x
