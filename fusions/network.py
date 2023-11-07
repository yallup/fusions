from flax import linen as nn
import jax.numpy as jnp
from tqdm import tqdm
import jax.random as random
import jax
from jax import jit
from functools import partial

# rng = random.PRNGKey(2022)


class ScoreApprox(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    @nn.compact
    def __call__(self, x, t):
        in_size = x.shape[1]
        n_hidden = 256
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
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x
