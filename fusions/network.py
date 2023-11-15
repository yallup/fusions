from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from flax.training import train_state


# class DataLoader(object):
#     def __init__(self, data, batch_size, rng, shuffle=True) -> None:
#         self.data = jnp.array(data)
#         self.rng = rng
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.i = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.i >= len(self.data):
#             self.i = 0
#             if self.shuffle:
#                 self.rng, rng = jax.random.split(self.rng)
#                 perm = jax.random.permutation(rng, len(self.data))
#                 self.data = self.data[perm]
#         batch = self.data[self.i : self.i + self.batch_size]
#         self.i += self.batch_size
#         if len(batch) != self.batch_size:
#             raise StopIteration
#         return batch


class TrainState(train_state.TrainState):
    batch_stats: Any
    losses: Any


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
        # t = jnp.concatenate(
        #     [
        #         t - 0.5,
        #         jnp.cos(2 * jnp.pi * t),
        #         jnp.sin(2 * jnp.pi * t),
        #         -jnp.cos(4 * jnp.pi * t),
        #     ],
        #     axis=1,
        # )
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
