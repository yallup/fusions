from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit


def plot_score(score, t, axes, area_min=-1, area_max=1):
    @partial(
        jit,
        static_argnums=[
            0,
        ],
    )
    def helper(score, t, area_min, area_max):
        x = jnp.linspace(area_min, area_max, 16)
        x, y = jnp.meshgrid(x, x)
        grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
        t = jnp.ones((grid.shape[0], 1)) * t
        scores = score(grid, t)
        return grid, scores

    grid, scores = helper(score, t, area_min, area_max)
    return axes.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 1])
