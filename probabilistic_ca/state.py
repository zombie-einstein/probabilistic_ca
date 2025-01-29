from abc import ABC, abstractmethod

import chex
import jax
import jax.numpy as jnp


class StateInit(ABC):
    @abstractmethod
    def __call__(self, k: chex.PRNGKey, width: int) -> chex.Array:
        ...


class RandomChoice(StateInit):
    def __call__(self, k: chex.PRNGKey, width: int) -> chex.Array:
        rand_probs = jax.random.choice(k, jnp.array([0.0, 1.0]), (width,))
        rand_probs = jnp.vstack([rand_probs, 1.0 - rand_probs])
        return rand_probs


class RandomUniform(StateInit):
    def __call__(self, k: chex.PRNGKey, width: int) -> chex.Array:
        rand_probs = jax.random.uniform(k, (width,))
        rand_probs = jnp.vstack([rand_probs, 1.0 - rand_probs])
        return rand_probs


class ChoiceWSet(StateInit):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, k: chex.PRNGKey, width: int) -> chex.Array:
        mid = width // 2
        rand_probs = jax.random.choice(k, jnp.array([0.0, 1.0]), (width,))
        rand_probs = rand_probs.at[mid].set(self.p)
        rand_probs = jnp.vstack([rand_probs, 1.0 - rand_probs])
        return rand_probs


class UniformWSet(StateInit):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, k: chex.PRNGKey, width: int) -> chex.Array:
        mid = width // 2
        rand_probs = jnp.zeros((width,)).at[mid].set(self.p)
        rand_probs = jnp.vstack([rand_probs, 1.0 - rand_probs])
        return rand_probs
