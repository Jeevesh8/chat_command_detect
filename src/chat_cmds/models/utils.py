from typing import List, Optional

import jax

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx


class NFoldHead(eqx.Module):
    heads: List[eqx.nn.Linear]
    names: List[str]

    def __init__(self, input_size, out_sizes, use_bias, names, *, key):
        keys = jrandom.split(key, len(out_sizes))
        self.names = names
        self.heads = [
            eqx.nn.Linear(input_size, output_size, use_bias=use_bias, key=keys[i])
            for i, output_size in enumerate(out_sizes)
        ]

    def __call__(self, input):
        return {name: head(input) for head, name in zip(self.heads, self.names)}


def flip_padded_seq(seq: jnp.ndarray, length: int = 0) -> jnp.ndarray:
    return jnp.flip(jnp.roll(seq, seq.shape[0] - length))
