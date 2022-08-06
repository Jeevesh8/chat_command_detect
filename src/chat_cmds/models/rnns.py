from typing import Callable

import jax

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx

from .utils import flip_padded_seq


class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module

    def __init__(self, in_size, hidden_size, cell_fn, *, key):
        self.hidden_size = hidden_size
        self.cell_fn = cell_fn
        self.cell = self.cell_fn(in_size, hidden_size, key=key)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = lax.scan(f, hidden, input)

        return out


class BiRNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module

    def __init__(self, in_size, hidden_size, cell_fn, *, key):
        f_key, b_key = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell_fn = cell_fn
        self.f_cell = self.cell_fn(in_size, hidden_size // 2, key=f_key)
        self.b_cell = self.cell_fn(
            in_size, hidden_size // 2 + hidden_size % 2, key=b_key
        )

    def run_cell(self, input, forward: bool = True):
        hidden = jnp.zeros(
            (self.hidden_size // 2 + (self.hidden_size % 2 and not forward),)
        )

        def f(carry, inp):
            if forward:
                return self.f_cell(inp, carry), None
            else:
                return self.b_cell(inp, carry), None

        out, _ = lax.scan(f, hidden, input)

        return out

    def __call__(self, input, length):
        forward_out = self.run_cell(input)
        input = flip_padded_seq(input, length)
        backward_out = self.run_cell(input, False)
        return jnp.concatenate([forward_out, backward_out], axis=-1)
