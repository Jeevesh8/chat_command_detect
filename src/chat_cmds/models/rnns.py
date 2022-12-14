from typing import Callable

import jax

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx

from .utils import flip_padded_seq


class RNN(eqx.Module):
    hidden_size: int = eqx.static_field()
    cell: eqx.Module

    def __init__(self, in_size, hidden_size, cell_fn, *, key):
        self.hidden_size = hidden_size
        self.cell = cell_fn(in_size, hidden_size, key=key)

    def __call__(self, input, *, key):
        init_state = (
            jnp.zeros(
                self.hidden_size,
            ),
            jnp.zeros(
                self.hidden_size,
            ),
        )

        def f(carry, inp):
            cell_out = self.cell(inp, carry)
            return (cell_out, cell_out[0])

        return lax.scan(f, init_state, input)[1]


class BiRNN(eqx.Module):
    hidden_size: int = eqx.static_field()
    f_cell: eqx.Module
    b_cell: eqx.Module

    def __init__(self, in_size, hidden_size, cell_fn, *, key):
        f_key, b_key = jrandom.split(key)
        self.hidden_size = hidden_size
        self.f_cell = cell_fn(in_size, hidden_size // 2, key=f_key)
        self.b_cell = cell_fn(in_size, hidden_size // 2 + hidden_size % 2, key=b_key)

    def run_cell(self, input, forward: bool = True):
        hidden_size = self.hidden_size // 2 + (self.hidden_size % 2 and not forward)

        init_state = (
            jnp.zeros(
                hidden_size,
            ),
            jnp.zeros(
                hidden_size,
            ),
        )

        def f(carry, inp):
            if forward:
                cell_out = self.f_cell(inp, carry)
            else:
                cell_out = self.b_cell(inp, carry)
            return (cell_out, cell_out[0])

        return lax.scan(f, init_state, input)[1]

    def __call__(self, input, *, key):
        length = jnp.sum(input.any(axis=-1))
        forward_out = self.run_cell(input)
        input = flip_padded_seq(input, length)
        backward_out = self.run_cell(input, False)
        return jnp.concatenate([forward_out, backward_out], axis=-1)

class pick_index(eqx.Module):
    idx: int = eqx.static_field()
    
    def __call__(self, inputs, *, key):
        return inputs[self.idx]