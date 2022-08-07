from functools import partial

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from chat_cmds.train_utils import cross_entropy_loss

def get_rnn_train_step(optim, num_labels):
    
    @eqx.filter_pmap(args=(None, None,), axis_name="device_axis", donate_argnums=(0,), out=None)
    def train_step(model, opt_state, x, y):
        
        @eqx.filter_value_and_grad
        def loss_fn(model, x, y,):
            logits = jax.vmap(model,)(x)
            loss = jtu.tree_reduce(lambda w,z: w+z, jtu.tree_map(cross_entropy_loss, logits, y, num_labels))
            return loss
        
        loss, grads = loss_fn(model, x, y)
        grads = jax.lax.pmean(grads, axis_name="device_axis")
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        loss = jax.lax.pmean(loss, axis_name="device_axis")
        return loss, model, opt_state
    
    return train_step

def get_rnn_eval_step():
    
    @eqx.filter_pmap(args=(None,), axis_name="device_axis",)
    def eval_step(model, x,):
        logits = jax.vmap(model,)(x)
        return jtu.tree_map(lambda z: jnp.argmax(z, axis=-1), logits)
    
    return eval_step

def get_trfrmr_train_step(num_labels):
    
    @partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
    def train_step(state, batch, dropout_rng):
        """Trains model with an optimizer (both in `state`) on `batch`, 
           returning a pair `(new_state, loss)`."""
        
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        targets = batch.pop("labels")

        def loss_fn(params):
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            logits = state.head_separator(logits)
            loss = jtu.tree_reduce(lambda w,z: w+z, state.loss_fn(logits, targets, num_labels))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)
        metrics = jax.lax.pmean({"loss": loss,}, axis_name="batch")
        return new_state, metrics, new_dropout_rng
    
    return train_step

def get_trfrmr_eval_step(num_labels):
    
    @partial(jax.pmap, axis_name="batch")
    def eval_step(state, batch,):
        logits = state.apply_fn(**batch, params=state.params, train=False)[0]
        logits = state.head_separator(logits)
        return state.preds_fn(logits)

    return eval_step

def get_train_step(optim, config):
    if config["rnn"]["use_rnn"]:
        return get_rnn_train_step(optim, config["n_heads"]["out_sizes"])
    elif config["transformer"]["use_transformer"]:
        return get_trfrmr_train_step(config["n_heads"]["out_sizes"])
    
def get_eval_step(config):
    if config["rnn"]["use_rnn"]:
        return get_rnn_eval_step()
    elif config["transformer"]["use_transformer"]:
        return get_trfrmr_eval_step(config["n_heads"]["out_sizes"])
    

