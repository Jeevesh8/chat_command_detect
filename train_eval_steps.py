import jax
import optax
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.common_utils import onehot

def cross_entropy_loss(logits, labels, num_labels):
    xentropy = optax.softmax_cross_entropy(logits, onehot(labels, num_classes=num_labels))
    return jnp.mean(xentropy)

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
    
    @eqx.filter_pmap(args=(None,), axis_name="device_axis", out=None)
    def eval_step(model, x,):
        logits = jax.vmap(model,)(x)
        return jnp.argmax(logits, axis=-1)
    
    return eval_step

def get_train_step(optim, config):
    if config["rnn"]["use_rnn"]:
        return get_rnn_train_step(optim, config["n_heads"]["out_sizes"])

def get_eval_step(config):
    if config["rnn"]["use_rnn"]:
        return get_rnn_eval_step()

