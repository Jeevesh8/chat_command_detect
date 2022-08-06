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
    
    @eqx.filter_pmap(axis_name="device_axis", donate_argnums=(0,1,))
    def train_step(model, opt_state, x, y):
        
        @eqx.filter_value_and_grad
        def loss_fn(model, x, y,):
            logits = eqx.filter_vmap(model, (None, 0))(x)
            loss = jtu.tree_reduce(lambda x,y: x+y, jtu.tree_map(cross_entropy_loss, logits, y, num_labels))
            return loss
        
        loss, grads = loss_fn(model, x, y)
        grads = jax.lax.pmean(grads, axis_name="device_axis")
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        loss = jax.lax.pmean(loss, axis_name="device_axis")
        return loss, model, opt_state
    
    return train_step

def get_rnn_eval_step():

    @eqx.filter_pmap(axis_name="device_axis")
    def eval_step(model, x,):
        logits = eqx.filter_vmap(model, (None, 0))(x)
        return jnp.argmax(logits, axis=-1)
    
    return eval_step

