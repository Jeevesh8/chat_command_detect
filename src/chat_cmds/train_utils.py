import os
import optax
import yaml

from typing import Any, Dict, Callable, Tuple, Union

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jrandom
from flax.training import train_state
from flax import traverse_util, struct
from flax.training.common_utils import onehot
from transformers import FlaxAutoModelForSequenceClassification

from chat_cmds.models.utils import NFoldHead
from chat_cmds.models.rnns import RNN, BiRNN, pick_index


def check_config(config):
    assert (
        config["rnn"]["use_rnn"] != config["transformer"]["use_transformer"]
    ), """
        Can only use one of rnn or transformer at a time!"""
    assert config["rnn"]["cell"] in [
        "lstm",
    ]

    if config["optimizer"]["wd"] != 0:
        assert config["optimizer"]["type"] == "adamw"

    if config["rnn"]["use_rnn"]:
        config["n_heads"]["input_size"] = config["rnn"]["hidden_size"]
    else:
        config["n_heads"]["input_size"] = config["transformer"]["hidden_size"]
    config["n_heads"]["out_sizes"] = {
        k: v for k, v in config["n_heads"]["out_sizes"].items() if v is not None
    }
    return config


def read_yaml(filename: os.PathLike) -> yaml.YAMLObject:
    with open(filename, "r") as f:
        attrs = yaml.safe_load(f)

    attrs["training"]["batch_size"] = (
        attrs["training"]["per_device_batch_size"] * jax.device_count()
    )

    return check_config(attrs)


def load_transformer(config: Dict[str, Any], key: jrandom.PRNGKey) -> eqx.Module:
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        config["transformer"]["pt_model"],
        num_classes=sum(config["n_heads"]["out_sizes"].values()),
        seed=key,
    )
    return model


def load_rnn(rnn_config: Dict[str, Any], key: jrandom.PRNGKey) -> eqx.Module:
    keys = jrandom.split(key, rnn_config["num_layers"])

    kwargs = dict(
        in_size=rnn_config["in_size"],
        hidden_size=rnn_config["hidden_size"],
        cell_fn=eqx.nn.LSTMCell if rnn_config["cell"] == "lstm" else None,
    )

    rnn_class = BiRNN if rnn_config["bidirectional"] else RNN

    layers = [rnn_class(**kwargs, key=keys[0])]

    kwargs.update(in_size=rnn_config["hidden_size"])

    for key in keys[1:]:
        layers.append(rnn_class(**kwargs, key=key))

    return eqx.nn.Sequential(layers)


def get_head(head_config: Dict[str, Any], key: jnp.ndarray) -> eqx.Module:
    return NFoldHead(
        input_size=head_config["input_size"],
        out_sizes=[val for val in head_config["out_sizes"].values() if val is not None],
        use_bias=head_config["use_bias"],
        names=[
            k
            for k in head_config["out_sizes"].keys()
            if head_config["out_sizes"][k] is not None
        ],
        key=key,
    )


def load_model(config: Dict[str, Any], key: jrandom.PRNGKey) -> eqx.Module:
    base_key, head_key = jrandom.split(key, 2)

    if config["rnn"]["use_rnn"]:
        base_model = load_rnn(config["rnn"], base_key)
    elif config["transformer"]["use_transformer"]:
        base_model = load_transformer(config["transformer"], base_key)
        return base_model
    
    classifier_head = get_head(config["n_heads"], head_key)

    return eqx.nn.Sequential([base_model, pick_index(-1), classifier_head])


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    warmup: float,
    decay_to: float,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    num_warmup_steps = int(num_train_steps * warmup)
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=decay_to,
        transition_steps=num_train_steps - num_warmup_steps,
    )

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


def get_lr_schedule(config: Dict[str, Any]):
    lr_curve = create_learning_rate_fn(
        config["data"]["train_length"],
        config["training"]["batch_size"],
        config["optimizer"]["epochs"],
        config["optimizer"]["lr"],
        config["optimizer"]["warmup"],
        config["optimizer"]["lr_decay"],
    )
    print(
        "Stepwise learning rates:",
        [lr_curve(i) for i in range(config["data"]["train_length"])],
    )
    return lr_curve


def get_rnn_optimizer(config: Dict[str, Any]) -> optax.GradientTransformation:
    if config["optimizer"]["type"] == "adam":
        return optax.adam(
            learning_rate=get_lr_schedule(config),
        )
    elif config["optimizer"]["type"] == "adamw":
        return optax.adamw(
            learning_rate=get_lr_schedule(config),
            weight_decay=config["optimizer"]["wd"],
        )
    else:
        raise NotImplementedError(config["optimizer"]["type"])


class TrainState(train_state.TrainState):
    """Train state with an Optax optimizer.
    The two functions below differ depending on whether the task is classification
    or regression.
    Args:
        preds_fn: Applied to last layer to obtain the predictions.
        loss_fn: Function to compute the loss.
    """

    head_separator: Callable = struct.field(pytree_node=False)
    preds_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)


def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    # find out all LayerNorm parameters
    layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
    layer_norm_named_params = set(
        [
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        ]
    )
    flat_mask = {
        path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params)
        for path in flat_params
    }
    return traverse_util.unflatten_dict(flat_mask)


def get_trfrmr_optimizer(config: Dict[str, Any]) -> optax.GradientTransformation:
    if config["optimizer"]["type"] == "adamw":
        return optax.adamw(
            learning_rate=get_lr_schedule(config),
            b1=0.9,
            b2=0.999,
            eps=1e-6,
            weight_decay=config["optimizer"]["wd"],
            mask=decay_mask_fn,
        )
    elif config["optimizer"]["type"] == "adam":
        return optax.adam(
            learning_rate=get_lr_schedule(config),
        )
    else:
        raise NotImplementedError(config["optimizer"]["type"])


def cross_entropy_loss(logits, labels, num_labels):
    xentropy = optax.softmax_cross_entropy(
        logits, onehot(labels, num_classes=num_labels)
    )
    return jnp.mean(xentropy)


def get_head_separator(config):
    def head_separator(all_logits):
        attr_wise_logits = {}
        for k, v in config["n_heads"]["out_sizes"].items():
            attr_wise_logits[k] = all_logits[..., :v]
            all_logits = all_logits[..., v:]
        return attr_wise_logits

    return head_separator


def get_optimizer(
    config: Dict[str, Any],
    model,
) -> Tuple[optax.GradientTransformation, Union[optax.OptState, TrainState]]:
    if config["rnn"]["use_rnn"]:
        optimizer = get_rnn_optimizer(config)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        return optimizer, opt_state
    else:
        optimizer = get_trfrmr_optimizer(config)
        training_state = TrainState.create(
            apply_fn=model.__call__,
            params=model.params,
            tx=optimizer,
            preds_fn=lambda logits: jtu.tree_map(
                lambda z: jnp.argmax(z, axis=-1), logits
            ),
            head_separator=get_head_separator(config),
            loss_fn=cross_entropy_loss,
        )

        return optimizer, training_state
