import os
import optax
import yaml

from typing import Any, Dict

import jax
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx

from chat_cmds.models.rnns import RNN, BiRNN
from chat_cmds.models.utils import NFoldHead


def check_config(config):
    assert (
        config["rnn"]["use_rnn"] != config["transformer"]["use_transformer"]
    ), """
        Can only use one of rnn or transformer at a time!"""
    assert config["rnn"]["cell"] in ["lstm", "gru"]

    if config["rnn"]["use_rnn"]:
        config["n_heads"]["input_size"] = config["rnn"]["hidden_size"]
    else:
        config["n_heads"]["input_size"] = config["transformer"]["hidden_size"]

    return config


def read_yaml(filename: os.PathLike) -> yaml.YAMLObject:
    with open(filename, "r") as f:
        attrs = yaml.safe_load(f)

    attrs["training"]["batch_size"] = (
        attrs["training"]["per_device_batch_size"] * jax.device_count()
    )

    return check_config(attrs)


def load_transformer(trfrmr_config: Dict[str, Any]) -> eqx.Module:
    raise NotImplementedError()


def load_rnn(rnn_config: Dict[str, Any], key: jrandom.PRNGKey) -> eqx.Module:
    keys = jrandom.split(key, rnn_config["num_layers"])

    kwargs = dict(
        in_size=rnn_config["in_size"],
        hidden_size=rnn_config["hidden_size"],
        cell_fn=eqx.nn.GRUCell if rnn_config["cell"] == "gru" else eqx.nn.LSTMCell,
    )

    rnn_class = BiRNN if rnn_config["bidirectional"] else RNN

    layers = [rnn_class(**kwargs, key=keys[0])]

    kwargs = kwargs.update(in_size=rnn_config["hidden_size"])

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

    classifier_head = get_head(config["n_heads"], head_key)

    return eqx.nn.Sequential([base_model, classifier_head])


def get_lr_schedule(optim_config):
    pass


def get_optimizer(
    config: Dict[str, Any],
) -> optax.GradientTransformation:
    if config["optimizer"]["type"] == "adam":
        return optax.adam(
            learning_rate=config["optimizer"]["lr"],
        )
