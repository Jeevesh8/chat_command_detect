import jax, os
import wandb

import equinox as eqx
from flax.jax_utils import replicate
from chat_cmds.train_utils import get_optimizer, load_model


def load_wandb_weights(config):
    api = wandb.Api()
    run = api.run(config["inference"]["run_name"])
    run.config.pop("inference")
    config.upate(run.config)

    wts_file = wandb.restore(config["logging"]["save_file"],,
                             run_path=config["inference"]["run_name"])
    return wts_file.name

def get_pretrained_model(config, wt_file):
    key = jax.random.PRNGKey(0)
    model = load_model(config, key)
    if config["rnn"]["use_rnn"]:
        model = eqx.tree_deserialise_leaves(wt_file, model)
        return model
    elif config["transformer"]["use_transformer"]:
        _, train_state = get_optimizer(config, model)
        model = model.from_pretrained(from_pretrained=wt_file)
        train_state.params = model.params
        train_state = replicate(train_state)
        return train_state
