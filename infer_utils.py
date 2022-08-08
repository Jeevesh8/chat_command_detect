import jax, os
import wandb
import pandas as pd

import equinox as eqx
from flax.jax_utils import replicate
from chat_cmds.train_utils import get_optimizer, load_model


def load_wandb_weights(config):
    api = wandb.Api()
    run = api.run(config["inference"]["run_name"])
    try:
        run.config.pop("inference")
    except KeyError:
        pass
    config.update(run.config)

    if config["rnn"]["use_rnn"]:
        wts_file = wandb.restore(
            config["logging"]["save_file"], run_path=config["inference"]["run_name"]
        )
        return wts_file.name
    elif config["transformer"]["use_transformer"]:
        for filename in ["flax_model.msgpack", "config.json"]:
            wts_file = wandb.restore(
                os.path.join(config["logging"]["save_file"], filename),
                run_path=config["inference"]["run_name"],
            )

        return os.path.dirname(wts_file.name)


def get_pretrained_model(config, wt_file):
    key = jax.random.PRNGKey(0)
    model = load_model(config, key)
    if config["rnn"]["use_rnn"]:
        model = eqx.tree_deserialise_leaves(wt_file, model)
        return model
    elif config["transformer"]["use_transformer"]:
        config["data"]["train_length"] = 1
        model = model.from_pretrained(wt_file)
        _, train_state = get_optimizer(config, model)
        train_state = replicate(train_state)
        config["data"].pop("train_length")
        return train_state


def show_table(evaluation_metrics):
    df = pd.DataFrame(
        {k: pd.json_normalize(v, sep=".") for k, v in evaluation_metrics.items()}
    )
    print(df.to_markdown())
