import sys, os
import copy

import jax, wandb
import jax.numpy as jnp
import equinox as eqx
import jax.tree_util as jtu
from sklearn.metrics import classification_report
from flax.training.common_utils import shard
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate

from chat_cmds.train_utils import get_optimizer, load_model, read_yaml
from chat_cmds.data_processors.load_data import get_data
from chat_cmds.data_processors.batch_loader import (
    get_test_loader,
    get_train_eval_loaders,
)
from train_eval_steps import get_eval_step, get_train_step
from infer_utils import load_wandb_weights, get_pretrained_model


def eval_rnn_main(
    config,
    eval_dataloader,
    eval_step,
    model,
    cat_to_int_map,
    print_labels: bool = False,
):
    labels = {k: [] for k, v in config["n_heads"]["out_sizes"].items()}

    preds = {k: [] for k, v in config["n_heads"]["out_sizes"].items()}

    for batch in eval_dataloader:
        batch_predictions = eval_step(model, shard(batch[0]))
        batch_predictions = jtu.tree_map(
            lambda x: x.flatten().tolist(), batch_predictions
        )
        batch_labels = jtu.tree_map(lambda arr: arr.tolist(), batch[1])
        preds = {k: preds[k] + batch_predictions[k] for k in preds}
        labels = {k: labels[k] + batch_labels[k] for k in labels}

    for k in preds:
        eval_metrics_dict = classification_report(
            labels[k],
            preds[k],
            target_names=cat_to_int_map[k].keys(),
            output_dict=True,
        )
        print(f"Metrics for {k}:", eval_metrics_dict)
        wandb.log({k: eval_metrics_dict})
    if print_labels:
        print(labels)


def train_rnn_main(config, train_dataloader, eval_dataloader, cat_to_int_map):

    key = jax.random.PRNGKey(config["training"]["seed"])
    model = load_model(config, key)

    optimizer, opt_state = get_optimizer(config, model)

    train_step = get_train_step(optimizer, config)
    eval_step = get_eval_step(config)

    train_metrics = {"losses": []}
    num_steps = 0

    for epoch in range(config["optimizer"]["epochs"]):
        for batch in train_dataloader:
            batch = shard(batch)
            loss, model, opt_state = train_step(model, opt_state, *batch)
            train_metrics["losses"].append(loss.item())
            num_steps += 1

            if num_steps % config["logging"]["eval_steps"] == 0:
                eval_rnn_main(config, eval_dataloader, eval_step, model, cat_to_int_map)

            wandb.log({"step": num_steps})

        wandb.log(
            {
                "running_train_loss": sum(train_metrics["losses"])
                / len(train_metrics["losses"]),
                "epoch": epoch + 1,
            }
        )

        train_metrics["losses"] = []
        print(f"Completed epoch {epoch}")

    print("Completed Training! Saving model at:", config["logging"]["save_file"])
    eqx.tree_serialise_leaves(config["logging"]["save_file"], model)
    wandb.save(config["logging"]["save_file"])


def eval_trfrmr_main(config, eval_dataloader, eval_step, train_state, cat_to_int_map):
    labels = {k: [] for k, v in config["n_heads"]["out_sizes"].items()}

    preds = {k: [] for k, v in config["n_heads"]["out_sizes"].items()}

    for batch in eval_dataloader:
        batch_labels = batch.pop("labels")
        batch = shard(batch)
        batch_predictions = eval_step(train_state, batch)
        batch_predictions = unreplicate(batch_predictions)
        batch_predictions = jtu.tree_map(
            lambda x: x.flatten().tolist(), batch_predictions
        )
        batch_labels = jtu.tree_map(lambda arr: arr.tolist(), batch_labels)
        preds = {k: preds[k] + batch_predictions[k] for k in preds}
        labels = {k: labels[k] + batch_labels[k] for k in labels}

    for k in preds:
        eval_metrics_dict = classification_report(
            labels[k],
            preds[k],
            target_names=cat_to_int_map[k].keys(),
            output_dict=True,
        )
        wandb.log({k: eval_metrics_dict})


def train_trfrmr_main(config, train_dataloader, eval_dataloader, cat_to_int_map):
    original_loaders = (train_dataloader, eval_dataloader)

    key = jax.random.PRNGKey(config["training"]["seed"])
    key, subkey = jax.random.split(key)

    model = load_model(config, subkey)

    dropout_rngs = jax.random.split(key, jax.local_device_count())

    _, train_state = get_optimizer(config, model)
    train_step = get_train_step(_, config)
    eval_step = get_eval_step(config)

    train_state = replicate(train_state)

    train_metrics = {"losses": []}
    num_steps = 0

    for epoch in range(config["optimizer"]["epochs"]):
        train_dataloader, eval_dataloader = copy.deepcopy(original_loaders)
        for batch in train_dataloader:
            batch = shard(batch)
            train_state, train_metric, dropout_rngs = train_step(
                train_state, batch, dropout_rngs
            )
            train_metrics["losses"].append(train_metric["loss"])
            num_steps += 1
            if num_steps % config["logging"]["eval_steps"] == 0:
                eval_trfrmr_main(
                    config, eval_dataloader, eval_step, train_state, cat_to_int_map
                )
            wandb.log({"step": num_steps})

        train_metrics = unreplicate(train_metrics)

        wandb.log(
            {
                "running_train_loss": sum(train_metrics["losses"]).item()
                / len(train_metrics["losses"]),
                "epoch": epoch + 1,
            }
        )

        train_metrics["losses"] = []
        print(f"Completed epoch {epoch}")

    print("Completed Training! Saving model at:", config["logging"]["save_file"])
    params = jax.device_get(unreplicate(train_state.params))
    model.save_pretrained(config["logging"]["save_file"], params=params)
    
    for filename in os.listdir(config["logging"]["save_file"]):
        wandb.save(os.path.join(config["logging"]["save_file"], filename))


def infer(config):
    wts_file = load_wandb_weights(config)
    _, cat_to_int_map = get_data(**config["data"])
    test_df = get_data(
        data_files=[config["inference"]["test_file"]], shuffle=False, cat_to_int=False
    )

    for col in ["action", "object", "location"]:
        test_df[col] = test_df[col].map(cat_to_int_map[col])
    test_loader = get_test_loader(config, test_df)
    model = get_pretrained_model(config, wts_file)

    if config["rnn"]["use_rnn"]:
        eval_step = get_eval_step(config)
        eval_rnn_main(config, test_loader, eval_step, model, cat_to_int_map)

    elif config["transformer"]["use_transformer"]:
        eval_step = get_eval_step(config)
        eval_trfrmr_main(config, test_loader, eval_step, model, cat_to_int_map)


def main():
    config = read_yaml(sys.argv[1])
    if config["inference"]["run_infer"]:
        infer(config)

    wandb.init(project="chat_cmds", config=config)

    wandb.run.name = config["logging"]["run_name"]

    df, cat_to_int_map = get_data(**config["data"])
    train_dataloader, eval_dataloader = get_train_eval_loaders(config, df)

    print(
        {
            "training_data_stats": {
                k: df[df["split"] == "train_data"][k].value_counts().tolist()
                for k in cat_to_int_map
            },
        }
    )

    train_dataloader, eval_dataloader = list(train_dataloader), list(eval_dataloader)

    config["data"]["train_length"] = len(df[df["split"] == "train_data"])
    config["data"]["valid_length"] = len(df[df["split"] == "valid_data"])
    if config["rnn"]["use_rnn"]:
        train_rnn_main(config, train_dataloader, eval_dataloader, cat_to_int_map)
    elif config["transformer"]["use_transformer"]:
        train_trfrmr_main(config, train_dataloader, eval_dataloader, cat_to_int_map)


if __name__ == "__main__":

    if len(sys.argv) > 2:
        wandb.login(key=sys.argv[2])
    else:
        wandb.login()

    main()
