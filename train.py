import sys

import jax, wandb
import jax.numpy as jnp
import equinox as eqx
import jax.tree_util as jtu
from sklearn.metrics import classification_report
from flax.training.common_utils import shard
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate

from chat_cmds.train_utils import get_optimizer, load_model, read_yaml
from chat_cmds.data_processors.load_data import get_data
from chat_cmds.data_processors.batch_loader import get_train_eval_loaders
from train_eval_steps import get_eval_step, get_train_step


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
                    wandb.log({k: eval_metrics_dict})
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


def train_trfrmr_main(config, train_dataloader, eval_dataloader, cat_to_int_map):

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
        for batch in train_dataloader:
            batch = shard(batch)
            train_state, train_metric, dropout_rngs = train_step(
                train_state, batch, dropout_rngs
            )
            train_metrics["losses"].append(train_metric["loss"])
            num_steps += 1
            if num_steps % config["logging"]["eval_steps"] == 0:
                labels = {k: [] for k, v in config["n_heads"]["out_sizes"].items()}

                preds = {k: [] for k, v in config["n_heads"]["out_sizes"].items()}

                for batch in eval_dataloader:
                    batch_labels = batch.pop("labels")
                    batch_predictions = eval_step(train_state, batch)
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
    wandb.save(config["logging"]["save_file"])


def main():
    config = read_yaml(sys.argv[1])

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
