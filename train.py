import sys

import jax, wandb
import jax.numpy as jnp
import equinox as eqx
import jax.tree_util as jtu
from sklearn.metrics import classification_report
from flax.training import train_state
from flax.training.common_utils import shard
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate

from chat_cmds.train_utils import get_optimizer, load_model, read_yaml
from chat_cmds.data_processors.load_data import get_data
from chat_cmds.data_processors.batch_loader import get_train_eval_loaders
from train_eval_steps import get_eval_step, get_train_step


def main():
    config = read_yaml(sys.argv[1])

    wandb.config = config
    wandb.run.name = config["logging"]["run_name"]
    
    df, cat_to_int_map = get_data(**config["data"])
    train_dataloader, eval_dataloader = get_train_eval_loaders(config, df)

    wandb.log(
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
                labels = {
                    k: []
                    for k, v in config["n_heads"]["out_sizes"].items()
                    if v is not None
                }

                preds = {
                    k: []
                    for k, v in config["n_heads"]["out_sizes"].items()
                    if v is not None
                }

                for batch in eval_dataloader:
                    batch_predictions = eval_step(model, shard(batch[0]))
                    batch_predictions = jtu.tree_map(
                        lambda x: x.flatten().tolist(), batch_predictions
                    )
                    batch_labels = jtu.tree_map(lambda arr: arr.tolist(), batch[1])
                    preds = {k: preds[k] + batch_predictions[k] for k in preds}
                    labels = {k: labels[k] + batch_labels[k] for k in labels}

                for k in preds:
                    wandb.log(
                        {"step": num_steps}.update(
                            classification_report(
                                labels[k],
                                preds[k],
                                target_names=cat_to_int_map[k].keys(),
                                output_dict=True,
                            )
                        )
                    )
        
        wandb.log({
            "running_train_loss": sum(train_metrics['losses'])/len(train_metrics['losses']),
            "epoch": epoch+1,
        })

        train_metrics["losses"] = []
        print(f"Completed epoch {epoch}")
    
    print("Completed Training! Saving model at:")
    eqx.tree_serialise_leaves(config["logging"]["save_file"], model)
    wandb.save(config["logging"]["save_file"])

if __name__ == "__main__":
    wandb.init(project="chat_cmds")
    main()
