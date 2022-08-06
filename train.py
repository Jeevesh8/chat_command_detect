import sys

import jax
import jax.numpy as jnp
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

    df = get_data(**config["data"])
    train_dataloader, eval_dataloader = get_train_eval_loaders(config, df)

    config["data"]["train_length"] = len(df[df["split"] == "train_data"])
    config["data"]["valid_length"] = len(df[df["split"] == "valid_data"])

    key = jax.random.PRNGKey(config["training"]["seed"])
    model = load_model(config, key)

    optimizer, opt_state = get_optimizer(config, model)

    train_step = get_train_step(optimizer, config)
    eval_step = get_eval_step(config)

    opt_state, model = replicate(opt_state), replicate(model)

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
                    predictions = eval_step(model, shard(batch[0]))
                    preds = jtu.tree_map(
                        lambda x, y: x + y,
                        jtu.tree_map(lambda x: x.flatten().tolist(), predictions),
                        preds,
                    )
                    labels += batch[1].tolist()

                for k in preds:
                    print(
                        f"Evaluation metrics at step {num_steps} for task of predicting {k}:",
                        classification_report(
                            labels[k],
                            preds[k],
                            target_names=sorted(df[k].unique().tolist()),
                        ),
                    )
        print(f"Completed epoch {epoch}")


if __name__ == "__main__":
    main()
