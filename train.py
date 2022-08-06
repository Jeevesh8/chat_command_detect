import sys

import jax
from flax.training.common_utils import shard
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate

from chat_cmds.train_utils import read_yaml
from chat_cmds.data_processors.load_data import get_data
from chat_cmds.data_processors.batch_loader import get_train_eval_loaders


def main():
    config = read_yaml(sys.argv[1])

    df = get_data(**config["data"])
    train_dataloader, eval_dataloader = get_train_eval_loaders(config, df)

    config["data"]["train_length"] = len(df[df["split"] == "train_data"])
    config["data"]["valid_length"] = len(df[df["split"] == "valid_data"])


if __name__ == "__main__":
    main()
