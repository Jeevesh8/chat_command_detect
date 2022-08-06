import shlex, os
import subprocess

from typing import Optional, List

import jax
import spacy

nlp = spacy.load("en_core_web_sm")

import numpy as np
import pandas as pd
import jax.numpy as jnp

from fasttext import load_model


class fasttext_embed:
    def __init__(self, fasttext_file: Optional[str] = None):
        self.fasttext_file = fasttext_file
        self.download_word_embeddings()
        self.subword_model = load_model(self.fasttext_file)

    def download_word_embeddings(self):
        if (
            "crawl-300d-2M-subword.bin" not in os.listdir("./")
            and self.fasttext_file is None
        ):
            print(f"Can't find model file in current directory. Downloading..")
            subprocess.run(
                shlex.split(
                    "wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"
                ),
                check=True,
            )
            subprocess.run(shlex.split("unzip crawl-300d-2M-subword.zip"), check=True)
            print("Downloaded file!!")
            self.fasttext_file = "./crawl-300d-2M-subword.bin"
        elif self.fasttext_file is None:
            self.fasttext_file = "./crawl-300d-2M-subword.bin"

    def __call__(self, tokens: List[str]) -> List[np.ndarray]:
        return [self.subword_model.get_word_vector(token) for token in tokens]


def spacy_tokenize(sample_text: str):
    return [token.text for token in nlp(sample_text)]


def rnn_data_loader(
    df: pd.DataFrame,
    batch_size: int,
    seq_len: int,
    cols: List[str] = ["action", "object", "location"],
):
    embedder = fasttext_embed()

    batched_inputs = []

    batched_labels = {col: [] for col in cols}

    for i, (index, row) in enumerate(df.iterrows()):

        if i > 0 and i % batch_size == 0:

            for k in batched_labels:
                batched_labels[k] = jnp.array(batched_labels[k])

            yield jnp.array(batched_inputs), batched_labels

            batched_inputs = []

            batched_labels = {col: [] for col in cols}

        token_wise_embedding = embedder(spacy_tokenize(row["transcription"]))
        token_wise_embedding += [np.zeros_like(token_wise_embedding[-1])] * (
            seq_len - len(token_wise_embedding)
        )
        batched_inputs.append(token_wise_embedding)

        for k in batched_labels:
            batched_labels[k].append(row[k])

    if len(batched_inputs) > 0:
        for k in batched_labels:
            batched_labels[k] = jnp.array(batched_labels[k])

        yield jnp.array(batched_inputs), batched_labels

def trfrmr_data_loader():
    raise NotImplementedError()

def get_loader(config, *args, **kwargs):
    if config["rnn"]["use_rnn"]:
        return rnn_data_loader(*args, **kwargs)
    else:
        return trfrmr_data_loader(*args, **kwargs)

def get_train_eval_loaders(config, df,):
    
    batch_size = config["training"]["per_device_batch_size"]*jax.device_count()
    
    loader_args = dict(batch_size=batch_size, 
                       seq_len=config["training"]["seq_len"],
                       cols=[k for k,v in config["n_heads"]["out_sizes"].items() if v is not None],)
    
    train_dataloader = get_loader(df[df["split"]=="train_data"], **loader_args)
    eval_dataloader  = get_loader(df[df["split"]=="valid_data"], **loader_args)
    
    return train_dataloader, eval_dataloader