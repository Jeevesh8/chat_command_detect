import os
import pandas as pd

from typing import List, Optional


def load_into_df(data_files: List[str]) -> pd.DataFrame:
    """Loads data from data_files into a Pandas DataFrame."""
    dfs = []
    for filename in data_files:
        df = pd.read_csv(filename)
        df["split"] = [filename.split("/")[-1][: -len(".csv")]] * len(df)
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.drop(["path"], axis=1)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def standardize_texts(df: pd.DataFrame) -> pd.Series:
    return df["transcription"].map(lambda text: text.replace("’", "'"))


def equalize_labels(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Repeats rows of df so that every value in col is
    equally likely.
    """
    max_size = df[col].value_counts().max()
    lst = [df]

    for _, group in df.groupby(col):
        if len(group) < max_size:
            num_grps_to_add = (max_size // len(group)) - 1
            lst += [group] * num_grps_to_add
            lst.append(
                group.sample(
                    max_size - len(group) * (num_grps_to_add + 1), replace=False
                )
            )

    return pd.concat(lst)


def get_data(
    data_dir: Optional[str] = None,
    data_files: Optional[List[str]] = None,
    drop_duplicates: bool = True,
    balance_data: Optional[List[str]] = None,
    shuffle: bool=True,
    cat_to_int: bool = True,
) -> pd.DataFrame:
    """Returns a preprocessed data-frame containing the loaded data.
    Args:
        data_dir:        The directory having (.*).csv files.
                         The name of .csv file will be available in the 'split'
                         column of the returned DataFrame.
        data_files:      A list of .csv files to load the data from. Not needed if
                         data_dir is specified. Ignored if data_dir is specified.
        drop_duplicates: If True, the duplicate rows are dropped. By default is True.
        balance_data:   An optional list of categorical columns, the rows of the
                         DataFrame will be replicated so that all of the possible
                         labels will have same frequency.
        shuffle:         If true, the rows of the dataset will be shuffled(deterministically)
                         Default: True.
        cat_to_int:      If true, the categorical columns of ['action', 'object', 'location']
                         are converted to integer values. The integers are assigned to the 
                         categories in alphabetic order. Default: True
    Returns:
        A DataFrame of the loaded data.
    """

    if data_dir is not None:
        data_files = [
            os.path.join(data_dir, filename)
            for filename in os.listdir(data_dir)
            if filename.endswith(".csv")
        ]
    else:
        assert data_files is not None
    df = load_into_df(data_files)
    df = standardize_texts(df)

    if drop_duplicates:
        df = drop_duplicates(df)

    if balance_data is not None:
        for col in balance_data:
            df = equalize_labels(df, col)
    
    if shuffle:
        df = df.sample(frac=1, random_state=42,).reset_index(drop=True)
    
    if cat_to_int:
        
        for col in ['action', 'object', 'location']:
            col_vals = df[col].unique().tolist()
            col_vals.sort()
            df[col] = df[col].map({v: i for i,v in enumerate(col_vals)})
    
    return df

