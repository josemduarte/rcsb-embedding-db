import os

import pandas as pd

from embedding_db import EmbeddingDB
import multiprocessing as mp


af_embedding_folder = "/mnt/vdc1/computed-models/embeddings"
collection_name = 'af_embeddings'
dim = 1280
embedding_db = EmbeddingDB(
    collection_name,
    dim
)


def insert_file(file):
    embedding_db.insert_df(pd.read_pickle(file))
    return f"Loaded {file}"


def main():

    num_processes = mp.cpu_count()
    print(f"Using {num_processes} CPU cores")
    with mp.Pool(num_processes) as pool:
        for _ in pool.imap_unordered(
                insert_file,
                [f'{af_embedding_folder}/{df}' for df in os.listdir(af_embedding_folder)]
        ):
            pass

    embedding_db.flush()
    embedding_db.index_collection()


if __name__ == '__main__':
    main()
