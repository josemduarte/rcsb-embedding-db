import os

import pandas as pd

from embedding_db import EmbeddingDB
import multiprocessing as mp

af_embedding_folder = "/mnt/vdc1/computed-models/embeddings"
collection_name = 'af_embeddings'
dim = 1280


def main():
    embedding_db = EmbeddingDB(
        collection_name,
        dim
    )
    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        for _ in pool.imap_unordered(
                lambda file: pd.read_pickle(file),
                [f'{af_embedding_folder}/{df}' for df in os.listdir(af_embedding_folder)]
        ):
            pass
    embedding_db.index_collection()


if __name__ == '__main__':
    main()
