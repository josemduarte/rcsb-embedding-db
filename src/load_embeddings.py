import os

import pandas as pd

from embedding_db import EmbeddingDB
import concurrent.futures


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

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(insert_file, f'{af_embedding_folder}/{df}') for df in os.listdir(af_embedding_folder)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    embedding_db.flush()
    embedding_db.index_collection()


if __name__ == '__main__':
    main()
