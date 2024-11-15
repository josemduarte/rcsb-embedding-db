import os

import pandas as pd

from embedding_db import EmbeddingDB

af_embedding_folder = "/mnt/vdc1/computed-models/embeddings"
collection_name = 'af_embeddings'
dim = 1280


def main():
    embedding_db = EmbeddingDB(
        collection_name,
        dim
    )
    for df in os.listdir(af_embedding_folder):
        embedding_db.insert_df(
            pd.read_pickle(f'{af_embedding_folder}/{df}')
        )
    embedding_db.index_collection()


if __name__ == '__main__':
    main()
