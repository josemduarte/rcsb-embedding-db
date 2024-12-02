import os
import pandas as pd
from elasticsearch import Elasticsearch, helpers

af_embedding_folder = "/data/struct_embeddings/embeddings-200M"
dim = 1280
INDEX_NAME = 'vector_embeddings'
ES_URL = 'http://localhost:9200'
BATCH_SIZE = 10


def create_index(es):
    # Delete the index if it already exists (optional)
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    # Create the index with the appropriate mapping
    mapping = {
        "mappings": {
            "properties": {
                "id": {
                    "type": "keyword",
                    "index": True
                },
                "struct-vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "similarity": "cosine",
                    "index": True
                }
            }
        }
    }

    es.indices.create(index=INDEX_NAME, body=mapping)


def index_batch(es, df_batch):

    # Prepare the data for bulk indexing
    actions = []
    for index, row in df_batch.iterrows():
        actions.append(
            {
                "_index": INDEX_NAME,
                "_id": index,
                "_source": {
                    "id": row['id'],
                    "vector": row['embedding']
                }
            }
        )

    # Bulk index the data
    helpers.bulk(es, actions)

    print(f"Indexed {len(df_batch)} vector embeddings into '{INDEX_NAME}' index.")


def get_batches_from_df(df, batch_size):
    """Thanks ChatGPT"""
    for start_row in range(0, df.shape[0], batch_size):
        yield df.iloc[start_row:start_row + batch_size]


def main():
    # Connect to Elasticsearch
    es = Elasticsearch([ES_URL])

    create_index(es)

    for df in os.listdir(af_embedding_folder):
        file = f'{af_embedding_folder}/{df}'
        data = pd.read_pickle(file)
        for batch in get_batches_from_df(data, BATCH_SIZE):
            index_batch(es, batch)

        # for index, row in data.iterrows():
        #     embed_length = len(row['embedding'])
        #     print("%s : %d" % (row['id'], embed_length))

if __name__ == '__main__':
    main()
