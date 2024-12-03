import os
import pandas as pd
from elasticsearch import Elasticsearch, helpers
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

af_embedding_folder = "/data/struct_embeddings/embeddings-200M"
dim = 1280
INDEX_NAME = 'chain_struct_embeddings'
ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv('ES_USER')
ES_PWD = os.getenv('ES_PWD')
BATCH_SIZE = 100
MAX_VECS_TO_INDEX = 1000


def create_index(es):
    # Delete the index if it already exists (optional)
    if es.indices.exists(index=INDEX_NAME):
        logger.info(f"Index {INDEX_NAME} already exists. Dropping index {INDEX_NAME}")
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
    logger.info(f"Creating index {INDEX_NAME}")
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


def get_batches_from_df(df, batch_size):
    """Thanks ChatGPT"""
    for start_row in range(0, df.shape[0], batch_size):
        yield df.iloc[start_row:start_row + batch_size]


def main():
    # Connect to Elasticsearch
    es = Elasticsearch([ES_URL], basic_auth=(ES_USER, ES_PWD), verify_certs=False)

    create_index(es)

    batch_index = 0
    over_max = False
    for df in os.listdir(af_embedding_folder):
        file = f'{af_embedding_folder}/{df}'
        logger.info("Starting processing dataframe file %s" % file)
        data = pd.read_pickle(file)
        for batch in get_batches_from_df(data, BATCH_SIZE):
            logger.info("Indexing batch %d from file %s" % (batch_index, file))
            index_batch(es, batch)
            batch_index += 1
            if batch_index * BATCH_SIZE > MAX_VECS_TO_INDEX:
                logger.info("Stopping indexing because we are over MAX_VECS_TO_INDEX=%d" % MAX_VECS_TO_INDEX)
                over_max = True
                break
        if over_max:
            break

if __name__ == '__main__':
    main()
