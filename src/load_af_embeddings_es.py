import os
import pandas as pd
from elasticsearch import Elasticsearch, helpers
import logging
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

AF_EMBEDDING_FOLDER = "/data/struct_embeddings/embeddings-200M"
dim = 1280
INDEX_NAME = 'chain_struct_embeddings'
ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv('ES_USER')
ES_PWD = os.getenv('ES_PWD')
BATCH_SIZE = 1000
MAX_VECS_TO_INDEX = 10000000


def create_index(es):
    # Delete the index if it already exists (optional)
    if es.indices.exists(index=INDEX_NAME):
        logger.info(f"Index {INDEX_NAME} already exists. Dropping index {INDEX_NAME}")
        es.indices.delete(index=INDEX_NAME)

    # Create the index with the appropriate mapping
    # Note type int8_hnsw is available since ~ 8.16 (definitely not available in 8.9)
    # In my tests with 10M vectors in a single node ES cluster, int8_hnsw performs queries ~ 100x faster:
    #  - baseline test on 8.9.1 ES (that doesn't even have the option), queries are ~ 10-20s
    #  - int8_hnsw test on 8.16.1 ES, queries are ~ 100-700ms
    mapping = {
        "mappings": {
            "properties": {
                "rcsb_id": {
                    "type": "keyword",
                    "index": True
                },
                "struct-vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "similarity": "cosine",
                    "index": True,
                    "index_options": {
                        "type": "int8_hnsw"
                    }
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
                "_id": row['id'],
                "_source": {
                    "rcsb_id": row['id'],
                    "struct-vector": row['embedding']
                }
            }
        )

    # Bulk index the data
    helpers.bulk(es, actions)


def get_batches_from_df(df, batch_size):
    """Thanks ChatGPT"""
    for start_row in range(0, df.shape[0], batch_size):
        yield df.iloc[start_row:start_row + batch_size]


def index_all(es, af_embedding_folder):
    batch_index = 0
    num_df_files_processed = 0
    over_max = False
    start_time = time.time()
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
        logger.info("Done processing dataframe file %s" % file)
        num_df_files_processed += 1
        if over_max:
            break
    end_time = time.time()
    logger.info(f"Finished indexing {num_df_files_processed} dataframe files in {end_time - start_time} s")


def main():
    # Connect to Elasticsearch
    es = Elasticsearch([ES_URL], basic_auth=(ES_USER, ES_PWD), verify_certs=False)

    create_index(es)

    index_all(es, AF_EMBEDDING_FOLDER)


if __name__ == '__main__':
    main()
