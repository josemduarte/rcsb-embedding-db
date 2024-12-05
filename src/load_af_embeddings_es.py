import os
import pandas as pd
from elasticsearch import Elasticsearch, helpers
import logging
import time
import argparse
import concurrent.futures

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

AF_EMBEDDING_FOLDER = "/data/struct_embeddings/embeddings-200M"
dim = 1280
ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv('ES_USER')
ES_PWD = os.getenv('ES_PWD')


def create_index(es, index_name):
    # Delete the index if it already exists (optional)
    if es.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists. Dropping index {index_name}")
        es.indices.delete(index=index_name)

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
    logger.info(f"Creating index {index_name}")
    es.indices.create(index=index_name, body=mapping)


def index_batch(es, df_batch, index_name):
    # Prepare the data for bulk indexing
    actions = []
    for index, row in df_batch.iterrows():
        actions.append(
            {
                "_index": index_name,
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


def index_all(es, af_embedding_folder, index_name, batch_size, num_vecs_to_load, num_threads=1):
    batch_index = 0
    num_df_files_processed = 0
    over_max = False
    start_time = time.time()

    # Create thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for df in os.listdir(af_embedding_folder):
            file = f'{af_embedding_folder}/{df}'
            logger.info("Starting processing dataframe file %s" % file)
            data = pd.read_pickle(file)
            for batch in get_batches_from_df(data, batch_size):
                logger.info("Submitting batch %d from file %s" % (batch_index, file))
                futures.append(executor.submit(index_batch, es, batch, index_name))
                batch_index += 1
                if batch_index * batch_size > num_vecs_to_load:
                    logger.info("Stopping indexing because we are over MAX_VECS_TO_INDEX=%d" % num_vecs_to_load)
                    over_max = True
                    break
                logger.info("Done processing dataframe file %s" % file)
                num_df_files_processed += 1
                if over_max:
                    break
        logger.info("Done submitting all batches to thread pool")
        # Wait for all threads to complete
        concurrent.futures.wait(futures)

    end_time = time.time()
    logger.info(f"Finished indexing {num_df_files_processed} dataframe files in {end_time - start_time} s")


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", default="chain_struct_embeddings", required=True, type=str)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--num_vecs_to_load', required=True, type=int)
    parser.add_argument('--num_threads', required=True, type=int)

    args = parser.parse_args()
    index_name = args.index_name
    batch_size = args.batch_size
    num_vecs_to_load = args.num_vecs_to_load
    num_threads = args.num_threads
    return index_name, batch_size, num_vecs_to_load, num_threads

def main():
    index_name, batch_size, num_vecs_to_load, num_threads = handle_args()
    es = Elasticsearch([ES_URL], basic_auth=(ES_USER, ES_PWD), verify_certs=False)
    create_index(es, index_name)
    index_all(es, AF_EMBEDDING_FOLDER, index_name, batch_size, num_vecs_to_load, num_threads=num_threads)


if __name__ == '__main__':
    main()
