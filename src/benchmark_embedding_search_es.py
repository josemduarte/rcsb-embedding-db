import os
import pandas as pd
from elasticsearch import Elasticsearch
import logging
import numpy as np
import argparse
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(threadName)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

AF_EMBEDDING_FOLDER = "/data/struct_embeddings/embeddings-200M"
dim = 1280
ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv('ES_USER')
ES_PWD = os.getenv('ES_PWD')


def get_batches_from_df(df, batch_size):
    """Thanks ChatGPT"""
    for start_row in range(0, df.shape[0], batch_size):
        yield df.iloc[start_row:start_row + batch_size]


def get_queries(af_embedding_folder, num_queries):
    queries = {}

    files = os.listdir(af_embedding_folder)
    random.shuffle(files)

    for df in files:
        file = f'{af_embedding_folder}/{df}'
        logger.info("Starting processing dataframe file %s" % file)
        data = pd.read_pickle(file)
        for batch in get_batches_from_df(data, num_queries):
            for index, row in batch.iterrows():
                queries[row['id']] = row['embedding']
            break
        break
    return queries


def run_query(es, index_name, query_id, query_vector, hits_to_return):
    knn_query = {
        "size": hits_to_return,
        "knn": {
            "field": "struct-vector",
            "query_vector": query_vector,
            "k": hits_to_return,
            "num_candidates": hits_to_return
        },
        "_source": False
    }
    response = es.search(index=index_name, body=knn_query)
    found_query = False
    took = response['took']
    for hit in response['hits']['hits']:
        doc_id = hit['_id']
        # score = hit['_score']
        if doc_id == query_id:
            found_query = True
    if not found_query:
        logger.warning("Did not find query for id %s" % query_id)
    return took

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", default="chain_struct_embeddings", required=True, type=str)
    parser.add_argument('--num_queries', required=True, type=int)

    args = parser.parse_args()
    index_name = args.index_name
    num_queries = args.num_queries
    return index_name, num_queries

def main():
    index_name, num_queries = handle_args()
    es = Elasticsearch([ES_URL], basic_auth=(ES_USER, ES_PWD), verify_certs=False)

    queries = get_queries(AF_EMBEDDING_FOLDER, num_queries)

    times = np.zeros(num_queries)

    i = 0
    for query_id, query in queries.items():
        times[i] = run_query(es, index_name, query_id, query, 100)
        i += 1

    time_min = np.min(times)
    time_max = np.max(times)
    time_median = np.median(times)
    time_percentile = np.percentile(times, 95)
    logger.info("Performed %d queries. Times (ms): min-max [%d, %d], median %.2f, 95-percentile %.2f" % (num_queries, time_min, time_max, time_median, time_percentile))


if __name__ == '__main__':
    main()
