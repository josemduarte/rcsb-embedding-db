import os
import pandas as pd
from elasticsearch import Elasticsearch
import logging
import numpy as np
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(threadName)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

AF_EMBEDDING_FOLDER = "/data/struct_embeddings/embeddings-200M"
ES_URL = os.getenv("ES_URL").split(";")
ES_USER = os.getenv('ES_USER')
ES_PWD = os.getenv('ES_PWD')


def get_queries(af_embedding_folder, num_queries):
    """
    Get a random set of query vectors of size num_queries from the first file in the directory listing of AF_EMBEDDING_FOLDER
    Note that indexing was done in the same order, thus all entries of first files should be in the index
    :param af_embedding_folder:
    :param num_queries:
    :return:
    """
    queries = {}

    files = os.listdir(af_embedding_folder)

    for df in files:
        file = f'{af_embedding_folder}/{df}'
        logger.info("Starting processing dataframe file %s" % file)
        data = pd.read_pickle(file)
        # we shuffle to avoid any effect from caching
        shuffled_data = data.sample(frac=1) # shuffle all rows
        i = 0
        for index, row in shuffled_data.iterrows():
            queries[row['id']] = row['embedding']
            if i >= num_queries - 1:
                break
            i += 1
        # we just need the first file.
        break
    return queries


def run_query(es, index_name, query_id, query_vector, hits_to_return, paginate_at):
    knn_query = {
        "size": paginate_at,
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
    return took, found_query

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", default="chain_struct_embeddings", required=True, type=str)
    parser.add_argument('--num_queries', required=True, type=int)
    parser.add_argument("--num_hits", required=False, type=int, default = 100)
    parser.add_argument("--paginate_at", required=False, type=int, default=10)

    args = parser.parse_args()
    index_name = args.index_name
    num_queries = args.num_queries
    num_hits = args.num_hits
    paginate_at = args.paginate_at
    return index_name, num_queries, num_hits, paginate_at

def main():
    index_name, num_queries, num_hits, paginate_at = handle_args()
    es = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PWD), verify_certs=False)

    queries = get_queries(AF_EMBEDDING_FOLDER, num_queries)

    times = np.zeros(num_queries)

    queries_not_found = []

    i = 0
    for query_id, query in queries.items():
        times[i], found_query = run_query(es, index_name, query_id, query, num_hits, paginate_at)
        if not found_query:
            queries_not_found.append(query_id)
        i += 1

    time_min = np.min(times)
    time_max = np.max(times)
    time_median = np.median(times)
    time_mean = np.mean(times)
    time_percentile = np.percentile(times, 95)

    if len(queries_not_found)>0:
        logger.warning("Could not find self for %d query ids: %s" % (len(queries_not_found), queries_not_found))

    logger.info("Index [%s]. Num queries [%d]. Num hits [%d]. Times (ms): min-max [%d, %d], median [%.2f], mean [%.2f], 95-percentile [%.2f]" %
                (index_name, num_queries, num_hits, time_min, time_max, time_median, time_mean, time_percentile))


if __name__ == '__main__':
    main()
