import os
import pandas as pd
from elasticsearch import Elasticsearch
import logging
import numpy as np
import argparse
from pymongo import MongoClient
from pymongo.synchronous.collection import Collection

from load_af_embeddings_mongo import __npvec_int32_to_float

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(threadName)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

AF_EMBEDDING_FOLDER = "/data/struct_embeddings/embeddings-200M"
ES_URL = os.getenv("ES_URL").split(";")
ES_USER = os.getenv('ES_USER')
ES_PWD = os.getenv('ES_PWD')
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "struc_embed"
COLL_NAME = "chain_struct_embeddings"
ES_ID_SUFFIX = ".A-POLYMER_INSTANCE"

def get_queries(coll: Collection, num_queries: int) -> dict:
    """
    Get a random set of query vectors of size num_queries from Mongo
    :param coll
    :param num_queries:
    :return:
    """
    queries = {}

    docs = coll.aggregate([
        {"$sample": {"size": num_queries}}
    ])
    for doc in docs:
        vec_from_db_int = np.array(doc["struct_vector"], dtype=int)
        embed = __npvec_int32_to_float(vec_from_db_int)
        queries[doc["rcsb_id"]] = embed
    return queries


def build_knn_query(query_vector, hits_to_return, paginate_at):
    return {
        "size": paginate_at,
        "knn": {
            "field": "struct-vector",
            "query_vector": query_vector,
            "k": hits_to_return,
            "num_candidates": hits_to_return
        },
        "_source": False
    }

def add_filter_to_knn_query(query: dict, filter_field, value):
    query["knn"]["filter"] = {
        "term": {
            filter_field: value
        }
    }

def get_embedding_from_mongo(coll, entry_id: str):
    doc = coll.find_one({"rcsb_id": entry_id})
    vec_from_db_int = np.array(doc["struct_vector"], dtype=int)
    return __npvec_int32_to_float(vec_from_db_int)

def run_query(es, index_name, query_id, query_vector, hits_to_return, paginate_at, with_filter: tuple[str,str] = None):
    knn_query = build_knn_query(query_vector, hits_to_return, paginate_at)
    if with_filter[0] is not None:
        add_filter_to_knn_query(knn_query, with_filter[0], with_filter[1])
    # timeout at 60 is because queries for 100M with k>1000 take more than 10s (the default timeout)
    response = es.search(index=index_name, body=knn_query, request_timeout=60)
    found_query = False
    took = response['took']
    i = 0
    count_above_05 = 0
    for hit in response['hits']['hits']:
        doc_id = str(hit['_id'])
        cosine_score = 2 * hit['_score'] - 1  # see elastic knn docs
        # print("%3d %18s %.3f" % (i, doc_id.removesuffix(ES_ID_SUFFIX), cosine_score))
        if doc_id == query_id:
            found_query = True
        if cosine_score > 0.5:
            count_above_05 += 1
        i += 1
    if not found_query:
        logger.warning("Did not find query for id %s" % query_id)
    return took, found_query, count_above_05

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", default="chain_struct_embeddings", required=True, type=str)
    parser.add_argument('--num_queries', required=True, type=int)
    parser.add_argument("--num_hits", required=False, type=int, default = 100)
    parser.add_argument("--paginate_at", required=False, type=int, default=10)
    parser.add_argument("--filter_field", required=False, type=str)
    parser.add_argument("--filter_value", required=False, type=str)

    args = parser.parse_args()
    index_name = args.index_name
    num_queries = args.num_queries
    num_hits = args.num_hits
    paginate_at = args.paginate_at
    filter_field = args.filter_field
    filter_value = args.filter_value
    return index_name, num_queries, num_hits, paginate_at, filter_field, filter_value

def main():
    index_name, num_queries, num_hits, paginate_at, filter_field, filter_value = handle_args()
    es = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PWD), verify_certs=False)
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    # rand_query_id = "AF_AFQ3THL5F1"
    # rand_query_id = "AF_AFA0A6N8GSJ8F1"

    queries = get_queries(coll, num_queries)
    # queries = { rand_query_id: get_embedding_from_mongo(coll, rand_query_id)}

    times = np.zeros(num_queries)

    queries_not_found = []

    counts_above05 = np.zeros(num_queries, dtype=int)

    i = 0
    for query_id, query in queries.items():
        query_id_with_suffix = query_id + ES_ID_SUFFIX
        times[i], found_query, counts_above05[i] = run_query(es, index_name, query_id_with_suffix, query, num_hits, paginate_at, (filter_field, filter_value))
        if not found_query:
            queries_not_found.append(query_id)
        i += 1

    time_min = np.min(times)
    time_max = np.max(times)
    time_median = np.median(times)
    time_mean = np.mean(times)
    time_percentile = np.percentile(times, 95)
    counts_above05_mean = np.mean(counts_above05)

    if len(queries_not_found) > 0:
        logger.warning("Could not find self for %d query ids: %s" % (len(queries_not_found), queries_not_found))

    logger.info(
        "Index [%s]. Num queries [%d]. num_candidates/k [%d]. Paginate [%d]. Times (ms): min-max [%d, %d], median [%.2f], mean [%.2f], 95-percentile [%.2f]. Mean num of non-random hits [%.1f]" %
        (index_name, num_queries, num_hits, paginate_at, time_min, time_max, time_median, time_mean, time_percentile, counts_above05_mean))


if __name__ == '__main__':
    main()
