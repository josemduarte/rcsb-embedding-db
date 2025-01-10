import os
import sys

import pandas as pd
import logging
import time
import argparse
import numpy as np
from pymongo import MongoClient, ASCENDING
from pymongo.synchronous.collection import Collection

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(threadName)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

AF_EMBEDDING_FOLDER = "/data/struct_embeddings/embeddings-200M"
dim = 1280
MONGO_URI = os.getenv("MONGO_URI")
MAX_INT32 = 2**31


def load_batch(coll, df_batch):
    documents = []
    for index, row in df_batch.iterrows():
        vec_norm = __normalize(row['embedding'])
        vec_int32 = __npvec_float_to_int32(vec_norm)
        vec = __npvec_int32_to_list(vec_int32)
        documents.append(
            {
                "rcsb_id": row['id'],
                "struct_vector": vec
            }
        )
    coll.insert_many(documents)

def __npvec_int32_to_list(npvec):
    vec = []
    for i in range(len(npvec)):
        vec.append(int(npvec[i]))
    return vec

def __normalize(npvec):
    return npvec / np.linalg.norm(npvec)

def __npvec_float_to_int32(npvec):
    return np.round(npvec * MAX_INT32).astype(np.int32)

def __npvec_int32_to_float(npvec):
    return npvec.astype(np.float32) / MAX_INT32

def get_batches_from_df(df, batch_size):
    """Thanks ChatGPT"""
    for start_row in range(0, df.shape[0], batch_size):
        yield df.iloc[start_row:start_row + batch_size]


def load_all(coll, af_embedding_folder, num_vecs_to_load, batch_size=10):
    batch_index = 0
    num_df_files_processed = 0
    over_max = False
    start_time = time.time()

    for df in os.listdir(af_embedding_folder):
        file = f'{af_embedding_folder}/{df}'
        logger.info("Starting processing dataframe file %s" % file)
        data = pd.read_pickle(file)
        for batch in get_batches_from_df(data, batch_size):
            logger.info("Submitting batch %d from file %s" % (batch_index, file))
            load_batch(coll, batch)
            batch_index += 1
            if batch_index * batch_size > num_vecs_to_load:
                logger.info("Stopping loading because we are over MAX_VECS_TO_INDEX=%d" % num_vecs_to_load)
                over_max = True
                break
        logger.info("Done processing dataframe file %s" % file)
        num_df_files_processed += 1
        if over_max:
            break

    end_time = time.time()
    logger.info(f"Finished loading {num_df_files_processed} dataframe files in {end_time - start_time} s")


def check_db_contents(coll: Collection, af_embedding_folder, batch_size=10):
    """
    Check that the vector contents of db after deserializing and decoding back to float, coincide with the vectors in
    panda files, i.e. a round trip check for the encoding done when data was written to db.
    :param coll:
    :param af_embedding_folder:
    :param batch_size:
    :return:
    """
    for df in os.listdir(af_embedding_folder):
        file = f'{af_embedding_folder}/{df}'
        logger.info("Starting processing dataframe file %s" % file)
        data = pd.read_pickle(file)
        i = 0
        for batch in get_batches_from_df(data, batch_size):
            for index, row in batch.iterrows():
                rcsb_id = row['id']
                doc = coll.find_one({"rcsb_id": rcsb_id})
                vec_from_db_int = np.array(doc["struct_vector"], dtype=int)
                vec_from_db_float = __npvec_int32_to_float(vec_from_db_int)
                vec_norm = __normalize(row['embedding'])
                if not np.allclose(vec_from_db_float, vec_norm):
                    print(rcsb_id)
                i += 1
                if i%1000 == 0:
                    print("Done %d" % i)



def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", required=True, type=str)
    parser.add_argument('--coll_name', required=True, type=str)
    parser.add_argument('--num_vecs_to_load', required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int)
    # don't load anything: simply check db contents against panda files content (with appropriate decoding)
    parser.add_argument("--check_db_contents_only", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    db_name = args.db_name
    coll_name = args.coll_name
    num_vecs_to_load = args.num_vecs_to_load
    batch_size = args.batch_size
    check_db = args.check_db_contents_only
    return db_name, coll_name, num_vecs_to_load, batch_size, check_db


def main():
    db_name, coll_name, num_vecs_to_load, batch_size, check_db = handle_args()

    client = MongoClient(MONGO_URI)
    db = client[db_name]
    coll = db[coll_name]

    if check_db:
        check_db_contents(coll, AF_EMBEDDING_FOLDER, batch_size=batch_size)
        sys.exit(0)

    coll.create_index([("rcsb_id", ASCENDING)], unique=True)

    load_all(coll, AF_EMBEDDING_FOLDER, num_vecs_to_load, batch_size=batch_size)


if __name__ == '__main__':
    main()
