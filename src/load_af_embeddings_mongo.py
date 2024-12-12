import os
import pandas as pd
import logging
import time
import argparse
from pymongo import MongoClient, ASCENDING

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(threadName)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

AF_EMBEDDING_FOLDER = "/data/struct_embeddings/embeddings-200M"
dim = 1280
MONGO_URI = os.getenv("MONGO_URI")


def load_batch(coll, df_batch):
    documents = []
    for index, row in df_batch.iterrows():
        np_vec = row['embedding']
        vec = []
        for i in range(len(np_vec)):
            vec.append(float(np_vec[i]))
        documents.append(
            {
                "rcsb_id": row['id'],
                "struct_vector": vec
            }
        )
    coll.insert_many(documents)


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


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", required=True, type=str)
    parser.add_argument('--coll_name', required=True, type=str)
    parser.add_argument('--num_vecs_to_load', required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int)

    args = parser.parse_args()
    db_name = args.db_name
    coll_name = args.coll_name
    num_vecs_to_load = args.num_vecs_to_load
    batch_size = args.batch_size
    return db_name, coll_name, num_vecs_to_load, batch_size


def main():
    db_name, coll_name, num_vecs_to_load, batch_size = handle_args()

    client = MongoClient(MONGO_URI)
    db = client[db_name]
    coll = db[coll_name]

    coll.create_index([("rcsb_id", ASCENDING)])

    load_all(coll, AF_EMBEDDING_FOLDER, num_vecs_to_load, batch_size=batch_size)


if __name__ == '__main__':
    main()
