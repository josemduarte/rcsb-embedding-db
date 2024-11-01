import os
import chromadb
import pandas as pd


def init_db_collection(chain_path, csm_path, assembly_path):
    chroma_client = chromadb.Client()
    chain_collection = chroma_client.create_collection(
        name="chain_collection",
        metadata={"hnsw:space": "cosine"}
    )
    csm_collection = chroma_client.create_collection(
        name="csm_collection",
        metadata={"hnsw:space": "cosine"}
    )
    assembly_collection = chroma_client.create_collection(
        name="assembly_collection",
        metadata={"hnsw:space": "cosine"}
    )
    load_path_to_collection(chain_path, chain_collection, csm_collection)
    load_path_to_collection(csm_path, csm_collection)
    load_path_to_collection(assembly_path, assembly_collection)

    return chain_collection, csm_collection, assembly_collection


def display_progress(current, total):
    percent = (current / total) * 100
    print(f"\rProgress: {percent:.2f}% ({current}/{total} files)", end='')
    if current == total:
        print()


def load_path_to_collection(embedding_path, *db_collection_list):
    print(f"DB loading data from path {embedding_path}")
    files = os.listdir(embedding_path)
    buffer = {
        "ids": [],
        "embeddings": []
    }
    buffer_size = 40000
    for idx, r in enumerate(files):
        instance_id = file_name(r)
        v = list(pd.read_csv(f"{embedding_path}/{r}").iloc[:, 0].values)
        buffer["ids"].append(instance_id)
        buffer["embeddings"].append(v)
        if idx % buffer_size == 0:
            for db_collection in db_collection_list:
                db_collection.add(
                    embeddings=buffer["embeddings"],
                    ids=buffer["ids"]
                )
            buffer["embeddings"] = []
            buffer["ids"] = []
        display_progress(idx+1, len(files))
    if len(buffer["embeddings"]) > 0:
        for db_collection in db_collection_list:
            db_collection.add(
                embeddings=buffer["embeddings"],
                ids=buffer["ids"]
            )
    print(f"DB load path {embedding_path} done")


def file_name(file):
    return os.path.splitext(file)[0]