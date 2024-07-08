import os
import chromadb
import pandas as pd


def init_db_collection(chain_path, assembly_path):
    chroma_client = chromadb.Client()
    chain_collection = chroma_client.create_collection(
        name="chain_collection",
        metadata={"hnsw:space": "cosine"}
    )
    assembly_collection = chroma_client.create_collection(
        name="assembly_collection",
        metadata={"hnsw:space": "cosine"}
    )
    load_path_to_collection(chain_path, chain_collection)
    load_path_to_collection(assembly_path, assembly_collection)

    return chain_collection, assembly_collection


def display_progress(current, total):
    percent = (current / total) * 100
    print(f"\rProgress: {percent:.2f}% ({current}/{total} files)", end='')
    if current == total:
        print()


def load_path_to_collection(embedding_path, db_collection):
    print(f"DB loading {db_collection.name} path {embedding_path} starts!")
    files = os.listdir(embedding_path)
    for idx, r in enumerate(files):
        instance_id = file_name(r)
        v = list(pd.read_csv(f"{embedding_path}/{r}").iloc[:, 0].values)
        db_collection.add(
            embeddings=[v],
            ids=[instance_id]
        )
        display_progress(idx+1, len(files))
    print(f"DB load {db_collection.name} done!")
    return db_collection


def file_name(file):
    return os.path.splitext(file)[0]