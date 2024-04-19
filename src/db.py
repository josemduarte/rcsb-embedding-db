import os
import chromadb
import pandas as pd


def init_db_collection(embedding_path):
    chroma_client = chromadb.Client()
    db_collection = chroma_client.create_collection(
        name="chain_collection",
        metadata={"hnsw:space": "cosine"}
    )

    print("DB loading starts!")
    for r in os.listdir(embedding_path):
        instance_id = ".".join(r.split(".")[0:2])
        v = list(pd.read_csv(f"{embedding_path}/{r}").iloc[:, 0].values)
        db_collection.add(
            embeddings=[v],
            ids=[instance_id]
        )
        break

    print("DB load done!")
    return db_collection
