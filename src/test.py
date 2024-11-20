import os

import numpy as np

import diskannpy as dap

from src.embedding_loader import load_embeddings_in_batches, EmbeddingLoader

num_vectors = len(os.listdir("/mnt/vdb1/embedding-34466681"))    # Total number of vectors
vector_dim = 1280         # Dimension of each vector

# Create a new memmap file with write+ mode
memmap_array = np.memmap('vectors.memmap', dtype='float32', mode='w+', shape=(num_vectors, vector_dim))
memmap_idx = 0
batch_size = 5000


for df in load_embeddings_in_batches("/mnt/vdb1/embedding-34466681", False, 100000):
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size  # Calculate the number of batches needed

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]

        embeddings = np.array([np.array(xi, dtype=np.float32) for xi in batch_df[EmbeddingLoader.EMBEDDING_FIELD]], dtype=np.float32)

        # Compute the L2 norm of each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        beg_idx = memmap_idx * batch_size
        end_idx = beg_idx + len(embeddings)
        memmap_array[beg_idx:end_idx, :] = embeddings

print(">>>>>>", memmap_array.shape, "<<<<<<<<<<<<")
dap.build_disk_index(
    data=memmap_array,
    distance_metric="mips", # can also be cosine, especially if you don't normalize your vectors like above
    index_directory="./my_index",
    complexity=128,  # the larger this is, the more candidate points we consider when ranking
    graph_degree=64,  # the beauty of a vamana index is it's ability to shard and be able to transfer long distances across the grpah without navigating the whole thing. the larger this value is, the higher quality your results, but the longer it will take to build
    search_memory_maximum=16.0, # a floating point number to represent how much memory in GB we want to optimize for @ query time
    build_memory_maximum=100.0, # a floating point number to represent how much memory in GB we are allocating for the index building process
    num_threads=0,  # 0 means use all available threads - but if you are in a shared environment you may need to restrict how greedy you are
    vector_dtype=np.float32,  # we specified this in the Commonalities section above
    index_prefix="ann",  # ann is the default anyway. all files generated will have the prefix `ann_`, in the form of `f"{index_prefix}_"`
    pq_disk_bytes=0  # using product quantization of your vectors can still achieve excellent recall characteristics at a fraction of the latency, but we'll do it without PQ for now
)

af_embedding_folder = "/mnt/vdc1/computed-models/embeddings"
futures = [f'{af_embedding_folder}/{df}' for df in os.listdir(af_embedding_folder)]


def insert_df(df):
    if not {EmbeddingLoader.ID_FIELD, EmbeddingLoader.EMBEDDING_FIELD}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain '{EmbeddingLoader.ID_FIELD}' and '{EmbeddingLoader.EMBEDDING_FIELD}' columns.")

    batch_size = EmbeddingLoader.BATCH_SIZE
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size  # Calculate the number of batches needed

    print(f"Total rows: {total_rows}, Batch size: {batch_size}, Number of batches: {num_batches}")

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]

        ids = batch_df[EmbeddingLoader.ID_FIELD].tolist()
        embeddings = [embedding.tolist() for embedding in batch_df[EmbeddingLoader.EMBEDDING_FIELD]]

        entities = [
            ids,         # List of identifiers
            embeddings   # List of embeddings
        ]

import struct
import pandas as pd
from src.embedding_loader import EmbeddingLoader


def df_to_bin(df, filename):
    data = np.array([
        np.array(xi, dtype=np.float32) / np.linalg.norm(np.array(xi, dtype=np.float32))
        for xi in df[EmbeddingLoader.EMBEDDING_FIELD]
    ], dtype=np.float32)
    with open(filename, 'ab') as f:
        f.write(data.tobytes())


df_files = os.listdir("/mnt/vdc1/computed-models/embeddings/")
af_embeddings_file = "/mnt/vdd1/af_embeddings.bin"
n_embeddings = 0

with open(af_embeddings_file, 'wb') as f:
    f.write(struct.pack('<i', n_embeddings))
    f.write(struct.pack('<i', 1280))

with tqdm(total=len(df_files), desc="Loading embeddings", unit="file") as pbar:
    for df_file in df_files:
        df = pd.read_pickle(f"/mnt/vdc1/computed-models/embeddings/{df_file}")
        n_embeddings += len(df)
        df_to_bin(
            df,
            af_embeddings_file
        )
        pbar.update(1)

with open(af_embeddings_file, 'r+b') as f:
    f.seek(0)
    f.write(struct.pack('<i', n_embeddings))
    f.write(struct.pack('<i', 1280))

print("Number of embeddings: ", n_embeddings)

