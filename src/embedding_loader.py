import os
import numpy as np
import pandas as pd

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections, utility
)


class EmbeddingLoader:

    HOST = 'localhost'
    PORT = '19530'
    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    BATCH_SIZE = 2000

    def __init__(
            self,
            collection_name,
            dim
    ):
        self.collection = None
        self.connect()
        self.create_embedding_collection(
            collection_name,
            dim
        )

    def connect(self):
        connections.connect(
            host=self.HOST,
            port=self.PORT
        )

    def create_embedding_collection(
            self,
            collection_name,
            dim
    ):
        id_field = FieldSchema(
            name=self.ID_FIELD,
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=100  # Adjust max_length based on your identifier length
        )

        embedding_field = FieldSchema(
            name=self.EMBEDDING_FIELD,
            dtype=DataType.FLOAT_VECTOR,
            dim=dim
        )

        collection_schema = CollectionSchema(
            fields=[id_field, embedding_field],
            description="Collection storing embeddings with cosine distance."
        )

        if collection_name in list(list_collections()):
            utility.drop_collection(collection_name)

        self.collection = Collection(name=collection_name, schema=collection_schema)

    def insert_folder(self, embedding_folder):
        print(f"Loading embeddings folder {embedding_folder}")
        df = load_embeddings(embedding_folder)
        if not {self.ID_FIELD, self.EMBEDDING_FIELD}.issubset(df.columns):
            raise ValueError(f"DataFrame must contain '{self.ID_FIELD}' and '{self.EMBEDDING_FIELD}' columns.")
        print(f"Loaded {len(df)} embeddings")

        batch_size = self.BATCH_SIZE
        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size  # Calculate the number of batches needed

        print(f"Total rows: {total_rows}, Batch size: {batch_size}, Number of batches: {num_batches}")

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]

            ids = batch_df[self.ID_FIELD].tolist()
            embeddings = [embedding.tolist() for embedding in batch_df[self.EMBEDDING_FIELD]]

            entities = [
                ids,         # List of identifiers
                embeddings   # List of embeddings
            ]

            print(f"Inserting batch {batch_num + 1}/{num_batches}, rows {start_idx} to {end_idx}")
            insert_result = self.collection.insert(entities)
            print(f"Inserted {len(insert_result.primary_keys)} entities into the collection.")

    def flush(self):
        self.collection.flush()

    def index_collection(self):
        # Create an index on the embedding field with cosine distance
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",  # You can choose other index types as needed
            "params": {"M": 16, "efConstruction": 128}
        }
        self.collection.create_index(
            field_name=self.EMBEDDING_FIELD,
            index_params=index_params
        )

        # Define the index parameters for an inverted index
        index_params = {
            "index_type": "AUTOINDEX",  # Use 'AUTOINDEX' for scalar fields
            "params": {}               # Additional parameters can be specified if needed
        }

        # Create the index on the 'id' field
        self.collection.create_index(
            field_name=self.ID_FIELD,
            index_params=index_params
        )
        print("Index created with cosine distance metric.")

        # Optionally, load the collection to memory for faster queries
        self.collection.load()
        print("Collection loaded to memory.")


def load_embeddings(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_id, _ = os.path.splitext(filename)  # Remove extension to get Id
            try:
                embedding = np.loadtxt(file_path)
                embedding = embedding.tolist()
                data.append({
                    f'{EmbeddingLoader.ID_FIELD}': file_id,
                    f'{EmbeddingLoader.EMBEDDING_FIELD}': embedding
                })
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                continue  # Skip files that cause errors

    df = pd.DataFrame(data)
    return df
