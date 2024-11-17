import os
import numpy as np
import pandas as pd
from tqdm import tqdm

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
        for df in load_embeddings_in_batches(embedding_folder, 5*self.BATCH_SIZE):
            if not {self.ID_FIELD, self.EMBEDDING_FIELD}.issubset(df.columns):
                raise ValueError(f"DataFrame must contain '{self.ID_FIELD}' and '{self.EMBEDDING_FIELD}' columns.")

            batch_size = self.BATCH_SIZE
            total_rows = len(df)
            num_batches = (total_rows + batch_size - 1) // batch_size  # Calculate the number of batches needed

            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]

                ids = batch_df[self.ID_FIELD].tolist()
                embeddings = batch_df[self.EMBEDDING_FIELD].tolist()

                entities = [
                    ids,  # List of identifiers
                    embeddings  # List of embeddings
                ]
                self.collection.insert(entities)

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
            "params": {}  # Additional parameters can be specified if needed
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


def load_embeddings_in_batches(folder_path, batch_size=10000):
    """
    Load embeddings from files in a folder in batches, yielding a DataFrame for each batch.

    Parameters:
        folder_path (str): Path to the folder containing embedding files.
        batch_size (int): Number of files to process in each batch.

    Yields:
        pd.DataFrame: DataFrame with 'Id' and 'embedding' columns for each batch.
    """
    # List all files in the folder
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    total_files = len(filenames)

    # Sort filenames for consistency (optional)
    filenames.sort()

    # Initialize the progress bar
    with tqdm(total=total_files, desc="Loading embeddings", unit="file") as pbar:
        for i in range(0, total_files, batch_size):
            batch_files = filenames[i:i+batch_size]
            data = []
            for filename in batch_files:
                file_path = os.path.join(folder_path, filename)
                file_id, _ = os.path.splitext(filename)  # Remove extension to get Id
                try:
                    # Load the embedding from the file
                    embedding = np.loadtxt(file_path)
                    # Ensure the embedding is a list
                    embedding = embedding.tolist()
                    data.append({
                      f'{EmbeddingLoader.ID_FIELD}': file_id,
                      f'{EmbeddingLoader.EMBEDDING_FIELD}': embedding
                    })
                except Exception as e:
                    print(f"Error loading file {filename}: {e}")
                    # You might want to handle the exception or skip the file
                finally:
                    # Update the progress bar for each file processed
                    pbar.update(1)
            # Yield DataFrame for the current batch
            df_batch = pd.DataFrame(data)
            yield df_batch

