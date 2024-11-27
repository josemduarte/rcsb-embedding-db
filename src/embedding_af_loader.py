
import numpy as np
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

    def insert_df(self, df):
        if not {self.ID_FIELD, self.EMBEDDING_FIELD}.issubset(df.columns):
            raise ValueError(f"DataFrame must contain '{self.ID_FIELD}' and '{self.EMBEDDING_FIELD}' columns.")

        batch_size = self.BATCH_SIZE
        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size  # Calculate the number of batches needed

        with tqdm(total=num_batches, desc="Loading embeddings", unit="batch") as pbar:
            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]

                ids = batch_df[self.ID_FIELD].tolist()
                embeddings = [(embedding/np.linalg.norm(embedding)).tolist() for embedding in batch_df[self.EMBEDDING_FIELD]]

                entities = [
                    ids,         # List of identifiers
                    embeddings   # List of embeddings
                ]
                self.collection.insert(entities)
                pbar.update(1)

    def flush(self):
        self.collection.flush()

    def index_collection(self):
        # Create an index on the embedding field with cosine distance
        index_params = {
            "metric_type": "IP",
            "index_type": "DISKANN",  # You can choose other index types as needed
            "params": {}
        }
        self.collection.create_index(
            field_name=self.EMBEDDING_FIELD,
            index_params=index_params
        )

        # # Define the index parameters for an inverted index
        # index_params = {
        #     "index_type": "AUTOINDEX",  # Use 'AUTOINDEX' for scalar fields
        #     "params": {}               # Additional parameters can be specified if needed
        # }

        # # Create the index on the 'id' field
        # self.collection.create_index(
        #     field_name=self.ID_FIELD,
        #     index_params=index_params
        # )
        # print("Index created with cosine distance metric.")

        # Optionally, load the collection to memory for faster queries

    def load_collection(self):
        self.collection.load()
        print("Collection loaded to memory.")
