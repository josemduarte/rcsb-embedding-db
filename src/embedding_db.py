
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections, utility
)


class EmbeddingDB:

    HOST = 'localhost'
    PORT = '19530'
    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'

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

        ids = df[self.ID_FIELD].tolist()
        embeddings = [embedding.tolist() for embedding in df[self.EMBEDDING_FIELD]]

        entities = [
            ids,         # List of identifiers
            embeddings   # List of embeddings
        ]

        print(f"Inserting {len(ids)} rows into collection {self.collection.name}")
        insert_result = self.collection.insert(entities)
        print(f"Inserted {len(insert_result.primary_keys)} entities into the collection.")

        # Flush to make sure data is persisted
        self.collection.flush()

    def index_collection(self):
        # Create an index on the embedding field with cosine distance
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",  # You can choose other index types as needed
            "params": {"M": 16, "efConstruction": 100}
        }
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print("Index created with cosine distance metric.")

        # Define the index parameters for an inverted index
        index_params = {
            "index_type": "AUTOINDEX",  # Use 'AUTOINDEX' for scalar fields
            "params": {}               # Additional parameters can be specified if needed
        }

        # Create the index on the 'id' field
        self.collection.create_index(field_name="id", index_params=index_params)

        # Optionally, load the collection to memory for faster queries
        self.collection.load()
