
from pymilvus import (
    connections, Collection
)


class EmbeddingProvider:

    HOST = 'localhost'
    PORT = '19530'
    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    CSM_FLAG = 'is_csm'
    BATCH_SIZE = 2000

    def __init__(
            self,
            collection_name
    ):
        self.connect()
        self.collection = Collection(name=collection_name)

    def connect(self):
        connections.connect(
            host=self.HOST,
            port=self.PORT
        )

    def get_by_embedding(self, query_embedding, is_csm=True, n_results=100):
        return self.collection.search(
            data=[query_embedding],
            expr=f'{self.CSM_FLAG} == False' if not is_csm else None,
            anns_field=self.EMBEDDING_FIELD,
            limit=n_results,
            param={
                "metric_type": "COSINE",
                "params": {}
            }
        )

    def get_by_id(self, query_id):
        result = self.collection.query(
            expr=f'{self.ID_FIELD} == "{query_id}"',
            output_fields=[self.EMBEDDING_FIELD]
        )
        if len(result) == 0:
            return None
        return result[0][self.EMBEDDING_FIELD]
