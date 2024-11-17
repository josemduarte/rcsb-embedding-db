from embedding_loader import EmbeddingLoader


dim = 1280


def main():

    embedding_loader = EmbeddingLoader(
        "instance_embeddings",
        dim
    )
    embedding_loader.insert_folder("/mnt/vdb1/embedding-34466681")
    embedding_loader.flush()
    embedding_loader.index_collection()


if __name__ == '__main__':
    main()
