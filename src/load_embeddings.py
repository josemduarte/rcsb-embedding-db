from embedding_loader import EmbeddingLoader


dim = 1280


def main():

    embedding_loader = EmbeddingLoader(
        "instance_embeddings",
        dim
    )
    embedding_loader.insert_folder("/mnt/vdb1/embedding-34466681", False)
    embedding_loader.insert_folder("/mnt/vdb1/csm-34466681", True)
    embedding_loader.flush()
    embedding_loader.index_collection()

    embedding_loader = EmbeddingLoader(
        "assembly_embeddings",
        dim
    )
    embedding_loader.insert_folder("/mnt/vdb1/assembly-34466681", False)
    embedding_loader.flush()
    embedding_loader.index_collection()


if __name__ == '__main__':
    main()
