from ragatouille import RAGPretrainedModel
from langchain_huggingface import HuggingFaceEmbeddings
from configs import DENSE_EMBEDDING_MODEL_NAME,RERANKER_NAME



def get_embedding_model(vector_type):
    # embedding_model = HuggingFaceEmbeddings(
    #     model_name=EMBEDDING_MODEL_NAME,
    #     multi_process=True,
    #     model_kwargs={"device": "cpu", "trust_remote_code":True,"config_kwargs":{"use_memory_efficient_attention": False,"unpad_inputs": False}},
    #     encode_kwargs={"normalize_embeddings": True} # Set `True` for cosine similarity
    # )
    if vector_type =='dense':
        embedding_model = HuggingFaceEmbeddings(
            model_name=DENSE_EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu", "trust_remote_code":True},
            encode_kwargs={"normalize_embeddings": True} # Set `True` for cosine similarity
        )
    elif vector_type =='sparse':
        embedding_model = None
    else:
        raise ValueError("The type of embedding model should be either 'dense' or 'sparse'")
    return embedding_model


def get_rerank_model():
    return RAGPretrainedModel.from_pretrained(RERANKER_NAME)