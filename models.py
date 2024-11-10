from ragatouille import RAGPretrainedModel
from langchain_huggingface import HuggingFaceEmbeddings
from configs import EMBEDDING_MODEL_NAME,RERANKER_NAME



def get_embedding_model():
    # embedding_model = HuggingFaceEmbeddings(
    #     model_name=EMBEDDING_MODEL_NAME,
    #     multi_process=True,
    #     model_kwargs={"device": "cpu", "trust_remote_code":True,"config_kwargs":{"use_memory_efficient_attention": False,"unpad_inputs": False}},
    #     encode_kwargs={"normalize_embeddings": True} # Set `True` for cosine similarity
    # )
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu", "trust_remote_code":True},
        encode_kwargs={"normalize_embeddings": True} # Set `True` for cosine similarity
    )
    return embedding_model


def get_rerank_model():
    return RAGPretrainedModel.from_pretrained(RERANKER_NAME)