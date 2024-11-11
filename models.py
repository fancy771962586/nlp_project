from ragatouille import RAGPretrainedModel
from langchain_huggingface import HuggingFaceEmbeddings
from configs import RERANKER_NAME,DENSE_EMBEDDING_MODEL_PATH,DENSE_EMBEDDING_MODEL_NAME
from loguru import logger
import time

def get_embedding_model(vector_type):
    # embedding_model = HuggingFaceEmbeddings(
    #     model_name=EMBEDDING_MODEL_NAME,
    #     multi_process=True,
    #     model_kwargs={"device": "cpu", "trust_remote_code":True,"config_kwargs":{"use_memory_efficient_attention": False,"unpad_inputs": False}},
    #     encode_kwargs={"normalize_embeddings": True} # Set `True` for cosine similarity
    # )
    start_time = time.time()
    if vector_type =='dense':
        logger.info('loading the embedding model...')
        embedding_model = HuggingFaceEmbeddings(
            model_name=DENSE_EMBEDDING_MODEL_PATH,
            multi_process=True,
            model_kwargs={"device": "cpu", "local_files_only":True,"trust_remote_code":True},
            encode_kwargs={"normalize_embeddings": True}, # Set `True` for cosine similarity

        )
    elif vector_type =='sparse':
        embedding_model = None
    else:
        raise ValueError("The type of embedding model should be either 'dense' or 'sparse'")
    end_time = time.time()
    logger.info('Finish loading model, cost {:.4f}s'.format(end_time - start_time))
    return embedding_model


def get_rerank_model():
    return RAGPretrainedModel.from_pretrained(RERANKER_NAME)