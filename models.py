from typing import List

from ragatouille import RAGPretrainedModel
from langchain.docstore.document import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from configs import RERANKER_NAME,DENSE_EMBEDDING_MODEL_PATH,DENSE_EMBEDDING_MODEL_NAME,FILE_LIST,SPARSE_EMBEDDING_MODEL_NAME
from loguru import logger
import time
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from utils import split_text_files_utils
import os
def get_embedding_model(vector_type):
    start_time = time.time()
    if vector_type =='dense':
        logger.info('loading the dense embedding model...')
        embedding_model = HuggingFaceEmbeddings(
            model_name=DENSE_EMBEDDING_MODEL_PATH,
            multi_process=True,
            model_kwargs={"device": "cpu","local_files_only":True,"trust_remote_code":True},
            encode_kwargs={"normalize_embeddings": True}, # Set `True` for cosine similarity

        )
    elif vector_type =='sparse':
        logger.info('loading the sparse embedding model...')
        doc = split_text_files_utils(FILE_LIST)
        analyzer = build_default_analyzer(language="en")
        embedding_model = BM25EmbeddingFunction(analyzer)
        embedding_model.fit(doc)
    else:
        raise ValueError("The type of embedding model should be either 'dense' or 'sparse'")
    end_time = time.time()
    logger.info('Finish loading model, cost {:.4f}s'.format(end_time - start_time))
    logger.info('Current embedding model: {}'.format(get_embedding_name(vector_type)))
    return embedding_model

def get_rerank_model():
    logger.info("loading the reranking embedding model...")
    start_time = time.time()
    model = RAGPretrainedModel.from_pretrained(RERANKER_NAME)
    end_time = time.time()
    logger.info('Finish loading model, cost {:.4f}s'.format(end_time - start_time))
    return model

def rerank(question, relevant_docs: List, num_docs_final,rerank_model):
    logger.debug("Reranking documents...")
    rerank_res = rerank_model.rerank(question, relevant_docs, k=num_docs_final)
    text_res = [doc.get('content') for doc in rerank_res]
    return text_res, rerank_res

def get_embedding_name(vector_type):
    if vector_type == 'dense':
        return DENSE_EMBEDDING_MODEL_NAME
    else:
        return SPARSE_EMBEDDING_MODEL_NAME