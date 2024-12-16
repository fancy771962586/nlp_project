from typing import List
from FlagEmbedding import FlagReranker
# from ragatouille import RAGPretrainedModel

from configs import RERANKER_NAME,DENSE_EMBEDDING_MODEL_PATH,DENSE_EMBEDDING_MODEL_NAME,FILE_LIST,SPARSE_EMBEDDING_MODEL_NAME,OPENAI_APIKEY, OPENAI_URL
from loguru import logger
import time
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from utils import split_text_files_utils,get_data
from langchain_openai import OpenAIEmbeddings
import os
def get_embedding_model(vector_type):
    start_time = time.time()
    if vector_type =='dense':
        logger.info('loading the dense embedding model...')
        # embedding_model = HuggingFaceEmbeddings(
        #     model_name=DENSE_EMBEDDING_MODEL_PATH,
        #     multi_process=True,
        #     model_kwargs={"device": "cuda:0","trust_remote_code":True},
        #     encode_kwargs={"normalize_embeddings": True}, # Set `True` for cosine similarity
        #
        # )
        embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_APIKEY, openai_api_base=OPENAI_URL,
                                  model=DENSE_EMBEDDING_MODEL_NAME)
    elif vector_type =='sparse':
        logger.info('loading the sparse embedding model...')
        doc = split_text_files_utils(get_data())
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
    # model = RAGPretrainedModel.from_pretrained(RERANKER_NAME)
    model = FlagReranker(RERANKER_NAME,use_fp16=True)
    end_time = time.time()
    logger.info('Finish loading model, cost {:.4f}s'.format(end_time - start_time))
    return model

# def rerank_jina(question, relevant_docs: List, num_docs_final,rerank_model):
#     logger.debug("Reranking documents...")
#
#     rerank_res = rerank_model.rerank(question, relevant_docs, k=num_docs_final)
#     print("rerank:{}".format(rerank_res))
#     text_res = [doc.get('content') for doc in rerank_res]
#     return text_res, rerank_res

def get_embedding_name(vector_type):
    if vector_type == 'dense':
        return DENSE_EMBEDDING_MODEL_NAME
    else:
        return SPARSE_EMBEDDING_MODEL_NAME

def rerank(question, relevant_docs: List, num_docs_final,rerank_model):
    passage_pair = [[question,doc] for doc in relevant_docs]
    rank_result= []
    text_res = []
    for pair in passage_pair:
        pair_res = {'content': pair[1], 'score': rerank_model.compute_score(pair, normalize=True)[0], 'rank': None}
        rank_result.append(pair_res)
    final = sorted(rank_result, key=lambda x: x['score'], reverse=True)
    for i, doc in enumerate(final, start=1):
        doc['rank'] = i
        text_res.append(doc.get('content'))
    return text_res[:num_docs_final], final[:num_docs_final]




# if __name__ == '__main__':
    # reranker = FlagReranker('BAAI/bge-reranker-v2-m3',
    #                         use_fp16=True)
    # jina = get_rerank_model()
    # a,b= rerank('what is your name',['myname is fancy','i am ur dad','i am eating monkey','fucking u'],4,reranker)
    # c, d = rerank2('what is your name', ['myname is fancy', 'i am ur dad', 'i am eating monkey', 'fucking u'], 4,
    #               jina)
    # print('a\n',a,'b\n',b,'c\n',c,'d\n')