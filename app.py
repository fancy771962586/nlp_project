import asyncio
import time
from fastapi import FastAPI
import uvicorn
from loguru import logger
from utils import remove_duplicated_text
from vector_store import SparseVectorStore, DenseVectorStore
from models import get_rerank_model, rerank
from main import mixed_search




app = FastAPI()



@app.post("/test")
def hybrid_search(question,top_k=5):
    start_time = time.time()
    res = asyncio.run(mixed_search(question, sparse_db, dense_db))
    res = remove_duplicated_text(res)
    logger.info(f"There are total {len(res)} documents after removing duplicates")
    res_final_text, res_final = rerank(question, res, top_k, reranker)
    end_time = time.time()
    logger.info('Finish searching and reranking, cost {:.4f}s in total'.format(end_time - start_time))
    return {"response": res_final_text}


if __name__ == '__main__':
    # Initializing vector database and models
    sparse_db = SparseVectorStore('localhost', '19530', 'default', 'sparse_test')
    dense_db = DenseVectorStore('localhost', '19530', 'default', 'dense_test')
    reranker = get_rerank_model()

    # lunching the service
    uvicorn.run(app, host='0.0.0.0', port=8001)

