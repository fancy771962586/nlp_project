from torch.nn.init import sparse
from utils import remove_duplicated_text
from vector_store import SparseVectorStore, DenseVectorStore
from models import rerank, get_rerank_model
from loguru import logger
import time
import asyncio

async def sparse_search(question, db):
    sparse_res = db.search_documents(question)
    return sparse_res

async def dense_search(question, db):
    dense_res = db.search_documents(question)
    return dense_res

async def mixed_search(question,sparse_db, dense_db):
    task1 = asyncio.create_task(sparse_search(question, sparse_db))
    task2 = asyncio.create_task(dense_search(question, dense_db))
    sparse_res = await task1
    dense_res = await task2
    res = sparse_res + dense_res
    return res







