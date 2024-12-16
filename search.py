import json
import asyncio

def sparse_search(question, db):
    sparse_res = db.search_documents(question)
    return sparse_res

def dense_search(question, db):
    dense_res = db.search_documents(question)
    return dense_res

async def mixed_search(question,sparse_db, dense_db):
    sparse_task = asyncio.to_thread(sparse_search, question, sparse_db)
    dense_task = asyncio.to_thread(dense_search, question, dense_db)
    # 使用 asyncio.gather 并行执行这两个异步任务
    sparse_res, dense_res = await asyncio.gather(sparse_task, dense_task)
    res = sparse_res + dense_res
    return res

async def handle_response(response):
    async for chunk in response.iter_content():
        data = json.loads(chunk)
        print(data['choices'][0]['text'])
        # 处理模型生成的文本




