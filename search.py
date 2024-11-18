import json

def sparse_search(question, db):
    sparse_res = db.search_documents(question)
    return sparse_res

def dense_search(question, db):
    dense_res = db.search_documents(question)
    return dense_res

def mixed_search(question,sparse_db, dense_db):
    sparse_res = sparse_search(question, sparse_db)
    dense_res = dense_search(question, dense_db)
    res = sparse_res + dense_res
    return res


async def handle_response(response):
    async for chunk in response.iter_content():
        data = json.loads(chunk)
        print(data['choices'][0]['text'])
        # 处理模型生成的文本




