import asyncio
import time
from fastapi import FastAPI
import uvicorn
from loguru import logger
from utils import remove_duplicated_text
from vector_store import SparseVectorStore, DenseVectorStore
from models import get_rerank_model, rerank
from search import mixed_search
from openai import OpenAI
from configs import OPENAI_APIKEY,OPENAI_URL,OPENAI_MODEL_NAME
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, APIRouter
import asyncio
from pydantic import BaseModel
from typing import List

from openai import OpenAI
from configs import OPENAI_APIKEY,OPENAI_URL,OPENAI_MODEL_NAME

# router = APIRouter()
# Initializing vector database and models
app=FastAPI()


class Message(BaseModel):
    role: str
    content: str

class RequestBody(BaseModel):
    messages: List[Message]

class Body(BaseModel):
    question: str



async def generate_response_stream(completion):
    # try:
    #     for chunk in completion:
    #         yield chunk.model_dump_json()
    #         await asyncio.sleep(0.2)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

    for chunk in completion:
        try:
            if chunk.choices[0].finish_reason != 'stop':
                yield chunk.choices[0].delta.content
                await asyncio.sleep(0)
        except:
            break


@app.post("/chat")
def chat_llm(request_body: RequestBody):

    try:
        client = OpenAI(base_url=OPENAI_URL, api_key=OPENAI_APIKEY)
        completion = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[message.dict() for message in request_body.messages],
            stream = True
        )
        #<openai.Stream object at 0x00000216F946F7C0>
        return StreamingResponse(generate_response_stream(completion), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/search")
def hybrid_search(ques:Body):
    question = ques.question
    top_k = 5
    start_time = time.time()
    res = mixed_search(question, sparse_db, dense_db)
    res = remove_duplicated_text(res)
    logger.info(f"There are total {len(res)} documents after removing duplicates")
    logger.debug('reranking the result..., current top k:{}'.format(top_k))
    res_final_text, res_final = rerank(question, res, top_k, reranker)
    end_time = time.time()
    logger.info('Finish searching and reranking, cost {:.4f}s in total'.format(end_time - start_time))
    res_final = "\n".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(res_final_text)])
    return {"response": res_final}




if __name__ == '__main__':
    sparse_db = SparseVectorStore('localhost', '19530', 'default', 'mcsparse_test')
    dense_db = DenseVectorStore('localhost', '19530', 'default', 'mcdense_test')
    reranker = get_rerank_model()
    uvicorn.run(app, host="0.0.0.0", port=8001)






