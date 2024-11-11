from fastapi import FastAPI
import uvicorn
from vector_store import get_vector_store,show_doc_content
app = FastAPI()


@app.post("/test")
async def home(q):
    results = vector_db.search_documents(q, None, 5)
    text = show_doc_content(results)
    return {"response": text}


if __name__ == '__main__':
    vector_db = get_vector_store('test', 'dense', is_new=False)

    uvicorn.run(app, host='0.0.0.0', port=8000)