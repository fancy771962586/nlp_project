from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.corpus.reader import documents
from transformers import AutoTokenizer
from tqdm import tqdm
import datasets
import matplotlib
from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import milvus
from langchain_community.document_loaders import TextLoader
from transformers.pipelines.audio_utils import chunk_bytes_iter

from cfg import EMBEDDING_MODEL_NAME,MARKDOWN_SEPARATORS,SEARCH_PARAMS



def get_embedding():
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )
    return embedding_model

class VectorStore:
    def __init__(self,ip,port,db_name,collection_name):
        embedding_model = get_embedding()
        # embedding_model = OpenAIEmbeddings()
        connection = {"host": ip, "port": port, "db_name": db_name}
        self.vector_store = milvus.Milvus(embedding_function=embedding_model,
                                          collection_name=collection_name,
                                          auto_id=True,
                                          connection_args=connection,
                                          primary_field='ID',
                                          drop_old=True,
                                          text_field='Content',)

    # Retrieve from v_db
    def search_documents(self, question, expr=None, top_k=5, distance=SEARCH_PARAMS):
        return self.vector_store.similarity_search_with_score(query=question, expr=expr, k=top_k, kwargs=distance)

    def insert_documents(self, docs):
        texts = [d.page_content for d in docs]
        metadata = [d.metadata for d in docs]
        return self.vector_store.add_texts(texts=texts, metadatas=metadata)

def get_vector_store():
    return VectorStore('localhost', 19530, "default", 'test')



# RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=doc["text"],
#                                         metadata={"source": doc["source"]}) for doc in tqdm(ds)]
#
#
def split_documents(chunk_size: int, knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=False,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    print(knowledge_base)
    docs_processed = []
    for doc in tqdm(knowledge_base):
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def split_text_files(txt_files):
    docs = []
    for txt_file in txt_files:
        loader = TextLoader(txt_file)
        docs.extend(loader.load())
    return split_documents(chunk_size = 50, knowledge_base=docs)






if __name__ == '__main__':
    vector_db = get_vector_store()

    txt = ['ame.txt']
    # a=split_documents(100,documents)
    # print(a)
    # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    # docs = text_splitter.split_documents(documents)
    # print(docs[0].page_content)
    d = split_text_files(txt)
    print(d)
    print(vector_db.insert_documents(d))
    print(vector_db.search_documents('123',None,5))
