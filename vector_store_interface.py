from utils import split_documents,txt_to_list
from langchain.docstore.document import Document as LangchainDocument
from typing import Optional, List, Tuple
from langchain_milvus import Milvus
from loguru import logger
from pymilvus import (
utility,
FieldSchema, CollectionSchema, DataType,
Collection, AnnSearchRequest, RRFRanker, connections,
)
from langchain_community.document_loaders import TextLoader

from models import get_embedding_model
from  configs import DENSE_EMBEDDING_MODEL_NAME,SEARCH_PARAMS


class VectorStore:
    def __init__(self,ip,port,db_name):
        self.db_type = None
        self.embedding_model = None
        self.model_name = None
        self.vector_store = None
        #connection = {"host": ip, "port": port, "db_name": db_name}








    def split_text_files(self,txt_files: list[any]) -> List[LangchainDocument]:
        raise NotImplemented("Split text files not implemented yet.")

    # Retrieve from v_db
    def search_documents(self, question, expr=None, top_k=5, distance=SEARCH_PARAMS):
        raise NotImplemented("Search documents not implemented yet.")



    def insert_documents(self, docs):
        raise NotImplemented("Insert documents not implemented yet.")



    def delete_documents_by_ids(self, ids:list[str]):
        self.vector_store.delete(ids)


