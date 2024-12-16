from langchain.docstore.document import Document as LangchainDocument
from typing import List
from langchain_milvus import Milvus
from langchain_community.document_loaders import TextLoader
from milvus_model.sparse.bm25 import build_default_analyzer, BM25EmbeddingFunction
from pymilvus.exceptions import SchemaNotReadyException
from sqlalchemy.orm.collections import collection
from tqdm import tqdm
from models import get_embedding_model,get_embedding_name
from configs import SEARCH_PARAMS, CHUNK_SIZE
from loguru import logger
from pymilvus import (
FieldSchema, CollectionSchema, DataType,
Collection,connections,
)
from pymilvus import MilvusClient
from vector_store_interface import VectorStore
from utils import split_documents, txt_to_list, split_text_files_utils, show_doc_content, get_data
import os

def has_collection(collection_name):
    try:
        connections.connect('default', host="localhost", port=19530)
        collection = Collection(collection_name)
        collection.load()
        return True
    except SchemaNotReadyException:
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False



class SparseVectorStore(VectorStore):
    """Class of Sparse Vector"""
    def __init__(self, ip, port, db_name, collection_name, is_new=False):
        super().__init__(ip, port, db_name)
        self.db_type = "sparse"
        self.embedding_model = get_embedding_model(self.db_type)
        self.embedding_name = get_embedding_name(self.db_type)
        self.collection_name = collection_name
        self.collection = None
        if has_collection(collection_name):
            connections.connect(db_name, host="localhost", port=port)
            self.client = MilvusClient(
                uri="http://localhost:19530"
            )
            self.collection=Collection(self.collection_name)
            self.collection.load()
            logger.success("Connected with Sparse Vector Database")


    def split_text_files(self,txt_files: list[any]) -> List[LangchainDocument]:
        logger.debug("Splitting text files into chunks...")
        return split_text_files_utils(txt_files)

    # Retrieve from v_db
    def search_documents(self, question, expr=None, top_k=5, distance=SEARCH_PARAMS):
        logger.debug(f'search documents matched with "{question}" in sparse database')
        # 查询
        # connections.connect("default", host="localhost", port="19530")
        # client = MilvusClient(
        #     uri="http://localhost:19530"
        # )
        # collection = Collection(self.collection_name)  # Get an existing collection.
        # collection.load()
        query_embeddings = self.embedding_model.encode_queries([question])
        try:
            if self.collection is not None:
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=query_embeddings,
                    anns_field="sparse_vector",
                    limit=top_k,
                    search_params={"metric_type": "IP", "params": {}},
                    output_fields=["Content"])
            else:
                raise ValueError("collection is not defined")
        except Exception as e:
            logger.error("search failed with: {}".format(e))
            results = []
        res = []
        for result in results[0]:
            if len(list(results[0])) != 0:
                doc = LangchainDocument(page_content=result.get("entity").get("Content"),
                                  metadatas = {})
                distance = result.get("distance")
                res.append((doc,distance))
        logger.info(f"Found {len(res)} results in Sparse Vector Database")
        return res

    def insert_documents(self,doc):
        connections.connect("default", host="localhost", port="19530")
        # 定义字段
        fields = [
            FieldSchema(name="ID", dtype=DataType.VARCHAR,is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="Content", dtype=DataType.VARCHAR,max_length=16384), # var length != token
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),

        ]
        # 创建集合模式
        schema = CollectionSchema(fields, "")
        col = Collection(self.collection_name, schema)
        # 创建稀疏倒排索引
        logger.debug('Creating sparse index...')
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        logger.debug(f'BM25 embedding documents...')
        doc_embeddings = self.embedding_model.encode_documents(doc)
        # 插入数据到集合
        entities = [doc, doc_embeddings]
        logger.debug(f'Inserting documents into sparse database')
        col.insert(entities)
        col.flush()




class DenseVectorStore(VectorStore):
    """Class of Dense Vector"""
    def __init__(self, ip, port, db_name, collection_name, is_new=False):
        super().__init__(ip, port, db_name)
        self.db_type = "dense"
        self.embedding_model = get_embedding_model(self.db_type)
        self.embedding_name = get_embedding_name(self.db_type)
        self.collection_name = collection_name
        connection = {"host": ip, "port": port, "db_name": db_name}
        self.vector_store = Milvus(embedding_function=self.embedding_model,
                                          collection_name=self.collection_name,
                                          auto_id=True,
                                          connection_args=connection,
                                          primary_field='ID',
                                          drop_old=is_new,
                                          text_field='Content',
                                   vector_field='dense_vector')
        logger.success("Connected with Dense Vector Database")

    def split_text_files(self,txt_files: list[any]) -> List[LangchainDocument]:
        logger.debug("Splitting text files into chunks...")
        docs = []
        for txt_file in txt_files:
            print(txt_file)
            result = TextLoader(txt_file, encoding='utf-8').load()
            docs.extend(result)
        return split_documents(chunk_size=CHUNK_SIZE, knowledge_base=docs,db_type=self.db_type,model_name = self.embedding_name)

    # Retrieve from v_db
    def search_documents(self, question, expr=None, top_k=5, distance=SEARCH_PARAMS):
        logger.debug(f'search documents matched with "{question}" in dense database')
        try:
            res = self.vector_store.similarity_search_with_score(query=question, expr=expr, k=top_k, kwargs=distance)
            logger.info(f"Found {len(res)} results in Dense Vector Database")
        except Exception as e:
            logger.error("search failed")
            res = []
        return res

    def insert_documents(self, docs: list[LangchainDocument]):
        texts = [d.page_content for d in docs]
        metadata = [d.metadata for d in docs]
        logger.debug(f'inserting {len(docs)} documents into dense database...')
        self.vector_store.add_texts(texts=texts, metadatas=metadata)
        logger.success(f'inserting {len(docs)} document(s) successfully')






if __name__ == '__main__':

    print(has_collection('MC_SPARSE'))
    # print(connections.has_connection('MC_SPARSE'))

###inserting data###
    # txt = get_data()
    # test dense

    # vector_db = DenseVectorStore('localhost', '19530', 'default','MC_DENSE',
    #                               is_new=False)
    # d = vector_db.split_text_files(txt)
    # print(d)
    # show_doc_content(d)
    # # #
    # for doc in tqdm(d, total=len(d)):
    #     vector_db.insert_documents([doc])
    # r = vector_db.search_documents("Where did first McDonald’s in mainland China opened",None,5)
    # print("results:", r)


    # test sparse
    # vector_db = SparseVectorStore('localhost', '19530', 'default', 'MC_SPARSE',
    #                                is_new=False)
    # doc = vector_db.split_text_files(txt)
    # print(len(doc))
    # # for d in doc:
    # #     print("__________\n")
    # #     print(d)
    #
    # vector_db.insert_documents(doc)
    # # r=vector_db.search_documents("McDonald",None,5)
    # # print("results:", r)







