from langchain.docstore.document import Document as LangchainDocument
from typing import List
from langchain_milvus import Milvus
from langchain_community.document_loaders import TextLoader
from milvus_model.sparse.bm25 import build_default_analyzer, BM25EmbeddingFunction
from tqdm import tqdm

from models import get_embedding_model,get_embedding_name
from configs import SEARCH_PARAMS
from loguru import logger



from pymilvus import (
FieldSchema, CollectionSchema, DataType,
Collection,connections,
)
from pymilvus import MilvusClient
from vector_store_interface import VectorStore
from utils import split_documents,txt_to_list,split_text_files_utils
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class SparseVectorStore(VectorStore):
    def __init__(self, ip, port, db_name, collection_name, is_new):

        super().__init__(ip, port, db_name)
        self.db_type = "sparse"
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
                                   vector_field='sparse_vector')









    def split_text_files(self,txt_files: list[any]) -> List[LangchainDocument]:
        return split_text_files_utils(txt_files)

    # Retrieve from v_db
    def search_documents(self, question, expr=None, top_k=5, distance=SEARCH_PARAMS):
        logger.debug(f'search documents matched with "{question}"')
        # 查询
        connections.connect("default", host="localhost", port="19530")
        client = MilvusClient(
            uri="http://localhost:19530"
        )

        collection = Collection(self.collection_name)  # Get an existing collection.
        print("collection_name",self.collection_name)
        collection.load()

        query_embeddings = self.embedding_model.encode_queries([question])
        print("embeddings_models",self.embedding_model)
        try:
            results = client.search(
                collection_name=self.collection_name,
                data=query_embeddings,
                anns_field="sparse_vector",
                limit=top_k,
                search_params={"metric_type": "IP", "params": {}},
                output_fields=["Content"])
        except Exception as e:
            logger.warning("search failed")
            results = []
        res = []



        for result in results:

            if len(list(results[0])) != 0:
                doc = LangchainDocument(page_content=dict(list(result)[0].get("entity")).get("Content"),
                                  metadatas = {})
                distance = result[0].get("distance")
                res.append((doc,distance))
        print(res)
        return res



    def insert_documents(self,doc):
        connections.connect("default", host="localhost", port="19530")
        # 定义字段
        fields = [
            FieldSchema(name="ID", dtype=DataType.VARCHAR,is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="Content", dtype=DataType.VARCHAR,max_length=512),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),

        ]

        # 创建集合模式
        schema = CollectionSchema(fields, "")
        col = Collection(self.collection_name, schema)
        # 创建稀疏倒排索引
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        print(doc)
        print()
        doc_embeddings = self.embedding_model.encode_documents(doc)
        print(doc_embeddings)
        print(len(doc))
        print(doc_embeddings.shape[0])

        # 插入数据到集合
        entities = [doc, doc_embeddings]
        col.insert(entities)
        col.flush()


        return None














class DenseVectorStore(VectorStore):
    def __init__(self, ip, port, db_name, collection_name, is_new):

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
    def split_text_files(self,txt_files: list[any]) -> List[LangchainDocument]:
        docs = []
        for txt_file in txt_files:
            result = TextLoader(txt_file, encoding='utf-8').load()
            docs.extend(result)
        return split_documents(chunk_size=200, knowledge_base=docs,db_type=self.db_type,model_name = self.embedding_name)

    # Retrieve from v_db
    def search_documents(self, question, expr=None, top_k=5, distance=SEARCH_PARAMS):
        logger.debug(f'search documents matched with "{question}"')
        return self.vector_store.similarity_search_with_score(query=question, expr=expr, k=top_k, kwargs=distance)

    def insert_documents(self, docs):
        texts = [d.page_content for d in docs]
        metadata = [d.metadata for d in docs]
        logger.debug(f'inserting {len(docs)} documents...')
        self.vector_store.add_texts(texts=texts, metadatas=metadata)











if __name__ == '__main__':


    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    txt = ["ame.txt"]
    # vector_db = DenseVectorStore('localhost', '19530', 'default','dense',
    #                               is_new=False)
    # d = vector_db.split_text_files(txt)
    # for doc in tqdm(d, total=len(d)):
    #     vector_db.insert_documents(d)

    # res = vector_db.search_documents("when was ai established",None,5)
    # print(res)
    vector_db = SparseVectorStore('localhost', '19530', 'default', 'sparse',
                                   is_new=True)
    doc = vector_db.split_text_files(txt)
    # # for doc in tqdm(d, total=len(d)):
    print("D",doc)

    vector_db.insert_documents(doc)
    vector_db.search_documents("model",None,5)


    # analyzer = build_default_analyzer(language="en")
    # embedding_model = BM25EmbeddingFunction(analyzer)
    # embedding_model.fit(d)
    # print("bm25_ef", embedding_model)
    #
    # # 插入数据库
    # doc_embeddings = embedding_model.encode_documents(d)
    # print("Document embeddings:", doc_embeddings)
    # print("Sparse dim:", embedding_model.dim, list(doc_embeddings)[0].shape)


    # 查询
    # connections.connect("default", host="localhost", port="19530")
    # client = MilvusClient(
    #     uri="http://localhost:19530"
    # )
    #
    # collection = Collection("sparse")  # Get an existing collection.
    # collection.load()
    # queries = ["model"]
    # query_embeddings = embedding_model.encode_queries(queries)
    # print(type(query_embeddings))
    # print("query_embeddings", query_embeddings)
    #
    #
    # res = client.search(
    #     collection_name="sparse",
    #     data=query_embeddings,
    #     anns_field="sparse_vector",
    #     limit=3,
    #     search_params={"metric_type": "IP", "params": {}},
    #     output_fields=["Content"])
    # print(res)

    # docs = []
    #
    # #连接milvus数据库
    # vector_db = get_vector_store('test', 'sparse', is_new=False,doc=docs)
    # spilt = vector_db.split_text_files(txt)
    # print("split",spilt)
    #
    #
    #
    # #插入数据库
    # doc_embeddings = vector_db.bm25_ef.encode_documents(docs)
    # print("Document embeddings:", doc_embeddings)
    # print("Sparse dim:", vector_db.bm25_ef.dim, list(doc_embeddings)[0].shape)
    # # 把处理好的数据插入milvus
    # vector_db.insert_collection(docs,doc_embeddings)
    #
    #
    #
    # #查询
    # connections.connect("default", host="localhost", port="19530")
    # client = MilvusClient(
    #     uri="http://localhost:19530"
    # )
    #
    # collection = Collection("test")  # Get an existing collection.
    # collection.load()
    # queries = ["When was AI established"]
    # query_embeddings = vector_db.bm25_ef.encode_queries(queries)
    # print(type(query_embeddings))
    # print("query_embeddings", query_embeddings)
    # # print("0",query_embeddings[0])
    # # print("1",query_embeddings[1])
    # print("[:,0]",query_embeddings[:, [1]])
    #
    # res = client.search(
    #     collection_name="test",
    #     data=query_embeddings,
    #     anns_field="sparse_vector",
    #     limit=3,
    #     search_params={"metric_type": "IP", "params": {}},
    #     output_fields=["Content"])
    # print(res)






