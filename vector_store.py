from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.corpus.reader import documents
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Optional, List, Tuple
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_milvus import Milvus
from langchain_community.document_loaders import TextLoader
from models import get_embedding_model, get_rerank_model
from configs import MARKDOWN_SEPARATORS,SEARCH_PARAMS,DENSE_EMBEDDING_MODEL_NAME,SPARSE_EMBEDDING_MODEL_NAME
from loguru import logger


from rank_bm25 import BM25Okapi
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus import MilvusClient
from pymilvus import (
utility,
FieldSchema, CollectionSchema, DataType,
Collection, AnnSearchRequest, RRFRanker, connections,
)
from pymilvus import MilvusClient




class VectorStore:
    def __init__(self,ip,port,db_name,collection_name,db_type, is_new,doc):
        embedding_model = get_embedding_model(db_type)
        self.db_type = db_type
        if self.db_type == 'dense':
            self.model_name = DENSE_EMBEDDING_MODEL_NAME
        else:
            self.model_name = None
            analyzer = build_default_analyzer(language="en")
            embedding_model = BM25EmbeddingFunction(analyzer)
            embedding_model.fit(doc)
            print("bm25_ef",embedding_model)
            self.bm25_ef = embedding_model

        connection = {"host": ip, "port": port, "db_name": db_name}
        self.vector_store = Milvus(embedding_function=embedding_model,
                                          collection_name=collection_name,
                                          auto_id=True,
                                          connection_args=connection,
                                          primary_field='ID',
                                          drop_old=is_new,
                                          text_field='Content',
                                   vector_field='sparse_vector')

    def _remove_duplicates(self,docs_processed: List):
        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if self.db_type == 'dense':
                if doc.page_content not in unique_texts:
                    unique_texts[doc.page_content] = True
                    docs_processed_unique.append(doc)
            else:
                if doc not in unique_texts:
                    unique_texts[doc] = True
                    docs_processed_unique.append(doc)
        return docs_processed_unique

    def split_documents(self,chunk_size: int, knowledge_base: List) -> List:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        docs_processed = []
        if self.db_type == 'dense':
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                AutoTokenizer.from_pretrained(self.model_name),
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size / 10),
                add_start_index=False,
                strip_whitespace=True,
                separators=MARKDOWN_SEPARATORS,
            )
            for doc in tqdm(knowledge_base):
                docs_processed += text_splitter.split_documents([doc])
        elif self.db_type == 'sparse':
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size / 10),
                add_start_index=False,
                strip_whitespace=True,
                separators=MARKDOWN_SEPARATORS,
            )
            for doc in tqdm(knowledge_base):
                docs_processed += text_splitter.split_text(doc)

        else:
            raise ValueError("Unsupported database type")
        return self._remove_duplicates(docs_processed)





    def split_text_files(self,txt_files: list[any]) -> List[LangchainDocument]:
        docs = []
        for txt_file in txt_files:
            if self.db_type == 'dense':
                result = TextLoader(txt_file, encoding='utf-8').load()

            else:
                result = txt_to_list(txt_file)
                docs.extend(result)

            docs.extend(result)
        return self.split_documents(chunk_size=100, knowledge_base=docs)

    # Retrieve from v_db
    def search_documents(self, question, expr=None, top_k=5, distance=SEARCH_PARAMS):
        logger.debug(f'search documents matched with "{question}"')
        return self.vector_store.similarity_search_with_score(query=question, expr=expr, k=top_k, kwargs=distance)



    def insert_documents(self, docs):
        texts = [d.page_content for d in docs]
        metadata = [d.metadata for d in docs]
        logger.debug(f'inserting {len(docs)} documents...')
        return self.vector_store.add_texts(texts=texts, metadatas=metadata)

    def insert_collection(self,doc,doc_embeddings):
        connections.connect("default", host="localhost", port="19530")
        # 定义字段
        fields = [
            FieldSchema(name="ID", dtype=DataType.VARCHAR,is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="Content", dtype=DataType.VARCHAR,max_length=512),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),

        ]

        # 创建集合模式
        schema = CollectionSchema(fields, "")
        col = Collection("test", schema)
        # 创建稀疏倒排索引
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        # 插入数据到集合
        entities = [doc, doc_embeddings]
        col.insert(entities)
        col.flush()


        return None

    def delete_documents_by_ids(self, ids:list[str]):
        self.vector_store.delete(ids)


def get_vector_store(collection_name, dbtype, is_new=False,doc: list = None):
    """
    `dbtype` would either be 'dense' or 'sparse'.  `is_new` is a flag would be set to True or False. If creating new
    database, set `is_new`=True. False is the default value of `is_new`.
    """
    logger.info(f'Current database collection: {collection_name}')
    return VectorStore('localhost', 19530, "default", collection_name, dbtype, is_new,doc)

def show_doc_content(relevant_docs):
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i}------------------------------------------------------------")
        print(doc[0].page_content)

def rerank(question,relevant_docs:list[LangchainDocument],num_docs_final):
        print("=> Reranking documents...")
        rerank_model = get_rerank_model()
        relevant_docs = rerank_model.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]
        return relevant_docs


def txt_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()  # Read all lines from the file
            # Split each line by commas and flatten the list

            return lines
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []














if __name__ == '__main__':
    docss= [
        "The field of artificial intelligence was established as an academic subject in 1956.",
        "Alan Turing was the pioneer in conducting significant research in artificial intelligence.",
        "Originating in Maida Vale, London, Turing grew up in the southern regions of England.",
        "In 1956, artificial intelligence emerged as a scholarly field.",
        "Turing, originally from Maida Vale, London, was brought up in the south of England."
    ]

    docs = ['Now, I will introduce the discussion section. '
            'As we mentioned before, our project on Hand Pose',
            'Hand Pose Recognition is divided into two parts: hand detection and hand pose recognition. We used',
            'We used two different models for these tasks which are YOLOv3 and MobileNetv2',
            'For hand detection, we chose the YOLOv3 model, the reason is that it a classic algorithm for object',
            "object detection using neural networks. It's known for its feature pyramid network, which has high",
            'has high accuracy and speed. It is capable of detecting objects in various environments and scenes.',
            'scenes. However, it does have some limitations, particularly when dealing with small objects due to',
            'due to their small number of pixel points. Also, It Uses a meshing method rather than a pixel-level',
            'method, resulting in poor performance in dealing with target edges. Additionally, YOLOv3 is pretty',
            'is pretty complicated, it requires a large number of parameters to be trained, which can be',
            'can be time-consuming and resource-intensive.',
            'Consider the larger number of training data and limited time, we choose MobileNetv2 for the second',
            'second task which is hand pose recognition. As a neural network, MobileNetv2 is smaller, faster,',
            'faster, and requires significantly fewer parameters than larger networks like YOLOv3. In other',
            'In other words, it requires less memory and computational effort than classic large networks.',
            'However, our model does have some limitations. For the Handpose Recognition task, While it performs',
            'performs well on images with clear hand contours, it struggles with images with unclear hand',
            'hand contours or those taken from specific angles.',
            'Future work will focus on optimizing the model to improve its performance on these challenging',
            'images. This could involve increasing the number of epochs, using more training']




    xx = [""]
    #处理document，txt---list，doc为转换后的list
    txt = ["ame.txt"]
    #连接milvus数据库
    vector_db = get_vector_store('test', 'sparse', is_new=True,doc=docs)
    spilt = vector_db.split_text_files(txt)
    print("split",spilt)



    #插入数据库
    doc_embeddings = vector_db.bm25_ef.encode_documents(docs)
    print("Document embeddings:", doc_embeddings)
    print("Sparse dim:", vector_db.bm25_ef.dim, list(doc_embeddings)[0].shape)
    # 把处理好的数据插入milvus
    vector_db.insert_collection(docs,doc_embeddings)



    #查询
    connections.connect("default", host="localhost", port="19530")
    client = MilvusClient(
        uri="http://localhost:19530"
    )

    collection = Collection("test")  # Get an existing collection.
    collection.load()
    queries = ["model is what?"]
    query_embeddings = vector_db.bm25_ef.encode_queries(queries)
    print(type(query_embeddings))
    print("query_embeddings", query_embeddings)
    # print("0",query_embeddings[0])
    # print("1",query_embeddings[1])
    print("[:,0]",query_embeddings[:, [1]])

    res = client.search(
        collection_name="test",
        data=query_embeddings,
        anns_field="sparse_vector",
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["Content"])
    print(res)






