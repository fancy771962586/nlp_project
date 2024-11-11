from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.corpus.reader import documents
from sqlalchemy import false
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Optional, List, Tuple
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_milvus import Milvus
from langchain_community.document_loaders import TextLoader
from models import get_embedding_model, get_rerank_model
from configs import MARKDOWN_SEPARATORS,SEARCH_PARAMS,DENSE_EMBEDDING_MODEL_NAME,SPARSE_EMBEDDUNG_MODEL_NAME
from loguru import logger


class VectorStore:
    def __init__(self,ip,port,db_name,collection_name,db_type, is_new):
        embedding_model = get_embedding_model(db_type)
        if db_type == 'dense':
            self.model_name = DENSE_EMBEDDING_MODEL_NAME
        else:
            self.model_name = None

        connection = {"host": ip, "port": port, "db_name": db_name}
        self.vector_store = Milvus(embedding_function=embedding_model,
                                          collection_name=collection_name,
                                          auto_id=True,
                                          connection_args=connection,
                                          primary_field='ID',
                                          drop_old=False,
                                          text_field='Content',)

    def _split_documents(self,chunk_size: int, knowledge_base: List[LangchainDocument]) -> List[LangchainDocument]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(self.model_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=False,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )
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

    def split_text_files(self,txt_files: list[any]) -> List[LangchainDocument]:
        docs = []
        for txt_file in txt_files:
            loader = TextLoader(txt_file, encoding='utf-8')
            docs.extend(loader.load())
        return self._split_documents(chunk_size=100, knowledge_base=docs)

    # Retrieve from v_db
    def search_documents(self, question, expr=None, top_k=5, distance=SEARCH_PARAMS):
        logger.debug(f'search documents matched with "{question}"')
        return self.vector_store.similarity_search_with_score(query=question, expr=expr, k=top_k, kwargs=distance)

    def insert_documents(self, docs):
        texts = [d.page_content for d in docs]
        metadata = [d.metadata for d in docs]
        logger.debug(f'inserting {len(docs)} documents...')
        return self.vector_store.add_texts(texts=texts, metadatas=metadata)

    def delete_documents_by_ids(self, ids:list[str]):
        self.vector_store.delete(ids)


def get_vector_store(collection_name, dbtype, is_new=False):
    """
    `dbtype` would either be 'dense' or 'sparse'.  `is_new` is a flag would be set to True or False. If creating new
    database, set `is_new`=True. False is the default value of `is_new`.
    """
    logger.info(f'Current database collection: {collection_name}')
    return VectorStore('localhost', 19530, "default", collection_name, dbtype, is_new)

def show_doc_content(relevant_docs):
    results = []
    output = None
    for i, doc in enumerate(relevant_docs):
        results.append(doc[0].page_content)
        print(f"Document {i}------------------------------------------------------------")
        print(doc[0].page_content)
        output = '\n'.join(results)
    return output

def rerank(question,relevant_docs:list[LangchainDocument],num_docs_final):
        print("=> Reranking documents...")
        rerank_model = get_rerank_model()
        relevant_docs = rerank_model.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]
        return relevant_docs



if __name__ == '__main__':
    import time
    search_latency_fmt = "search latency = {:.4f}s"
    start_time = time.time()
    vector_db = get_vector_store('test', 'dense', is_new=False)
    # txt = ['ame.txt']
    #
    # d = vector_db.split_text_files(txt)
    # print(d)
    # vector_db.insert_documents(d)
    # print(vector_db.search_documents('what is the name of the author',None,10))
    results = vector_db.search_documents('what is the name of the author', None, 5)
    end_time = time.time()

    show_doc_content(results)
    print(search_latency_fmt.format(end_time - start_time))