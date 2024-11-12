from ragatouille import RAGPretrainedModel
from langchain.docstore.document import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from configs import RERANKER_NAME,DENSE_EMBEDDING_MODEL_PATH,DENSE_EMBEDDING_MODEL_NAME,FILE_LIST,SPARSE_EMBEDDING_MODEL_NAME
from loguru import logger
import time
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from utils import split_text_files_utils
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
def get_embedding_model(vector_type):
    doc = ['Now, I will introduce the discussion section. '
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
            'images. This could involve increasing the number of epochs, using more training data, improving the',
            "the model's generalization ability, or exploring different models if resources and time satisfied.",
            "That's our presentation, thank you.",
            "my name is yang."]
    start_time = time.time()
    if vector_type =='dense':
        logger.info('loading the embedding model...')
        embedding_model = HuggingFaceEmbeddings(
            model_name=DENSE_EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu","trust_remote_code":True},
            encode_kwargs={"normalize_embeddings": True}, # Set `True` for cosine similarity

        )
    elif vector_type =='sparse':
        # doc = split_text_files_utils(FILE_LIST)
        analyzer = build_default_analyzer(language="en")
        embedding_model = BM25EmbeddingFunction(analyzer)
        embedding_model.fit(doc)
        print("bm25_ef", embedding_model)
    else:
        raise ValueError("The type of embedding model should be either 'dense' or 'sparse'")
    end_time = time.time()
    logger.info('Finish loading model, cost {:.4f}s'.format(end_time - start_time))
    return embedding_model


def get_rerank_model():
    return RAGPretrainedModel.from_pretrained(RERANKER_NAME)

def rerank(question, relevant_docs: list[LangchainDocument], num_docs_final):
    print("=> Reranking documents...")
    rerank_model = get_rerank_model()
    relevant_docs = rerank_model.rerank(question, relevant_docs, k=num_docs_final)
    relevant_docs = [doc["content"] for doc in relevant_docs]
    return relevant_docs
def get_embedding_name(vector_type):
    if vector_type == 'dense':
        return DENSE_EMBEDDING_MODEL_NAME
    else:
        return SPARSE_EMBEDDING_MODEL_NAME