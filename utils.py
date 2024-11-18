from configs import CHUNK_SIZE
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoTokenizer
from loguru import logger
from configs import RERANKER_NAME,DENSE_EMBEDDING_MODEL_PATH,DENSE_EMBEDDING_MODEL_NAME,FILE_LIST, USR_PROMPT,SYS_PROMPT
from langchain.docstore.document import Document as LangchainDocument
from typing import List
from configs import MARKDOWN_SEPARATORS
import re


def extract_txt(text):
    # 使用正则表达式匹配数字 12123
    if 'User' in text:
        match = re.search(r"<div class='message'>(.*?)</div>", text, re.DOTALL)
    else:
        match = re.search(r"<div class='label'><b>McSmartBot</b></div><div class='message'>(.*?)</div>", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def save_history(chat_history, turn):
    messages=[]
    for history in chat_history:
        # history = json.loads(history)
        messages.append({'role':history.get('role'),'content':extract_txt(history.get('content'))})
        print(messages)
    return messages[-turn*2:]



def history_to_str(messages):
    msg = ''
    for history in messages:
        role = history.get('role')
        text = history.get('content')
        if text is None:
            text =''
        msg += '{}: {}\n'.format(role, text)
    return msg





def split_text_files_utils(txt_files: list[any]) -> List[LangchainDocument]:
    docs = []
    for txt_file in txt_files:
        result = txt_to_list(txt_file)
        docs.extend(result)
    return split_documents(chunk_size=CHUNK_SIZE, knowledge_base=docs, db_type="sparse",model_name=None)


def txt_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read()  # Read all lines from the file
            # Split each line by commas and flatten the list

            return [lines]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def remove_duplicated_text(docs_processed):
    unique_texts = {}
    docs_processed_unique = []
    docs = [doc[0].page_content for doc in docs_processed]
    for doc in docs:
        if doc not in unique_texts:
            unique_texts[doc] = True
            docs_processed_unique.append(doc)
    return docs_processed_unique


def _remove_duplicates(docs_processed: List,db_type):
    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if db_type == 'dense':
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)
        else:
            if doc not in unique_texts:
                unique_texts[doc] = True
                docs_processed_unique.append(doc)
    return docs_processed_unique


def split_documents(chunk_size: int, knowledge_base: List,db_type,model_name) -> List:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    logger.info("Current chunk size: {}".format(chunk_size))
    docs_processed = []
    if db_type == 'dense':
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(model_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=False,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )
        for doc in tqdm(knowledge_base):
            docs_processed += text_splitter.split_documents([doc])
    elif db_type == 'sparse':
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name = 'gpt-4o',
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
    return _remove_duplicates(docs_processed,db_type)


def convert_to_langchain_documents(docs):
    langchain_docs = []
    for text in docs:
        langchain_doc = LangchainDocument(text)
        langchain_docs.append(langchain_doc)
    return langchain_docs



def build_prompt(context, history, question):
    sys_prompt = SYS_PROMPT.format(context=context)
    user_prompt = USR_PROMPT.format(history=history, question=question)
    final_prompt = {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            }

    return final_prompt, sys_prompt

def show_doc_content(relevant_docs):
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i}------------------------------------------------------------")
        print(doc.page_content)



