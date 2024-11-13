CHUNK_SIZE = 100
MODEL_ROOT_PATH = "C:/Users/Administrator.DESKTOP-LTJNBTE/.cache/huggingface/hub/"
DENSE_EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
DENSE_EMBEDDING_MODEL_PATH = MODEL_ROOT_PATH + "models--Alibaba-NLP--gte-large-en-v1.5/snapshots/104333d6af6f97649377c2afbde10a7704870c7b"
SPARSE_EMBEDDING_MODEL_NAME = 'BM25'
SPARSE_EMBEDDING_MODEL_PATH = ''
RERANKER_NAME = "jinaai/jina-colbert-v2"
RERANKER_PATH = MODEL_ROOT_PATH+"models--jinaai--jina-colbert-v2/snapshots/4cf816e5e2b03167b132a3c847a9ecd48ba708e1"
OPENAI_URL='https://xiaoai.plus/v1'
OPENAI_APIKEY='sk-KVKO74Uiu07jU8EwzZUARrfuYtiFDXWMt7Hgv2GPaElTLI9i'
OPENAI_MODEL_NAME = 'gpt-4o'
FILE_LIST = ['ame.txt']
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",]

SEARCH_PARAMS = {
    "metric_type": "IP",
    "params": {"nprobe": 10},
}