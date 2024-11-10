# EMBEDDING_MODEL_NAME = "thenlper/gte-small"
DENSE_EMBEDDING_MODEL_NAME =  "Alibaba-NLP/gte-large-en-v1.5"
RERANKER_NAME = "jinaai/jina-colbert-v2"
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