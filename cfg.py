EMBEDDING_MODEL_NAME = "thenlper/gte-small"
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