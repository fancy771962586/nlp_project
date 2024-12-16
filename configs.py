import yaml

class Config:

    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = {}
        self._load_config()

    def _load_config(self):
        with open(self.config_file, 'r', encoding="utf-8") as file:
            self.config_data = yaml.safe_load(file)

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config_data
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        return value



config = Config('config.yaml')
# Vector store Config
CHUNK_SIZE  = config.get('VECTOR_STORE.CHUNK_SIZE')
SEARCH_PARAMS = config.get('VECTOR_STORE.SEARCH_PARAMS')
MARKDOWN_SEPARATORS = config.get('MARKDOWN_SEPARATORS')
FILE_LIST = config.get('FILE_LIST')
# Model Config
MODEL_ROOT_PATH = config.get('MODEL_ROOT_PATH')
DENSE_EMBEDDING_MODEL_NAME = config.get('DENSE_EMBEDDING.MODEL_NAME')
DENSE_EMBEDDING_MODEL_PATH = config.get('DENSE_EMBEDDING.MODEL_PATH')
SPARSE_EMBEDDING_MODEL_NAME = config.get('SPARSE_EMBEDDING.MODEL_NAME')
SPARSE_EMBEDDING_MODEL_PATH = config.get('SPARSE_EMBEDDING.MODEL_PATH')
RERANKER_NAME = config.get('RERANKER.MODEL_NAME')
RERANKER_PATH = config.get('RERANKER.MODEL_PATH')
# OPENAI Config
OPENAI_URL = config.get('OPENAI.URL')
OPENAI_APIKEY = config.get('OPENAI.APIKEY')
OPENAI_MODEL_NAME = config.get('OPENAI.MODEL_NAME')
print(OPENAI_MODEL_NAME)




