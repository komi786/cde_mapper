
import os
from datetime import datetime
CLASS = 'omop_v5.4' #class
os.environ['HF_HOME'] = 'resources/models'
CUDA_NUM = 0 # used GPU num
LOOK_UP_FILE = "data/output/lookup.csv"
CROSS_MODEL_ID = "ncbi/MedCPT-Cross-Encoder"
EMB_MODEL_NAME  ="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
VECTOR_PATH="komal.qdrant.137.120.31.148.nip.io"   #:6333
QDRANT_PORT = 443
SYN_COLLECTION_NAME = 'concept_mapping_1'
#"HIERARCHY_AWARE_SYNONYMS_MAPPING_SAPBERT"   #we embed entities entity+synonym as whole instead of there aggregation
SYN_SPARSE_COLLECTION_NAME = "SYNONYMS_MAPPING_SPARSE_SPLADE"  # SYNONYMS_MAPPING_SPARSE
SYN_SPARSE_COLLECTION_NAME_2 = "SYNONYMS_MAPPING_SPARSE_SPLADE_ST_SLAMUR"  # SYNONYMS_MAPPING_SPARSE with semantic type and few vocabs  = snomed, loinc, atc, mesh,usum,rxnorm
SELECTED_COLLECTION_NAME = None
DB_FILE = 'variables.db'
NEAREST_SAMPLE_NUM = 64
QUANT_TYPE = 'scalar'
LLM_ID = 'llama3.1'   #'gpt-4o-mini'
CANDIDATE_GENERATOR_BATCH_SIZE = 64
CACHE_DIR = 'resources/models'
LLAMA_CACHE_DIR = 'resources/models/llama'
DATA_DIR="data"
RETRIEVER = 'dense+sparse'
MODEL_NAME = 'llama3.1'   #'gpt-4o-mini'
# MAPPING_CACHE_FILE = 'data/output/mapping.csv'
# MAPPING_CACHE_FILE = f"../data/output/mapping_{MODEL_NAME}.csv"
# DB_PATH = '/workspace/rag_pipeline/db/llama_mapping_bm25.csv'
TOPK = 10
CHAT_HISTORY_FILE = f"data/output/chat_history{MODEL_NAME}.pkl"
LEARNING_RATE = 1e-5
DOCUMENT_PATH = f"{DATA_DIR}/output/bm25docs.pkl"
GRAPH_DATA = f"{DATA_DIR}/input/omop_v5.4/omop_bi_graph.pkl"
INPUT_DATA_PATH = f"{DATA_DIR}/input/omop_v5.4/concepts.csv"
OUTPUT_DATA_PATH = f"{DATA_DIR}/output/sapbert_emb_docs_json.jsonl"
# concepts.jsonl,data/output/sapbert_emb_docs_json.jsonl
DES_DICT_PATH = None #description data path
LOG_FILE = f"resources/logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
DES_LIMIT_LENGTH = 256
MAPPING_FILE = f"{DATA_DIR}/input/mapping_templates.json"
SELECTOR_PATH = "data/output/selector_.pkl"
