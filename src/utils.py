from langchain.schema import Document
import json
import tiktoken
import torch
import logging
from itertools import chain
import pickle
from collections.abc import Hashable
from typing import Any, List, Tuple, Dict,Iterable,Callable,Iterator,TypeVar
import os
import sys
from rag.param import *
from tqdm import tqdm
import psutil
import pandas as pd
import time
import re


def monitor_performance():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        logging.info(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
        time.sleep(10)  # Adjust the frequency as needed
        
STOP_WORDS = ['stop','start','combinations','combination','various combinations','various','left','right','blood','finding','finding status',
              'status','extra','point in time','pnt','oral','product','oral product','several','types','several types','random','nominal',
              'p time','quant','qual','quantitative','qualitative','ql','qn','quan','anti','antibodies','wb','whole blood','serum','plasma','diseases',
              'disorders','disorder','disease','lab test','measurements','lab tests','meas value','measurement','procedure','procedures',
              'panel','ordinal','after','before','survey','level','levels','others','other','p dose','dose','dosage','frequency','calc','calculation',
              'calculation method','method','calc method','calculation methods','methods','calc methods','calculation method','calculation methods',
              'measurement','measurements','meas value','meas values','meas','meas val','meas vals','meas value','meas values','meas','meas val', 'vals', 'val'
    
]


def filter_synonyms(entity_name, entity_synonyms: set, stop_words: list=STOP_WORDS):

        cleaned_synonyms = set()
        for synonym in entity_synonyms:
            cleaned = synonym.lower().replace(',', ' ').strip() if synonym else synonym
            words = set(cleaned.split())
            
            # Check if any whole word in the synonym is a stop word
            if any(word in stop_words for word in words) or len(cleaned) < 2:
                continue  # Skip synonyms that contain stop words as standalone words

            # Add the cleaned synonym if it isn't a substring of any existing synonym
            # and doesn't fully contain any existing synonym as a full word
            if not any(cleaned in s for s in cleaned_synonyms) and not any(word in s.split() for s in cleaned_synonyms for word in words) and cleaned != entity_name:
                cleaned_synonyms.add(cleaned)
        return cleaned_synonyms

       

def save_jsonl(data, file):
 with open(file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
 print("Data saved to file.")

def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            # print(doc.json())
            jsonl_file.write(doc.model_dump_json() + '\n')

def create_document(data):
    try:
        # print(data.keys())
        # Check if 'page_content' exists in data, use an empty string as default if not
        page_content = data.get('kwargs', {}).get('page_content', {})
        print(f"page_content={page_content}")
        # Access 'metadata' safely
        metadata = data.get('kwargs', {}).get('metadata', {})
        
        # Create the Document object
        document = Document(
            page_content=page_content,
            metadata=metadata
        )
        return document

    except Exception as e:
        print(f"Error loading document: {e}")
        # Return None or handle the error appropriately (perhaps re-raise the exception or log it)
        return None
    
    
def load_custom_docs_from_jsonl(file_path) -> list:
    docs = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            try:
                obj = Document(**data)
            except Exception as e:
                print(f"document object translated into Dictionary format")
                obj=create_document(data)
            docs.append(obj)
    
    print(f"Total Custom Documents: {len(docs)}")
    return docs

def load_docs_from_jsonl(file_path) -> list:
    docs_dict = {}
    count = 0
    with open(file_path, 'r') as jsonl_file:
        print("Opening file...")
        for line in tqdm(jsonl_file, desc="Loading Documents"):
            # if count >= 100:
            #     break
            data = json.loads(line)
            try:
                obj = Document(**data)
            except Exception as e:
                print(f"document object translated into Dictionary format")
                obj=create_document(data)
            # print(f"data={obj}")
            if 'vocab' in obj.metadata:
                vocab = obj.metadata['vocab'].lower()
                #  for ncbi. miid and bc5csr dataset
                #['snomed', 'loinc', 'atc', 'ucum', 'rxnorm', 'omop extension', 'mesh','meddra','cancer modifier']
                #['snomed','icd10cm','icd10','mesh','meddra','icd9cm']
                if vocab in ['atc','loinc',  'ucum', 'rxnorm', 'omop extension', 'mesh','meddra','cancer modifier','snomed','rxnorm extension']:
                    # Define a unique key based on page content and critical metadata
                    # This might include other metadata fields you consider critical for uniqueness
                    # Here we use a combination of page_content and a sorted JSON dump of metadata to ensure the key is unique and consistently formatted
                    key = (obj.page_content, json.dumps(obj.metadata, sort_keys=True))
                    # Only add to dictionary if it is truly unique
                    if key not in docs_dict:
                        docs_dict[key] = obj
                        count += 1
            else:
                key = (obj.page_content, json.dumps(obj.metadata, sort_keys=True))
                if key not in docs_dict:
                    docs_dict[key] = obj
                    count += 1

    # Convert dictionary values to a sorted list to process documents in a specific order
    
    sorted_docs = sorted(docs_dict.values(), key=lambda doc: doc.metadata['vocab'].lower()) if 'vocab' in docs_dict.values() else sorted(docs_dict.values(), key=lambda doc: doc.metadata['label'].lower())
    print(f"Total Unique Documents: {len(sorted_docs)}\n")
    return sorted_docs

# def load_docs_from_jsonl(file_path) -> list:
#     """
#     Loads documents from a JSONL file, ensuring uniqueness based on
#     'label', 'synonyms', 'vocab', 'concept_class', and 'domain' metadata fields.

#     Args:
#         file_path (str): Path to the JSONL file.

#     Returns:
#         list: A list of unique Document objects.
#     """
#     docs_dict = {}
#     unique_fields = ['label', 'synonyms', 'vocab', 'concept_class', 'domain']
#     count = 0

#     try:
#         with open(file_path, 'r', encoding='utf-8') as jsonl_file:
#             logging.info(f"Opening file: {file_path}")
#             for line in tqdm(jsonl_file, desc="Loading Documents"):
#                 try:
#                     data = json.loads(line)
#                     obj = Document(**data)
#                 except Exception as e:
#                     logging.error(f"Error creating Document from line: {e}")
#                     continue  # Skip malformed lines

#                 # Extract relevant metadata fields
#                 metadata = obj.metadata
#                 key_fields = ['label','vocab','concept_class','domain']
#                 for field in unique_fields:
#                     # Use lowercase for consistency and default to empty string if missing
#                     key_fields.append(metadata.get(field, '').lower())

#                 # Define the unique key as a tuple of the selected fields
#                 key = tuple(key_fields)

#                 if key not in docs_dict:
#                     docs_dict[key] = obj
#                     count += 1
#                 # Optionally, you can implement a limit for testing
#                 # if count >= 100:
#                 #     break

#         # Convert dictionary values to a list
#         unique_docs = list(docs_dict.values())

#         # Optionally, sort the documents based on 'vocab' or 'label'
#         try:
#             unique_docs = sorted(unique_docs, key=lambda doc: doc.metadata.get('vocab', '').lower())
#         except KeyError:
#             # Fallback to sorting by 'label' if 'vocab' is missing
#             unique_docs = sorted(unique_docs, key=lambda doc: doc.metadata.get('label', '').lower())

#         logging.info(f"Total Unique Documents: {len(unique_docs)}")
#         return unique_docs

#     except FileNotFoundError:
#         logging.error(f"File not found: {file_path}")
#         return []
#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#         return []

logger = logging.getLogger(__name__)

def init_log(args, stdout_only=False):
    logger = logging.getLogger(__name__)

    # Setup logging level
    if torch.distributed.is_initialized():
        # If this is part of a distributed system, synchronize here
        torch.distributed.barrier()
        # Set different logging levels depending on whether this is the main process
        if torch.distributed.get_rank() == 0:
            log_level = logging.INFO
        else:
            log_level = logging.WARN
    else:
        # If not distributed or the rank cannot be determined, default to INFO
        log_level = logging.INFO

    # Create stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)

    # Define log format
    formatter = logging.Formatter("[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", "%m/%d/%Y %H:%M:%S")
    stdout_handler.setFormatter(formatter)

    # Handlers list
    handlers = [stdout_handler]

    # Create file handler if needed
    if not stdout_only and hasattr(args, 'output_dir'):
        log_file_path = os.path.join(args.output_dir, "run.log")
        file_handler = logging.FileHandler(filename=log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Basic configuration of logging
    logging.basicConfig(level=log_level, handlers=handlers)

    return logger
def save_json_data(file_path,data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {file_path}")
    
def init_logger(log_file_path=LOG_FILE):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG
    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Set the logging level for the file handler

    # Create a stream handler (to print to console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Set the logging level for the stream handler

    # Define the format for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
global_logger = init_logger()
    
def save_txt_file(file_path, data):
    with open(file_path, 'a') as file:
        for item in data:
            file.write(f"{item}\n")
    print(f"Total Data = {len(data)} saved to file.")
def save_documents(filepath: str, docs):
        """Save the BM25 documents to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(docs, f)
            
from langchain_core.messages import HumanMessage
from typing import List, Dict

def load_documents(filepath: str) -> List[Document]:
    """Load the BM25 documents from a file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_chat_history(file_path: str, history: List[HumanMessage]):
    with open(file_path, 'wb') as f:
        pickle.dump(history, f)

# Function to load chat history from a file
def load_chat_history(file_path: str) -> List[HumanMessage]:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return []

# def symlink_force(target, link_name):
#     try:
#         os.symlink(target, link_name)
#     except OSError as e:
#         if e.errno == errno.EXIST:
#             os.remove(link_name)
#             os.symlink(target, link_name)
#         else:
#             raise e
    
# def save(model, optimizer, scheduler, step, opt, dir_path, name):
#     model_to_save = model.module if hasattr(model, "module") else model
#     path = os.path.join(dir_path, "checkpoint")
#     epoch_path = os.path.join(path, name)  # "step-%s" % step)
#     os.makedirs(epoch_path, exist_ok=True)
#     cp = os.path.join(path, "latest")
#     fp = os.path.join(epoch_path, "checkpoint.pth")
#     checkpoint = {
#         "step": step,
#         "model": model_to_save.state_dict(),
#         "optimizer": optimizer.state_dict(),
#         "scheduler": scheduler.state_dict(),
#         "opt": opt,
#     }
#     torch.save(checkpoint, fp)
#     symlink_force(epoch_path, cp)
#     if not name == "lastlog":
#         logger.info(f"Saving model to {epoch_path}")


############ OPTIM


# class WeightedAvgStats:
#     """provides an average over a bunch of stats"""

#     def __init__(self):
#         self.raw_stats: Dict[str, float] = defaultdict(float)
#         self.total_weights: Dict[str, float] = defaultdict(float)

#     def update(self, vals: Dict[str, Tuple[Number, Number]]) -> None:
#         for key, (value, weight) in vals.items():
#             self.raw_stats[key] += value * weight
#             self.total_weights[key] += weight

#     @property
#     def stats(self) -> Dict[str, float]:
#         return {x: self.raw_stats[x] / self.total_weights[x] for x in self.raw_stats.keys()}

#     @property
#     def tuple_stats(self) -> Dict[str, Tuple[float, float]]:
#         return {x: (self.raw_stats[x] / self.total_weights[x], self.total_weights[x]) for x in self.raw_stats.keys()}

#     def reset(self) -> None:
#         self.raw_stats = defaultdict(float)
#         self.total_weights = defaultdict(float)

#     @property
#     def average_stats(self) -> Dict[str, float]:
#         keys = sorted(self.raw_stats.keys())
#         if torch.distributed.is_initialized():
#             torch.distributed.broadcast_object_list(keys, src=0)
#         global_dict = {}
#         for k in keys:
#             if not k in self.total_weights:
#                 v = 0.0
#             else:
#                 v = self.raw_stats[k] / self.total_weights[k]
#             v, _ = dist_utils.weighted_average(v, self.total_weights[k])
#             global_dict[k] = v
#         return global_dict

# def parse(string_value, local_llm=False):
#     if local_llm:
#         string_value = string_value.split("assistant<|end_header_id|>")[-1]
#         print(string_value)
#     try:
#         return json.loads(string_value)  # Use json.loads for strings
#     except json.JSONDecodeError as e:
#         print(f"Failed to decode JSON: {e}")
#         return None
# def load_hf(object_class, model_name):
#     try:
#         obj = object_class.from_pretrained(model_name, local_files_only=True)
#     except:
#         obj = object_class.from_pretrained(model_name, local_files_only=False)
#     return obj

def extract_cid(description):
    # Extracts the CID number from a description if present
    prefix = 'CID: '
    start = description.find(prefix)
    if start != -1:
        start += len(prefix)
        end = description.find(')', start)
        return description[start:end]
    return None


VOCAB_CACHE = {}


def load_vocabulary(file_path=MAPPING_FILE):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config['vocabulary_rules']
def select_vocabulary(query_text=None, config_path=MAPPING_FILE, domain=None):
    global VOCAB_CACHE
    # Normalize the domain name to lower case or set to 'unknown' if not provided
    domain = domain.lower() if domain else 'all'
    
    # Check if the vocabulary for the domain is alRETRIEVER_CACHEready loaded
    if domain in VOCAB_CACHE:
        selected_vocab = VOCAB_CACHE[domain]
    else:
        # Load the configuration file if the domain's vocabulary isn't cached
        vocabulary_rules = load_vocabulary(config_path)
        
        # Get domain-specific vocabulary or default to 'unknown' if not found
        selected_vocab = vocabulary_rules['domains'].get(domain, vocabulary_rules['domains']['unknown'])
        
        # Cache the selected vocabulary
        VOCAB_CACHE[domain] = selected_vocab

    # print(f"Selected vocabulary for query={query_text}====Domain={domain}====Vocabulary={selected_vocab}")

    return selected_vocab


# def select_vocabulary(query_text=None, config_path=MAPPING_FILE, domain=None):
#     with open(config_path, 'r') as file:
#         config = json.load(file)
#     vocabulary_rules = config['vocabulary_rules']
#     # Load domain-specific vocabulary, defaulting if the domain is not found
#     domain = domain.lower() if domain else 'unknown'
#     selected_vocab = vocabulary_rules['domains'].get(domain, vocabulary_rules['domains']['unknown'])
#     print(f"Selected vocabulary for query={query_text}====Domain={domain}====Vocabulary={selected_vocab}")
    # Check for special rules applicable to the domain
    # special_rules = vocabulary_rules.get('special_rules', {})
    # if query_text and domain in special_rules:
    #     query_text = query_text.lower()
    #     rule = special_rules[domain]
    #     # Create regex pattern to match any of the triggers
    #     pattern = re.compile(r'\b(' + '|'.join(map(re.escape, rule['triggers'])) + r')\b', re.IGNORECASE)
    #     if re.search(pattern, query_text):
    #         # Modify the vocabulary based on triggers
    #         updated_vocab = [vocab for vocab in selected_vocab if vocab not in rule.get('exclude_vocab', [])]
    #         updated_vocab.extend(rule.get('additional_vocab', []))
    #         selected_vocab = updated_vocab
    #         print(f"Updated Selected vocabulary for query={query_text}====Domain={domain}====Vocabulary={selected_vocab}")

    # return selected_vocab

# def load_config(config_path=MAPPING_FILE):
#     with open(config_path, 'r') as file:
#         config = json.load(file)

#     # Precompile regex patterns for each domain in special rules
#     special_rules = config['vocabulary_rules'].get('special_rules', {})
#     for domain, rule in special_rules.items():
#         rule['compiled_pattern'] = re.compile(r'\b(' + '|'.join(map(re.escape, rule['triggers'])) + r')\b', re.IGNORECASE)

#     return config

    # elif domain == 'observation':
    #     vocab =  ['loinc','snomed','icd10',"uk biobank","atc","ucum"]
    # elif domain == 'unit':
    #     vocab =  ['ucum']
    # elif domain == 'family history':
    #     vocab =  ['snomed']
      # elif domain == 'measurement':
    #     vocab = ["loinc","ucum"]
        # if domain == 'condition':
    #     if custom_data == False:
    #         vocab = ["snomed","uk biobank","loinc"]
    #     else:
    #         vocab = ["snomed","icd10","loinc","uk biobank"]

def post_process_candidates(candidates: List[Document], max=1):
    processed_candidates = []
    seen = set()
    print(f"Total Candidates={len(candidates)}")
    if not candidates:
        print("No candidates found.")
        return processed_candidates
    
    first_label = None  # To store the label of the first document for comparison
    
    for index, doc in enumerate(candidates):
        if doc.metadata['sid'] in seen:
            continue  # Skip if the SID has already been processed
        seen.add(doc.metadata['sid'])
        
        # Create a dictionary for the current document
        current_doc_dict = {
            'standard_label': normalize(doc.metadata['label']),
            'semantic_type': f"{doc.metadata['concept_class']}:{doc.metadata['domain']}",
            'standard_code': f"{doc.metadata['vocab']}:{doc.metadata['scode']}",
            'concept_id': doc.metadata['sid']
        }
        # Always process the first document or any document if max > 1
        if index == 0:
            first_label = doc.metadata['label']
            first_doc_dict = current_doc_dict
            processed_candidates.append(first_doc_dict)
        elif index > 0 and doc.metadata['label'] == first_label:
            # If another document's label matches the first and only modifying the first if max == 1
            if max == 1:
                first_doc_dict['standard_code'] += f", {current_doc_dict['standard_code']}"
                first_doc_dict['concept_id'] += f", {current_doc_dict['concept_id']}"
            else:
                processed_candidates.append(current_doc_dict)
        
        # Exit loop early if we've reached the desired number of processed candidates
        if len(processed_candidates) >= max:
            break
    
    return processed_candidates



import csv
def save_to_csv(data, filename):
    if not data:
        return

    fieldnames = [
        'VARIABLE LABEL', 'VARIABLE LABEL_REVISED', 'DOMAIN', 'Label Concept Name', 
        'Label Concept ID','Label Concept CODE' ,'CATEGORICAL VALUES', 'CATEGORICAL VALUES CODES','Categorical Value Concept Name', 'UNIT', 'UNIT CODE','Unit Concept Name'
    ]

    # Map and combine fields in the data rows
    def map_and_combine_fields(row):
        # Map fields
        mapped_row = {
            'VARIABLE LABEL': row.get('query_text', ''),
            'VARIABLE LABEL_REVISED': row.get('revised_query', ''),
            'Label Concept Name': row.get('standard_label'),
            'Label Concept ID': '',
            'Label Concept CODE': '',
            'DOMAIN': row.get('domain', ''),
            'CATEGORICAL VALUES': row.get('categorical_values', ''),
            'CATEGORICAL VALUES CODES': row.get('categorical_codes', ''),
            'Categorical Value Concept Name':row.get('categorical_values', ''),
            'UNIT': row.get('unit', ''),
            'UNIT CODE': row.get('unit_code', ''),
            'Unit Concept Name': row.get('unit', ''),
        }
        
        # Combine fields
        label_ids = '|'.join(filter(None, [row.get('standard_concept_id'), row.get('additional_context_concept_ids')]))
        label_codes = '|'.join(filter(None, [row.get('standard_code'), row.get('additional_context_codes')]))
        mapped_row['Label Concept ID'] = label_ids
        mapped_row['Label Concept CODE'] = label_codes
        return mapped_row

    with open(filename, mode='w', newline='') as file:
        dict_writer = csv.DictWriter(file, fieldnames=fieldnames)
        dict_writer.writeheader()
        for row in data:
            combined_row = map_and_combine_fields(row)
            dict_writer.writerow(combined_row)
        
def load_mapping(filename, domain):
    print(f"domain={domain}")
    try:
        with open(filename, 'r') as file:
            data = json.load(file)

        domain = domain if domain else 'all'
        # print(f"domain={domain}")
        mapping = data['mapping_rules'].get(domain, {})
        
        # Get examples or default to empty list if not present
        relevance_examples = data.get('rel_relevance_examples', {}).get(domain, [])
        ranking_examples = data.get('ranking_examples', {}).get(domain, [])
        # print(f"ranking_examples={ranking_examples[:2]} for domain={domain}")
        # Format examples as string representations of dictionaries
        relevance_examples_string = [
            {"input": ex["input"], "output": str([
                f"{{'answer': '{out['answer']}', 'relationship': '{out['relationship']}', 'explanation': '{out['explanation']}'}}"
                for out in ex["output"]
            ])}
            for ex in relevance_examples
        ]

        ranking_examples_string = [
            {"input": ex["input"], "output": str([
                f"{{'answer': '{out['answer']}', 'score': '{out['score']}', 'explanation': '{out['explanation']}'}}"
                for out in ex["output"]
            ])}
            for ex in ranking_examples
        ]
        # print(f"ranking_examples_string={ranking_examples_string[:2]}")

        if not mapping:
            return None, ranking_examples_string, relevance_examples_string

        return {
            "prompt": mapping.get('description', 'No description provided.'),
            "examples": mapping.get('example_output', [])
        }, ranking_examples_string, relevance_examples_string

    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None, None, None
    except json.JSONDecodeError:
        print("JSON decoding error.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None
def parse_term(extracted_terms, domain):
    domain = domain.lower() if domain else 'all'
    if domain in extracted_terms.keys():
        term = extracted_terms[domain]
        if domain == 'condition':
            if 'procedure' in extracted_terms:
                procedure = extracted_terms['procedure']
                
                
def save_result_to_jsonl(array:Iterable[dict], file_path:str)->None:
    print(f"Saving to file: {file_path}")
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            json_string = json.dumps(doc)
            print(json_string)
            jsonl_file.write(json_string + '\n')
    print(f"Saved {len(array)} documents to {file_path}")


# def exact_match_found(query_text, documents, domain=None):
#     # logger.info(f"Searching for exact match for query={query_text}====Domain={domain}")
#     # print(f"Searching for exact match for query={query_text}====Domain={domain}")
#     # print(f"Find Exact Match docs={documents}")
#     # start_time = time.time()
#     matched_docs = []
#     # semantic_type = None
#     query_text =  query_text.lower() if query_text else None
#     if documents is None:
#         # print(f"NO DOCUMENTS FOUND FOR QUERY={query_text}")
#         return []
#     selected_vocab = select_vocabulary(query_text, domain=domain)
#     for doc in documents:
#         if 'scode' in doc.metadata and 'vocab' in doc.metadata:
#             if (query_text == normalize(doc.metadata['label']) and doc.metadata['vocab'] in selected_vocab) or (query_text == doc.metadata['scode'] and doc.metadata['vocab'] in selected_vocab):
#             # print(f"EXACT MATCH FOUND FOR QUERY={query_text}")
#                 matched_docs.append(doc)
#         else:
#             if query_text.strip().lower() == normalize(doc.metadata['label']):
#                 print(f"EXACT MATCH FOUND FOR QUERY={query_text}")
#                 matched_docs.append(doc)
#     # end_time = time.time()
#     # print(f"Exact Match Search Time: {end_time - start_time:.4f} seconds")
#     return matched_docs[:1]

from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher

def exact_match_found(query_text, documents, domain=None):

    if not query_text or not documents:
        print("NO DOCUMENTS FOUND FOR QUERY={query_text}")
        return []

    # Create a database and populate it
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    for doc in documents:
        if 'label' in doc.metadata:
            db.add(normalize(doc.metadata['label']))

    # Create a searcher with cosine similarity
    searcher = Searcher(db, CosineMeasure())

    # Normalize query text
    normalized_query = normalize(query_text)
    results = searcher.search(normalized_query, 0.95)  # Set threshold to 0.95 for high similarity

    matched_docs = []
    selected_vocab = select_vocabulary(query_text, domain=domain)

    for result in results:
        for doc in documents:
            if normalize(doc.metadata['label']) == result:
                if 'vocab' in doc.metadata and doc.metadata['vocab'] in selected_vocab:
                    print(f"EXACT MATCH FOUND FOR QUERY={query_text}")
                    matched_docs.append(doc)
    return matched_docs[:1]


def exact_match_wo_vocab(query_text, documents, domain=None):

    if not query_text or not documents:
        print("NO DOCUMENTS FOUND FOR QUERY={query_text}")
        return []

    # Create a database and populate it
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    for doc in documents:
        if 'label' in doc.metadata:
            db.add(normalize(doc.metadata['label']))

    # Create a searcher with cosine similarity
    searcher = Searcher(db, CosineMeasure())

    # Normalize query text
    normalized_query = normalize(query_text)
    results = searcher.search(normalized_query, 0.95)  # Set threshold to 0.95 for high similarity

    matched_docs = []
    for result in results:
        for doc in documents:
            if normalize(doc.metadata['label']) == result:
                print(f"EXACT MATCH FOUND FOR QUERY={query_text}")
                matched_docs.append(doc)
    return matched_docs[:1]

def create_document_string(doc):
    label = normalize(doc.metadata['label'])
    # concept_class, domain = doc.metadata['ctype'].split(':')
    # label = f"{label}: Domain:{domain}"
    #['label1,'label2....labeln']
    return label

from json_repair import repair_json
def fix_json_quotes(json_like_string):
    try:
        # Trying to convert it directly
        return repair_json(json_like_string, return_objects=True)       
    except json.JSONDecodeError as e:
            print(f"Failed to parse JSON after trying to fix quotes, error: {str(e)}")
            return None
        
        
        
def handle_query_timing(index):
    if index % 50 == 0:
        time.sleep(0.00005)
    if (index + 1) % 100 == 0:
        time.sleep(0.005)
        
        
# def extract_domain_info(query, all_queries_domain):
#     if isinstance(query, Document):
#         return all_queries_domain.get(query.page_content, ('unknown', None, None, None, None, None))
#     return ('unknown', None, None, None, None, None)


def format_query_result(query, domain_info):
    initial_domain, mapping_code, standard_labels, values, values_codes, unit, unit_code = domain_info
    return {
        'query': query,
        'revised_query': query,
        'domain': initial_domain,
        'standard_label': standard_labels,
        "standard_code": mapping_code,
        'values': values,
        'value_codes': values_codes,
        'unit': unit,
        'unit_code': unit_code
    }
    
    """
source code from : "https://github.com/facebookresearch/contriever/blob/main/src/normalize_text.py"

"""

#: Control characters.
CONTROLS = {
    '\u0001', '\u0002', '\u0003', '\u0004', '\u0005', '\u0006', '\u0007', '\u0008', '\u000e', '\u000f', '\u0011',
    '\u0012', '\u0013', '\u0014', '\u0015', '\u0016', '\u0017', '\u0018', '\u0019', '\u001a', '\u001b',
}
# There are further control characters, but they are instead replaced with a space by unicode normalization
# '\u0009', '\u000a', '\u000b', '\u000c', '\u000d', '\u001c',  '\u001d', '\u001e', '\u001f'


#: Hyphen and dash characters.
HYPHENS = {
    '-',  # \u002d Hyphen-minus
    '‐',  # \u2010 Hyphen
    '‑',  # \u2011 Non-breaking hyphen
    '⁃',  # \u2043 Hyphen bullet
    '‒',  # \u2012 figure dash
    '–',  # \u2013 en dash
    '—',  # \u2014 em dash
    '―',  # \u2015 horizontal bar
}

#: Minus characters.
MINUSES = {
    '-',  # \u002d Hyphen-minus
    '−',  # \u2212 Minus
    '－',  # \uff0d Full-width Hyphen-minus
    '⁻',  # \u207b Superscript minus
}

#: Plus characters.
PLUSES = {
    '+',  # \u002b Plus
    '＋',  # \uff0b Full-width Plus
    '⁺',  # \u207a Superscript plus
}

#: Slash characters.
SLASHES = {
    '/',  # \u002f Solidus
    '⁄',  # \u2044 Fraction slash
    '∕',  # \u2215 Division slash
}

#: Tilde characters.
TILDES = {
    '~',  # \u007e Tilde
    '˜',  # \u02dc Small tilde
    '⁓',  # \u2053 Swung dash
    '∼',  # \u223c Tilde operator #in mbert vocab
    '∽',  # \u223d Reversed tilde
    '∿',  # \u223f Sine wave
    '〜',  # \u301c Wave dash #in mbert vocab
    '～',  # \uff5e Full-width tilde #in mbert vocab
}

#: Apostrophe characters.
APOSTROPHES = {
    "'",  # \u0027
    '’',  # \u2019
    '՚',  # \u055a
    'Ꞌ',  # \ua78b
    'ꞌ',  # \ua78c
    '＇',  # \uff07
}

#: Single quote characters.
SINGLE_QUOTES = {
    "'",  # \u0027
    '‘',  # \u2018
    '’',  # \u2019
    '‚',  # \u201a
    '‛',  # \u201b

}

#: Double quote characters.
DOUBLE_QUOTES = {
    '"',  # \u0022
    '“',  # \u201c
    '”',  # \u201d
    '„',  # \u201e
    '‟',  # \u201f
}

#: Accent characters.
ACCENTS = {
    '`',  # \u0060
    '´',  # \u00b4
}

#: Prime characters.
PRIMES = {
    '′',  # \u2032
    '″',  # \u2033
    '‴',  # \u2034
    '‵',  # \u2035
    '‶',  # \u2036
    '‷',  # \u2037
    '⁗',  # \u2057
}

#: Quote characters, including apostrophes, single quotes, double quotes, accents and primes.
QUOTES = APOSTROPHES | SINGLE_QUOTES | DOUBLE_QUOTES | ACCENTS | PRIMES

def normalize(text):
    if text is None:
        return None
    text = str(text)
    # Replace control characters
    for control in CONTROLS:
        text = text.replace(control, '')
    text = text.replace('\u000b', ' ').replace('\u000c', ' ').replace(u'\u0085', ' ')

    # Replace hyphens and minuses with '-'
    for hyphen in HYPHENS | MINUSES:
        text = text.replace(hyphen, '-')
    text = text.replace('\u00ad', '')

    # Replace various quotes with standard quotes
    for double_quote in DOUBLE_QUOTES:
        text = text.replace(double_quote, '"')
    for single_quote in (SINGLE_QUOTES | APOSTROPHES | ACCENTS):
        text = text.replace(single_quote, "'")
    text = text.replace('′', "'")     # \u2032 prime
    text = text.replace('‵', "'")     # \u2035 reversed prime
    text = text.replace('″', "''")    # \u2033 double prime
    text = text.replace('‶', "''")    # \u2036 reversed double prime
    text = text.replace('‴', "'''")   # \u2034 triple prime
    text = text.replace('‷', "'''")   # \u2037 reversed triple prime
    text = text.replace('⁗', "''''")  # \u2057 quadruple prime
    text = text.replace('…', '...').replace(' . . . ', ' ... ')  # \u2026


    # Replace slashes with '/'
    for slash in SLASHES:
        text = text.replace(slash, '/')

    # Ensure there's only one space between words
    text = ' '.join(text.split())
    text = text.lower()
    if text in ['null','nil', 'none', 'n/a','','nan']:
        return None
    return text

def normalize_page_content(page_content):
    if page_content is None:
        return page_content
    page_content = page_content.strip().lower()
    
    if '<ent>' in page_content:
        page_content = page_content.split('<ent>')[1].split('</ent>')[0]
        if '||' in page_content:
            page_content = page_content.split('||')[0]
        if '.' in page_content:
            page_content = page_content.split('.')[0]
    elif '||' in page_content:
        page_content = page_content.split('||')[0]
    elif '.' in page_content:
        page_content = page_content.split('.')[0]
    # print(f"Page Content: {page_content}")
    return page_content

BASE_IRI = "http://ccb.hms.harvard.edu/t2t/"

STOP_WORDS = {'in', 'the', 'any', 'all', 'for', 'and', 'or', 'dx', 'on', 'fh', 'tx', 'only', 'qnorm', 'w', 'iqb', 's',
              'ds', 'rd', 'rdgwas', 'ICD', 'excluded', 'excluding', 'unspecified', 'certain', 'also', 'undefined',
              'ordinary', 'least', 'squares', 'FINNGEN', 'elsewhere', 'more', 'excluded', 'classified', 'classifeid',
              'unspcified', 'unspesified', 'specified', 'acquired', 'combined', 'unspeficied', 'elsewhere', 'not', 'by',
              'strict', 'wide', 'definition', 'definitions', 'confirmed', 'chapter', 'chapters', 'controls',
              'characterized', 'main', 'diagnosis', 'hospital', 'admissions', 'other', 'resulting', 'from'}

TEMPORAL_WORDS = {'age', 'time', 'times', 'date', 'initiation', 'cessation', 'progression', 'duration', 'early', 'late',
                  'later', 'trimester'}

QUANTITY_WORDS = {'hourly', 'daily', 'weekly', 'monthly', 'yearly', 'frequently', 'per', 'hour', 'day', 'week', 'month',
                  'year', 'years', 'total', 'quantity', 'amount', 'level', 'levels', 'volume', 'count', 'counts',
                  'percentage', 'abundance', 'proportion', 'content', 'average', 'prevalence', 'mean', 'ratio'}

BOLD = "\033[1m"
END = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def check_extracted_details(query_details):
    if query_details:
        if 'unit' in query_details:
            if isinstance(query_details['unit'], list):
                if len(query_details['unit']) > 0:
                    unit_value = normalize(query_details['unit'][0])
                    if unit_value != 'unknown':
                        query_details['unit'] = unit_value 
                    else:
                        del query_details['unit']
                else:
                    del query_details['unit']
                    
        if 'status' in query_details:
            if isinstance(query_details['status'], list):
                if len(query_details['status']) > 0:
                    query_details['status'] = [normalize(status) for status in query_details['status'] if status != 'unknown']
                else:
                    del query_details['status']
            else:
                status_val =  normalize(query_details['status'])
                if status_val != 'unknown':
                    query_details['status'] = [status_val]
                else:
                    del query_details['status']
    return query_details

def revise_query(original_query, query, context, status, unit):
    if query == original_query:
        return query
    if context:
        query = f"base_entity:{query}, context: {context}"
    if status:
        query = f"{query},status:{status}"
    if unit:
        query = f"{query},unit:{unit}"
    return query
    
def get_component_value(primary, cache, cache_key, component_key):
    if primary:
        return primary
    if cache:
        if cache.get(component_key) and cache_key in cache[component_key]:
            return cache[component_key][cache_key]
    return None
def remove_duplicates(values):
    seen = set()
    return [x for x in values if not (x in seen or seen.add(x))]


def join_or_single(items):
    unique_items = remove_duplicates(filter(None, items))
    return '|'.join(unique_items) if len(unique_items) > 1 else unique_items[0] if unique_items else ''
def create_processed_result(query_text, main_term, processed_docs, domain, values_docs=None, status_docs=None, unit_docs=None, context=None, status=None, unit=None, component_cache=None):
    
    """
    Processes the input query and maps it to standardized concepts, returning the processed result.

    Parameters:
    -----------
    query_text : str
        The original query text that needs to be processed.
    main_term : str
        The main term to be mapped from the query.
    processed_docs : list of dict
        A list of dictionaries containing processed documents with information like standard labels, codes, and concept IDs.
    domain : str
        The domain or category the query belongs to (e.g., medical condition, measurement).
    values_docs : list of dict, optional
        A list of documents related to additional values (e.g., context) that need to be processed.
    status_docs : list of dict, optional
        A list of documents related to categorical values (e.g., status) that need to be processed.
    unit_docs : list of dict, optional
        A list of documents related to unit information that need to be processed.
    context : str, optional
        Additional context to help process and map the query.
    status : str, optional
        Status information (e.g., Yes/No) related to the query.
    unit : str, optional
        The unit associated with the main term.
    component_cache : dict, optional
        A cache that stores pre-processed component data for faster lookups.

    Returns:
    --------
    dict
        A dictionary containing the processed result, including:
        - 'query_text': The original query.
        - 'revised_query': The revised version of the query.
        - 'domain': The domain of the query.
        - 'standard_label': The combined standard label for the query.
        - 'standard_code': The combined standard codes.
        - 'standard_concept_id': The combined concept IDs.
        - 'additional_context', 'categorical_values', 'unit': Processed values, status, and unit information.
    """
    values_docs = get_component_value(values_docs, component_cache, 'additional_context', 'context')
    status_docs = get_component_value(status_docs, component_cache, 'categorical_values', 'status')
    unit_docs_label = get_component_value(unit_docs[0]['standard_label'] if unit_docs else None, component_cache, 'unit', 'unit')
    unit_codes = get_component_value(unit_docs[0]['standard_code'] if unit_docs else None, component_cache, 'unit_code', 'unit')
    unit_concept_ids = get_component_value(unit_docs[0]['concept_id'] if unit_docs else None, component_cache, 'unit_concept_id', 'unit')

    values_labels, value_codes, values_ids  = format_categorical_values(values_docs, context)
    status_label, status_codes,status_ids = format_categorical_values(status_docs, status)
    if len(processed_docs) > 0 and query_text == str(processed_docs[0]['standard_label']):
        labels_parts = [query_text]
        mapping_codes = [str(processed_docs[0]['standard_code'])]
        concept_ids = [str(processed_docs[0]['concept_id'])]
        values_labels,value_codes, status_codes, status_label  = None,None,None,None
    else:
        labels_parts = [str(doc['standard_label']) for doc in processed_docs] if processed_docs else []
        mapping_codes = [str(doc['standard_code']) for doc in processed_docs] if processed_docs else []
        concept_ids = [str(doc['concept_id']) for doc in processed_docs] if processed_docs else []
        
    if values_labels and query_text in values_labels.split(";;"):
       values, codes, ids = values_labels.split(";;"), value_codes.split(';;'), values_ids.split(';;')
       index_ = values.index(query_text)
       combine_docs_label = values[index_]
       combine_mapping_codes = codes[index_]
       combine_mapping_ids = ids[index_]
       values_labels = ''
       value_codes = ''
       values_ids = ''
    else:
            combine_docs_label = join_or_single(labels_parts)
            combine_mapping_codes = join_or_single(mapping_codes)
            combine_mapping_ids = join_or_single(concept_ids)
    # print(f"created processed result for query={query_text}====Domain={domain}====Main Term={main_term}====Context={context}====Status={status}====Unit={unit}")
    return {
        'query_text': query_text,
        'revised_query': revise_query(query_text, main_term, context=context, status=status, unit=unit),
        'domain': domain,
        'standard_label': combine_docs_label,
        'standard_code': combine_mapping_codes,
        'standard_concept_id':combine_mapping_ids,
        'additional_context': values_labels,
        'additional_context_codes': value_codes,
        'additional_context_concept_ids': values_ids,
        'categorical_values': status_label,
        'categorical_codes': status_codes,
        'categorical_concept_ids': status_ids,
        'unit': unit_docs_label,
        'unit_code': unit_codes,
        'unit_concept_id': unit_concept_ids
    }

def create_processed_result_from_components(query_text, component_cache, query_final_result):
    def get_value(key, cache_key):
        return component_cache[cache_key][key] if component_cache.get(cache_key) and key in component_cache[cache_key] else query_final_result.get(key, '')
    return {
        'query_text': query_text,
        'revised_query': get_value('revised_query', 'context'),
        'domain': get_value('domain', 'context') or get_value('domain', 'status') or get_value('domain', 'unit') or get_value('domain', 'main_query'),
        'standard_label': get_value('standard_label', 'context'),
        'standard_code': get_value('standard_code', 'context'),
        'standard_concept_id': get_value('standard_concept_id', 'context'),
        'additional_context': get_value('additional_context', 'context'),
        'additional_context_codes': get_value('additional_context_codes', 'context'),
        'additional_context_concept_ids': get_value('additional_context_concept_ids', 'context'),
        'categorical_values': get_value('categorical_values', 'status'),
        'categorical_codes': get_value('categorical_codes', 'status'),
        'categorical_concept_ids': get_value('categorical_concept_ids', 'status'),
        'unit': get_value('unit', 'unit'),
        'unit_code': get_value('unit_code', 'unit'),
        'unit_concept_id': get_value('unit_concept_id', 'unit')
    }

import re

def rule_base_decomposition(text):
    # Normalize text
    normalized_text = normalize(text)
   
    # Regular expression to find various visit and month number patterns
    # Now includes 'at visit month' in the pattern
    pattern = re.compile(
        r'\bat\s*(visit|month)\s*(\d+)|\b(visit|month)(\d+)|\bat\s*visit\s*month\s*(\d+)',  # Added 'at visit month'
        re.IGNORECASE
    )
    
    # Search for matches in the normalized text
    match = pattern.search(normalized_text)
    id_pattern = re.compile(r'\bid\b', re.IGNORECASE)

    if match:
        # Extract the visit number from the matched groups
        # Find the first group that contains digits and is not None
        visit_number = next((g for g in match.groups() if g and g.isdigit()), None)
        
        # Log the found visit or month number
        logging.info(f"Found visit/month number: {visit_number} in text: {text}")
        
        # Replace the detected pattern with "Follow-up month {visit_number}"
        normalized_text = re.sub(pattern, f'Follow-up month {visit_number}', normalized_text)
    
    # Check for 'id' and replace with 'identifier'
    if id_pattern.search(normalized_text):
        logging.info("Found 'id' in text, replacing with 'identifier'.")
        normalized_text = re.sub(id_pattern, 'identifier', normalized_text)

    return normalized_text


def format_categorical_values(status_docs, status):
    labels = []
    codes = []
    ids = []
    print(f"values: {status}\nDocs: {status_docs}")
    
    if status_docs is None:
        return '', '', ''
    
    # Normalize the keys of status_docs to lower case for case-insensitive matching
    normalized_status_docs = {k.lower(): v for k, v in status_docs.items()}
    
    for v_ in status:
        if  v_: normalized_v_ = v_.lower()
        else: normalized_v_ = v_
        if normalized_v_ in normalized_status_docs:
            docs = normalized_status_docs[normalized_v_]
            if len(docs) > 1:
                labels.append(' and/or '.join(remove_duplicates([str(doc['standard_label']) for doc in docs])))
                codes.append(' and/or '.join(remove_duplicates([str(doc['standard_code']) for doc in docs])))
                ids.append(' and/or '.join(remove_duplicates([str(doc['concept_id']) for doc in docs])))
            elif len(docs) == 1:
                labels.append(str(docs[0]['standard_label']))
                codes.append(str(docs[0]['standard_code']))
                ids.append(str(docs[0]['concept_id']))
    
    return ';;'.join(labels), ';;'.join(codes) , ';;'.join(ids)



def parse_query_text(query_text):
    # Initialize default values for query, status, and unit
    query = None
    status = None
    unit = None
    
    # Split the query_text by commas and process each component
    components = query_text.split(',')
    for component in components:
        # Strip whitespace and identify parts by checking the prefix or format
        part = component.strip()
        if "status:" in part:
            status = part.split("status:")[1].strip()
        elif "unit:" in part:
            unit = part.split("unit:")[1].strip()
        else:
            # Assume any part without 'status:' or 'unit:' is part of the main query
            query = part if query is None else query + ', ' + part  # This handles multiple query components
    print(f"Query={query}====Status={status}====Unit={unit}")
    return query, status, unit

def process_synonyms(synonyms_text: str) -> List[str]:
        """
        Processes the synonyms text to split by ';;' if it exists, otherwise returns the whole text as a single item list.
        If synonyms_text is empty, returns an empty list.
        """
        if synonyms_text:
            if ';;' in synonyms_text:
                return synonyms_text.split(';;')
            else:
                return [synonyms_text]
        return []
def parse_tokens(text: str, semantic_type: bool = False, domain: str = None) -> Tuple[str, List[str], str]:
    """Extract entity and synonyms from the formatted text, if present."""
    if text is None:
        return None, [], None
    text_ = text.strip().lower()
    print(f"text={text_}")
    # Initialize default values
    entity = None
    synonyms = []
    semantic_type_of_entity = None
    
    # Check if the text follows the new format
    if 'concept name:' in text_ and 'synonyms:' in text_ and 'domain:' in text_ and 'concept class:' in text_ and 'vocabulary:' in text_:
        parts = text_.split(', ')
        concept_dict = {}
        for part in parts:
            if ':' in part:
                key, value = part.split(':',1)
                concept_dict[key.strip()] = value.strip()
            else:
                concept_dict['concept name'] = part.strip()
        entity = concept_dict.get('concept name', None)
        synonyms_str = concept_dict.get('synonyms', '')
        synonyms = process_synonyms(synonyms_str)
        if semantic_type:
            if domain:
                semantic_type_of_entity = concept_dict.get('domain', '')
                entity = f"{entity}||{semantic_type_of_entity}"
        logger.info(f"Entity: {entity}, Synonyms: {synonyms}, Semantic Type of Entity: {semantic_type_of_entity}")
    else:
        # Fallback to the old format handling
        if '<ent>' in text_ and '<syn>' in text_:
            entity = text_.split('<ent>')[1].split('</ent>')[0]
            synonyms_text = text_.split('<syn>')[1].split('</syn>')[0]
            synonyms = process_synonyms(synonyms_text)
            logger.info(f"Entity: {entity}, Synonyms: {synonyms}")
        else:
            entity = text_
            synonyms = []
        # if "||" in entity: 
        #     entity_type_info = entity.split("||") 
        #     entity = ' '.join(entity_type_info[:-1])  # All but last part
        #     if semantic_type:
        #         if domain:
        #             semantic_type_of_entity = entity_type_info[-1].replace(':',' ').strip()
        #         else:
        #             semantic_type_of_entity = entity_type_info[-1].split(':')[-1].strip()
        #         logger.info(f"Entity: {entity}, Semantic Type of Entity: {semantic_type_of_entity}")

    return entity, synonyms, None
    
def combine_ent_synonyms(text: str,semantic_type = False,domain=None) -> str:
        """Extract entity and synonyms from the formatted text, if present."""
        text_ = text.strip().lower()
        if '<desc>' in text_:
            description = text_.split('<desc>')[1].split('</desc>')[0]
            logger.info(f"Description: {description}")
            text = text_.split('<desc>')[0]
        return text_
    
    
    
def combined_profiler(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_memory = process.memory_info().rss
        start_cpu = process.cpu_times()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = process.memory_info().rss
        end_cpu = process.cpu_times()

        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds")
        print(f"Memory usage increased by {((end_memory - start_memory) / (1024 ** 2)):.4f} MB")
        print(f"User CPU time increased by {end_cpu.user - start_cpu.user:.4f} seconds")
        print(f"System CPU time increased by {end_cpu.system - start_cpu.system:.4f} seconds")

        return result
    return wrapper


def print_docs(docs):
    for res, i in zip(docs, range(len(docs))):
        print("----" + str(i + 1) + "----")
        # print("LABEL: " + res.metadata["label"])
        print(f"{res.metadata['label']}:{res.metadata['domain']}")

from collections import defaultdict 
T = TypeVar("T")
H = TypeVar("H", bound=Hashable)      
def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    """Yield unique elements of an iterable based on a key function.

    Args:
        iterable: The iterable to filter.
        key: A function that returns a hashable key for each element.

    Yields:
        Unique elements of the iterable based on the key function.
    """
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e


def weighted_reciprocal_rank(
        doc_lists: List[List[Document]]
    ) -> List[Document]:
        """
        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        """
        weights = [0.5,0.5]
        id_key = None
        c = 60
        print(f"weighted Reciprocal Rank Fusion\n:{doc_lists}")
        if len(doc_lists) != len(weights):
            raise ValueError(
                "Number of rank lists must be equal to the number of weights."
            )

        # Associate each doc's content with its RRF score for later sorting by it
        # Duplicated contents across retrievers are collapsed & scored cumulatively
        rrf_score: Dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[
                    doc.page_content
                    if id_key is None
                    else doc.metadata[id_key]
                ] += weight / (rank + c)

        # Docs are deduplicated by their contents then sorted by their scores
        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(
            unique_by_key(
                all_docs,
                lambda doc: doc.page_content
                if id_key is None
                else doc.metadata[id_key],
            ),
            reverse=True,
            key=lambda doc: rrf_score[
                doc.page_content if id_key is None else doc.metadata[id_key]
            ],
        )
        print(f"total top docs={len(sorted_docs)}")
        return sorted_docs
    
def filter_irrelevant_domain_candidates(docs, domain):
    select_vocabs = select_vocabulary(domain=domain)
    docs_ =  [doc for doc in docs if doc.metadata['vocab'] in select_vocabs]
    # print(f"Original List ={len(docs)} vs Filtered Docs={len(docs_)}")
    return docs_



def filter_documents(documents):
    """
    Filters documents based on the condition that if a document has a label 
    where other documents with the same label have 'is_standard' as 's' or 'c',
    the document with 'is_standard' as NaN is discarded.
    
    Args:
    - documents (list): A list of LangChain Document objects where metadata includes 'label' and 'is_standard'.

    Returns:
    - list: A filtered list of LangChain Document objects.
    """

    # Step 1: Group documents by label
    label_groups = {}
    for doc in documents:
        label = doc.metadata['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(doc)

    # Step 2: Filter out documents where `is_standard` is nan and there is another document with 's' or 'c'
    filtered_documents = []
    for label, docs in label_groups.items():
        has_standard = any(doc.metadata['is_standard'] in ['s', 'c'] for doc in docs)
        
        # If there is a 's' or 'c', discard documents with `is_standard` as NaN
        for doc in docs:
            is_standard = doc.metadata.get('is_standard')
            if not (is_standard is None or (is_standard != 's' and is_standard != 'c')):
                filtered_documents.append(doc)
            elif not (has_standard and pd.isna(is_standard)):
                filtered_documents.append(doc)

    return filtered_documents

def estimate_token_cost(text,filename):
    # Get the encoder for the GPT-4 model
    enc = tiktoken.encoding_for_model("gpt-4")
    
    # Encode the text and count the tokens
    n_tokens = len(enc.encode(text))
    
    # Save the token count to a text file
    with open(filename, 'w') as f:
        f.write(f"Token count: {n_tokens}")
    
    print(f"Token count saved to {filename}")