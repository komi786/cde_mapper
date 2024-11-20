from langchain.schema import Document
import json
import logging
from json_repair import repair_json
from typing import List, Tuple, Dict, Iterable
import os
from .param import LOG_FILE, MAPPING_FILE
from tqdm import tqdm
import pandas as pd
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher
from .py_model import RetrieverResultsModel, ProcessedResultsModel
from itertools import chain
import csv
import tiktoken
import numpy as np
from collections import defaultdict
import re


def remove_punctuation(text):
    # use regex to remove punctuation
    text = text.strip().lower()
    return re.sub(r"[^\w\s]", "", text)


STOP_WORDS = [
    "stop",
    "start",
    "combinations",
    "combination",
    "various combinations",
    "various",
    "left",
    "right",
    "blood",
    "finding",
    "finding status",
    "status",
    "extra",
    "point in time",
    "pnt",
    "oral",
    "product",
    "oral product",
    "several",
    "types",
    "several types",
    "random",
    "nominal",
    "p time",
    "quant",
    "qual",
    "quantitative",
    "qualitative",
    "ql",
    "qn",
    "quan",
    "anti",
    "antibodies",
    "wb",
    "whole blood",
    "serum",
    "plasma",
    "diseases",
    "disorders",
    "disorder",
    "disease",
    "lab test",
    "measurements",
    "lab tests",
    "meas value",
    "measurement",
    "procedure",
    "procedures",
    "panel",
    "ordinal",
    "after",
    "before",
    "survey",
    "level",
    "levels",
    "others",
    "other",
    "p dose",
    "dose",
    "dosage",
    "frequency",
    "calc",
    "calculation",
    "calculation method",
    "method",
    "calc method",
    "calculation methods",
    "methods",
    "calc methods",
    "calculation method",
    "calculation methods",
    "measurement",
    "measurements",
    "meas value",
    "meas values",
    "meas",
    "meas val",
    "meas vals",
    "meas value",
    "meas values",
    "meas",
    "meas val",
    "vals",
    "val",
]
DEFAULT_QUALIFIER_VALUES = [
    "yes",
    "no",
    "not available",
    "unknown",
    "missing",
    "1",
    "0",
    "no",
    "not",
    "once a day",
    "once a week",
    "once a month",
    "once a year",
    "once daily",
    "twice daily",
    "three times daily",
    "twice a day",
    "three times a day",
    "more than three times per day",
    "3-4 times a day",
    "1-2 times a day",
    "1 time per day",
    "2 times per day",
    "3 times per day",
    "1 time",
    "2 times",
    "2 times per week",
    "2 times a week",
    "3 times",
    "3 times per week",
    "3 times a week",
    "never",
    "Three or more times a day",
    "none",
    "more than once a day",
    "once a day",
    "2-3 times a week",
    "5 or more times a week",
    "5 or more times per day",
    "3 or more times per day",
    "3 or more times a week",
    "2 or more times per day",
    "4 or more times per day" "2 more times a week",
    "on a daily basis",
    "1 or more times per day",
    "1 or more times a week",
    "frequently",
    "sometimes",
]


def save_jsonl(data, file):
    with open(file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print("Data saved to file.")


def save_docs_to_jsonl(array: Iterable[Document], file_path: str) -> None:
    with open(file_path, "w") as jsonl_file:
        for doc in array:
            # print(doc.json())
            jsonl_file.write(doc.model_dump_json() + "\n")


def convert_to_document(data):
    try:
        page_content = data.get("kwargs", {}).get("page_content", {})
        print(f"page_content={page_content}")
        metadata = data.get("kwargs", {}).get("metadata", {})

        # Create the Document object
        document = Document(page_content=page_content, metadata=metadata)
        return document

    except Exception as e:
        print(f"Error loading document: {e}")
        return None


def load_custom_docs_from_jsonl(file_path) -> list:
    docs = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            try:
                obj = Document(**data)
            except Exception as e:
                print(f"{e}: document object translated into Dictionary format")
                obj = convert_to_document(data)
            docs.append(obj)

    print(f"Total Custom Documents: {len(docs)}")
    return docs


def load_docs_from_jsonl(file_path) -> list:
    docs_dict = {}
    count = 0
    with open(file_path, "r") as jsonl_file:
        print("Opening file...")
        for line in tqdm(jsonl_file, desc="Loading Documents"):
            # if count >= 100:
            #     break
            data = json.loads(line)
            try:
                obj = Document(**data)
            except Exception as e:
                print(f"{e}: document object translated into Dictionary format")
                obj = convert_to_document(data)
            # print(f"data={obj}")
            if "vocab" in obj.metadata:
                vocab = obj.metadata["vocab"].lower()

                if vocab in [
                    "atc",
                    "loinc",
                    "ucum",
                    "rxnorm",
                    "omop extension",
                    "mesh",
                    "meddra",
                    "cancer modifier",
                    "snomed",
                    "rxnorm extension",
                ]:
                    key = (obj.page_content, json.dumps(obj.metadata, sort_keys=True))
                    if key not in docs_dict:
                        docs_dict[key] = obj
                        count += 1
            else:
                key = (obj.page_content, json.dumps(obj.metadata, sort_keys=True))
                if key not in docs_dict:
                    docs_dict[key] = obj
                    count += 1

    # Convert dictionary values to a sorted list to process documents in a specific order

    sorted_docs = (
        sorted(docs_dict.values(), key=lambda doc: doc.metadata["vocab"].lower())
        if "vocab" in docs_dict.values()
        else sorted(docs_dict.values(), key=lambda doc: doc.metadata["label"].lower())
    )
    print(f"Total Unique Documents: {len(sorted_docs)}\n")
    return sorted_docs


def save_json_data(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {file_path}")


def init_logger(log_file_path=LOG_FILE) -> logging.Logger:
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG
    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Set the logging level for the file handler

    # Create a stream handler (to print to console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(
        logging.INFO
    )  # Set the logging level for the stream handler

    # Define the format for log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


global_logger = init_logger()


def save_txt_file(file_path, data) -> None:
    with open(file_path, "a") as file:
        for item in data:
            file.write(f"{item}\n")
    print(f"Total Data = {len(data)} saved to file.")


def save_documents(filepath: str, docs) -> None:
    import pickle

    """Save the BM25 documents to a file."""
    with open(filepath, "wb") as f:
        pickle.dump(docs, f)


def load_documents(filepath: str) -> List[Document]:
    import pickle

    """Load the BM25 documents from a file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


# def save_chat_history(file_path: str, history: List[HumanMessage]) -> None:
#     with open(file_path, 'wb') as f:
#         pickle.dump(history, f)

# Function to load chat history from a file
# def load_chat_history(file_path: str) -> List[HumanMessage]:
#     if os.path.exists(file_path):
#         with open(file_path, 'rb') as f:
#             return pickle.load(f)
#     return []

VOCAB_CACHE = {}


def load_vocabulary(file_path=MAPPING_FILE) -> dict:
    with open(file_path, "r") as file:
        config = json.load(file)
    return config["vocabulary_rules"]


# def select_vocabulary(query_text=None, config_path=MAPPING_FILE, domain=None):
#     global VOCAB_CACHE
#     # Normalize the domain name to lower case or set to 'unknown' if not provided
#     domain = domain.lower() if domain else 'all'

#     # Check if the vocabulary for the domain is alRETRIEVER_CACHEready loaded
#     if domain in VOCAB_CACHE:
#         selected_vocab = VOCAB_CACHE[domain]
#     else:
#         # Load the configuration file if the domain's vocabulary isn't cached
#         vocabulary_rules = load_vocabulary(config_path)

#         # Get domain-specific vocabulary or default to 'unknown' if not found
#         selected_vocab = vocabulary_rules['domains'].get(domain, vocabulary_rules['domains']['all'])

#         # Cache the selected vocabulary
#         VOCAB_CACHE[domain] = selected_vocab

#     return selected_vocab

# def select_vocabulary(query_text=None, config_path=MAPPING_FILE, domain=None):
#     global VOCAB_CACHE
#     domain = domain.lower() if domain else 'all'

#     if domain in VOCAB_CACHE:
#         selected_vocab = VOCAB_CACHE[domain]
#     else:
#         vocabulary_rules = load_vocabulary(config_path)
#         selected_vocab = tuple(vocabulary_rules['domains'].get(domain, vocabulary_rules['domains']['all']))
#         VOCAB_CACHE[domain] = selected_vocab

#     return selected_vocab


def post_process_candidates(candidates: List[Document], max=1):
    processed_candidates = []
    # seen = set()
    # print(f"Total Candidates={len(candidates)}")
    if not candidates or len(candidates) == 0:
        print("No candidates found.")
        return []

    for _, doc in enumerate(candidates[:max]):
        label = doc.metadata.get("label", "none")
        label = f'"{label}"' if "|" in label else label
        current_doc_dict = {
            "standard_label": label,
            "domain": f"{doc.metadata['domain']}",
            "concept_class": f"{doc.metadata['concept_class']}",
            "standard_code": f"{doc.metadata['vocab']}:{doc.metadata['scode']}",
            "standard_omop_id": str(doc.metadata["sid"]),
            "vocab": doc.metadata["vocab"],
        }
        doc_obj = RetrieverResultsModel(**current_doc_dict)
        if doc_obj not in processed_candidates:
            processed_candidates.append(doc_obj)

    return processed_candidates


def save_to_csv(data, filename):
    if not data:
        return

    fieldnames = [
        "VARIABLE NAME",
        "VARIABLE LABEL",
        "Domain",
        "Variable Concept Label",
        "Variable Concept Code",
        "Variable Concept OMOP ID",
        "Additional Context Concept Label",
        "Additional Context Concept Code",
        "Additional Context OMOP ID",
        "Primary to Secondary Context Relationship",
        "Categorical Values Concept Label",
        "Categorical Values Concept Code",
        "Categorical Values Concept OMOP ID",
        "UNIT",
        "Unit Concept Label",
        "Unit Concept Code",
        "Unit OMOP ID",
        "Reasoning",
        "Prediction",
    ]

    # Map and combine fields in the data rows
    def map_and_combine_fields(row):
        # Map fields
        mapped_row = {
            "VARIABLE NAME": row.get("VARIABLE NAME", ""),
            "VARIABLE LABEL": row.get("VARIABLE LABEL", ""),
            "Variable Concept Label": row.get("Variable Concept Label", ""),
            "Variable Concept OMOP ID": row.get("Variable Concept OMOP ID", ""),
            "Variable Concept Code": row.get("Variable Concept Code", ""),
            "Domain": row.get("Domain", ""),
            "Additional Context Concept Label": row.get(
                "Additional Context Concept Label", ""
            ),
            "Additional Context Concept Code": row.get(
                "Additional Context Concept Code", ""
            ),
            "Additional Context OMOP ID": row.get("Additional Context OMOP ID", ""),
            "Primary to Secondary Context Relationship": row.get(
                "Primary to Secondary Context Relationship", ""
            ),
            "Categorical Values Concept Label": row.get(
                "Categorical Values Concept Label", ""
            ),
            "Categorical Values Concept Code": row.get(
                "Categorical Values Concept Code", ""
            ),
            "Categorical Values Concept OMOP ID": row.get(
                "Categorical Values Concept OMOP ID", ""
            ),
            "Unit Concept Label": row.get("Unit Concept Label", ""),
            "Unit Concept Code": row.get("Unit Concept Code", ""),
            "Unit OMOP ID": row.get("Unit OMOP ID", ""),
            "Reasoning": row.get("reasoning", ""),
            "Prediction": row.get("prediction", ""),
        }
        # Combine fields
        # label_ids = '|'.join(filter(None, [row.get('standard_concept_id'), row.get('additional_context_omop_ids')]))
        # label_codes = '|'.join(filter(None, [row.get('standard_code'), row.get('additional_context_codes')]))
        # mapped_row['Variable Concept OMOP ID'] = label_ids
        # mapped_row['Label Concept CODE'] = label_codes
        return mapped_row

    # if file already exists don't readd the header
    if os.path.exists(filename):
        mode = "a"
    else:
        mode = "w"
    with open(filename, mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for row in data:
            writer.writerow(map_and_combine_fields(row))


def load_mapping(filename, domain):
    # print(f"domain={domain}")
    try:
        with open(filename, "r") as file:
            data = json.load(file)

        domain = domain if domain else "all"
        # print(f"domain={domain}")
        mapping = data["mapping_rules"].get(domain, {})

        # Get examples or default to empty list if not present
        relevance_examples = data.get("rel_relevance_examples", {}).get(domain, [])
        ranking_examples = data.get("ranking_examples", {}).get(domain, [])
        # print(f"ranking_examples={ranking_examples[:2]} for domain={domain}")
        # Format examples as string representations of dictionaries
        relevance_examples_string = [
            {
                "input": ex["input"],
                "output": str(
                    [
                        f"{{'answer': '{out['answer']}', 'relationship': '{out['relationship']}', 'explanation': '{out['explanation']}'}}"
                        for out in ex["output"]
                    ]
                ),
            }
            for ex in relevance_examples
        ]

        ranking_examples_string = [
            {
                "input": ex["input"],
                "output": str(
                    [
                        f"{{'answer': '{out['answer']}', 'score': '{out['score']}', 'explanation': '{out['explanation']}'}}"
                        for out in ex["output"]
                    ]
                ),
            }
            for ex in ranking_examples
        ]
        # print(f"ranking_examples_string={ranking_examples_string[:2]}")

        if not mapping:
            return None, ranking_examples_string, relevance_examples_string

        return (
            {
                "prompt": mapping.get("description", "No description provided."),
                "examples": mapping.get("example_output", []),
            },
            ranking_examples_string,
            relevance_examples_string,
        )

    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None, None, None
    except json.JSONDecodeError:
        print("JSON decoding error.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None


# def parse_term(extracted_terms, domain):
#     domain = domain.lower() if domain else 'all'
#     if domain in extracted_terms.keys():
#         # term = extracted_terms[domain]
#         if domain == 'condition':
#             if 'procedure' in extracted_terms:
#                 procedure = extracted_terms['procedure']


def save_result_to_jsonl(array: Iterable[dict], file_path: str) -> None:
    print(f"Saving to file: {file_path}")
    with open(file_path, "w") as jsonl_file:
        for doc in array:
            json_string = json.dumps(doc)
            print(json_string)
            jsonl_file.write(json_string + "\n")
    print(f"Saved {len(array)} documents to {file_path}")


# def exact_match_found(query_text, documents, domain=None):
#     # print(f"documents={documents}")
#     if not query_text or not documents:
#         # print("NO DOCUMENTS FOUND FOR QUERY={query_text}")
#         return []
#     for doc in documents:
#         if 'score' in doc.metadata:
#             if doc.metadata['score'] >= 0.95:
#                 # print(f"EXACT MATCH FOUND FOR QUERY using Score={query_text}")
#                 return [doc]

#     # Create a database and populate it
#     label_to_docs = {}
#     db = DictDatabase(CharacterNgramFeatureExtractor(2))
#     for doc in documents:
#         label = doc.metadata.get('label', None)
#         if label:
#             db.add(doc.metadata['label'])
#             label_key = label.strip().lower()
#             if label_key not in label_to_docs:
#                 label_to_docs[label_key] = []
#             label_to_docs[label_key].append(doc)


#     # Create a searcher with cosine similarity
#     searcher = Searcher(db, CosineMeasure())

#     # Normalize query text
#     results = searcher.search(query_text, 0.9)  # Set threshold to 0.95 for high similarity

#     matched_docs = []
#     selected_vocab = list(select_vocabulary(query_text, domain=domain))
#     matched_docs = [
#         doc
#         for result in results
#         if result in label_to_docs
#         for doc in label_to_docs[result]
#         if doc.metadata.get('vocab') in selected_vocab
#     ]

#     if len(matched_docs) > 1:

#         if query_text in DEFAULT_QUALIFIER_VALUES:
#                 selected_vocab = ['loinc'] + selected_vocab
#                 matched_docs = sorted(matched_docs, key=lambda x: selected_vocab.index(x.metadata['vocab']))
#         else:
#             domain_ = set(list({doc.metadata['domain'] for doc in matched_docs}))
#             unique_domain = len(domain_) == 1
#             # match_docs_vocab =select_vocabulary(query_text, domain=domain_)
#             print(f"is domain unique :{unique_domain}")
#             if unique_domain:
#                 domain_ = domain_.pop()
#                 match_docs_vocab = list(select_vocabulary(query_text, domain=domain_))
#                 match_docs_vocab += selected_vocab
#                 print(f"selected_vocab for domain={selected_vocab}.. matching docs vocab={match_docs_vocab}")
#                 first_priority_vocab = match_docs_vocab[0]
#                 matched_docs = sorted(matched_docs, key=lambda x: (x.metadata['vocab'] != first_priority_vocab, match_docs_vocab.index(x.metadata['vocab'])))
#             else:
#                 matched_docs = sorted(matched_docs, key=lambda x: selected_vocab.index(x.metadata['vocab']))
#         print(f"Exact match candidates")
#         pretty_print_docs(matched_docs)
#     return matched_docs


def convert_db_result(result) -> RetrieverResultsModel:
    #  dict must have four elemenets according to sql query result if a component of query is found as subset of variable standard_omop_id
    # check total items in result
    if len(result) > 6:
        resut_dict = {
            "standard_label": str(result[3]),
            "standard_code": str(result[4]),
            "standard_omop_id": str(result[5]),
            "score": 10.0,
            "domain": "",
            "vocab": "",
        }
    else:
        resut_dict = {
            "standard_label": str(result[1]),
            "standard_code": str(result[2]),
            "standard_omop_id": str(result[3]),
            "score": 10.0,
            "domain": "",
            "vocab": "",
        }
    return [RetrieverResultsModel(**resut_dict)]


def exact_match_found_no_vocab(query_text, documents, domain=None):
    if not query_text or not documents:
        return []
    query_text = query_text.lower()
    # Check if there is a high score match

    # rerank the documents based on the score
    documents = (
        sorted(documents, key=lambda x: x.metadata.get("score", 0), reverse=True)
        if "score" in documents[0].metadata
        else documents
    )
    # check if any docs has score more than 0.95 return all those in ranked order
    matched_docs = [doc for doc in documents if doc.metadata.get("score", 0) >= 0.95]
    if len(matched_docs) > 1:
        len_matches = len(matched_docs)
        # append the remaining docs to the end
        matched_docs += documents
        return matched_docs, len_matches
    # check for all docs where doc.metadata['label'] == query_text
    matched_docs = [
        doc
        for doc in documents
        if remove_punctuation(doc.metadata.get("label", ""))
        == remove_punctuation(query_text)
    ]
    if len(matched_docs) > 1:
        len_matches = len(matched_docs)
        matched_docs += documents
        return matched_docs, len_matches

    # Create a database and populate it
    label_to_docs = {}
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    for doc in documents:
        label = doc.metadata.get("label", None)
        if label:
            db.add(doc.metadata["label"])
            label_key = label.strip().lower()
            if label_key not in label_to_docs:
                label_to_docs[label_key] = []
            label_to_docs[label_key].append(doc)

    # Create a searcher with cosine similarity
    searcher = Searcher(db, CosineMeasure())

    # Normalize query text
    results = searcher.search(query_text, 0.9)
    if len(results) > 0:
        matched_docs = [
            doc
            for result in results
            if result in label_to_docs
            for doc in label_to_docs[result]
        ]
        len_matches = len(matched_docs)
        matched_docs += documents
        return matched_docs, len_matches
    return matched_docs, len(matched_docs) if len(matched_docs) > 0 else 0


# def sim_string_search(query, candidates):


def exact_match_found(query_text, documents, domain=None):
    if not query_text or not documents:
        return []
    query_text = query_text.lower()
    # Check if there is a high score match
    for doc in documents:
        if "score" in doc.metadata and doc.metadata["score"] >= 0.95:
            return [doc]

    # Create a database and populate it
    label_to_docs = {}
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    for doc in documents:
        label = doc.metadata.get("label", None)
        if label:
            db.add(doc.metadata["label"])
            label_key = label.strip().lower()
            if label_key not in label_to_docs:
                label_to_docs[label_key] = []
            label_to_docs[label_key].append(doc)

    # Create a searcher with cosine similarity
    searcher = Searcher(db, CosineMeasure())

    # Normalize query text
    results = searcher.search(query_text, 0.9)

    # Select vocabulary
    selected_vocab = select_vocabulary(query_text, domain=domain)

    # Match documents with the selected vocabulary
    matched_docs = [
        doc
        for result in results
        if result in label_to_docs
        for doc in label_to_docs[result]
        if doc.metadata.get("vocab") in selected_vocab
    ]

    if len(matched_docs) > 1:
        if query_text in DEFAULT_QUALIFIER_VALUES:
            # Use itertools.chain to concatenate 'loinc' and selected_vocab without modifying the original list
            combined_vocab = list(chain(["loinc"], selected_vocab))
            matched_docs = sorted(
                matched_docs, key=lambda x: combined_vocab.index(x.metadata["vocab"])
            )
        else:
            domain_ = set({doc.metadata["domain"] for doc in matched_docs})
            unique_domain = len(domain_) == 1

            if unique_domain:
                domain_ = domain_.pop()
                match_docs_vocab = select_vocabulary(query_text, domain=domain_)
                print(
                    f"selected_vocab for domain={selected_vocab}.. matching docs vocab={match_docs_vocab}"
                )
                # Use itertools.chain to concatenate match_docs_vocab and selected_vocab
                combined_vocab = list(chain(match_docs_vocab, selected_vocab))

                first_priority_vocab = combined_vocab[0]
                matched_docs = sorted(
                    matched_docs,
                    key=lambda x: (
                        x.metadata["vocab"] != first_priority_vocab,
                        combined_vocab.index(x.metadata["vocab"]),
                    ),
                )
            else:
                matched_docs = sorted(
                    matched_docs,
                    key=lambda x: selected_vocab.index(x.metadata["vocab"]),
                )

        print("Exact match candidates")
        pretty_print_docs(matched_docs)

    return matched_docs


def select_vocabulary(query_text=None, config_path=MAPPING_FILE, domain=None):
    global VOCAB_CACHE
    domain = domain.lower() if domain else "all"

    if domain in VOCAB_CACHE:
        selected_vocab = VOCAB_CACHE[domain]
    else:
        vocabulary_rules = load_vocabulary(config_path)
        # Store the vocabulary as a tuple for immutability
        selected_vocab = tuple(
            vocabulary_rules["domains"].get(domain, vocabulary_rules["domains"]["all"])
        )
        VOCAB_CACHE[domain] = selected_vocab

    return selected_vocab


def exact_match_wo_vocab(query_text, documents, domain=None):
    if not query_text or not documents:
        # print("NO DOCUMENTS FOUND FOR QUERY={query_text}")
        return []

    # Create a database and populate it
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    for doc in documents:
        if "label" in doc.metadata:
            db.add(doc.metadata["label"])

    # Create a searcher with cosine similarity
    searcher = Searcher(db, CosineMeasure())

    # Normalize query text
    results = searcher.search(
        query_text, 0.95
    )  # Set threshold to 0.95 for high similarity

    matched_docs = []
    for result in results:
        for doc in documents:
            if doc.metadata["label"] == result:
                print(f"EXACT MATCH FOUND FOR QUERY={query_text}")
                matched_docs.append(doc)
    return matched_docs[:1]


def create_document_string(doc):
    # return f"{doc.metadata.get('label', 'none')}. The parent term of this concept is {doc.metadata.get('parent_term', 'none')}"
    label = f"{doc.metadata.get('label', 'none')}"
    # if doc.metadata.get('parent_term', None):
    #     label += f", {doc.metadata.get('parent_term', 'none')}"
    return label


def fix_json_quotes(json_like_string):
    try:
        # Trying to convert it directly
        return repair_json(json_like_string, return_objects=True)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON after trying to fix quotes, error: {str(e)}")
        return None

    """
source code from : "https://github.com/facebookresearch/contriever/blob/main/src/normalize_text.py"

"""


# #: Control characters.
# CONTROLS = {
#     "\u0001",
#     "\u0002",
#     "\u0003",
#     "\u0004",
#     "\u0005",
#     "\u0006",
#     "\u0007",
#     "\u0008",
#     "\u000e",
#     "\u000f",
#     "\u0011",
#     "\u0012",
#     "\u0013",
#     "\u0014",
#     "\u0015",
#     "\u0016",
#     "\u0017",
#     "\u0018",
#     "\u0019",
#     "\u001a",
#     "\u001b",
# }
# # There are further control characters, but they are instead replaced with a space by unicode normalization
# # '\u0009', '\u000a', '\u000b', '\u000c', '\u000d', '\u001c',  '\u001d', '\u001e', '\u001f'


# #: Hyphen and dash characters.
# HYPHENS = {
#     "-",  # \u002d Hyphen-minus
#     "‐",  # \u2010 Hyphen
#     "‑",  # \u2011 Non-breaking hyphen
#     "⁃",  # \u2043 Hyphen bullet
#     "‒",  # \u2012 figure dash
#     "–",  # \u2013 en dash
#     "—",  # \u2014 em dash
#     "―",  # \u2015 horizontal bar
# }

# #: Minus characters.
# MINUSES = {
#     "-",  # \u002d Hyphen-minus
#     "−",  # \u2212 Minus
#     "－",  # \uff0d Full-width Hyphen-minus
#     "⁻",  # \u207b Superscript minus
# }

# #: Plus characters.
# PLUSES = {
#     "+",  # \u002b Plus
#     "＋",  # \uff0b Full-width Plus
#     "⁺",  # \u207a Superscript plus
# }

# #: Slash characters.
# SLASHES = {
#     "/",  # \u002f Solidus
#     "⁄",  # \u2044 Fraction slash
#     "∕",  # \u2215 Division slash
# }

# #: Tilde characters.
# TILDES = {
#     "~",  # \u007e Tilde
#     "˜",  # \u02dc Small tilde
#     "⁓",  # \u2053 Swung dash
#     "∼",  # \u223c Tilde operator #in mbert vocab
#     "∽",  # \u223d Reversed tilde
#     "∿",  # \u223f Sine wave
#     "〜",  # \u301c Wave dash #in mbert vocab
#     "～",  # \uff5e Full-width tilde #in mbert vocab
# }

# #: Apostrophe characters.
# APOSTROPHES = {
#     "'",  # \u0027
#     "’",  # \u2019
#     "՚",  # \u055a
#     "Ꞌ",  # \ua78b
#     "ꞌ",  # \ua78c
#     "＇",  # \uff07
# }

# #: Single quote characters.
# SINGLE_QUOTES = {
#     "'",  # \u0027
#     "‘",  # \u2018
#     "’",  # \u2019
#     "‚",  # \u201a
#     "‛",  # \u201b
# }

# #: Double quote characters.
# DOUBLE_QUOTES = {
#     '"',  # \u0022
#     "“",  # \u201c
#     "”",  # \u201d
#     "„",  # \u201e
#     "‟",  # \u201f
# }

# #: Accent characters.
# ACCENTS = {
#     "`",  # \u0060
#     "´",  # \u00b4
# }

# #: Prime characters.
# PRIMES = {
#     "′",  # \u2032
#     "″",  # \u2033
#     "‴",  # \u2034
#     "‵",  # \u2035
#     "‶",  # \u2036
#     "‷",  # \u2037
#     "⁗",  # \u2057
# }

# #: Quote characters, including apostrophes, single quotes, double quotes, accents and primes.
# QUOTES = APOSTROPHES | SINGLE_QUOTES | DOUBLE_QUOTES | ACCENTS | PRIMES


# def normalize(text):
#     if text is None:
#         return None
#     text = str(text)
#     # Replace control characters
#     for control in CONTROLS:
#         text = text.replace(control, "")
#     text = text.replace("\u000b", " ").replace("\u000c", " ").replace("\u0085", " ")

#     # Replace hyphens and minuses with '-'
#     for hyphen in HYPHENS | MINUSES:
#         text = text.replace(hyphen, "-")
#     text = text.replace("\u00ad", "")

#     # Replace various quotes with standard quotes
#     for double_quote in DOUBLE_QUOTES:
#         text = text.replace(double_quote, '"')
#     for single_quote in SINGLE_QUOTES | APOSTROPHES | ACCENTS:
#         text = text.replace(single_quote, "'")
#     text = text.replace("′", "'")  # \u2032 prime
#     text = text.replace("‵", "'")  # \u2035 reversed prime
#     text = text.replace("″", "''")  # \u2033 double prime
#     text = text.replace("‶", "''")  # \u2036 reversed double prime
#     text = text.replace("‴", "'''")  # \u2034 triple prime
#     text = text.replace("‷", "'''")  # \u2037 reversed triple prime
#     text = text.replace("⁗", "''''")  # \u2057 quadruple prime
#     text = text.replace("…", "...").replace(" . . . ", " ... ")  # \u2026

#     # Replace slashes with '/'
#     for slash in SLASHES:
#         text = text.replace(slash, "/")

#     # Ensure there's only one space between words
#     text = " ".join(text.split())
#     text = text.lower()
#     if text in ["null", "nil", "none", "n/a", "", "nan"]:
#         return None
#     return text


def normalize_page_content(page_content):
    if page_content is None:
        return page_content
    page_content = page_content.strip().lower()

    if "<ent>" in page_content:
        page_content = page_content.split("<ent>")[1].split("</ent>")[0]
        if "||" in page_content:
            page_content = page_content.split("||")[0]
        if "." in page_content:
            page_content = page_content.split(".")[0]
    elif "||" in page_content:
        page_content = page_content.split("||")[0]
    elif "." in page_content:
        page_content = page_content.split(".")[0]
    # print(f"Page Content: {page_content}")
    return page_content


BASE_IRI = "http://ccb.hms.harvard.edu/t2t/"

STOP_WORDS = {
    "in",
    "the",
    "any",
    "all",
    "for",
    "and",
    "or",
    "dx",
    "on",
    "fh",
    "tx",
    "only",
    "qnorm",
    "w",
    "iqb",
    "s",
    "ds",
    "rd",
    "rdgwas",
    "ICD",
    "excluded",
    "excluding",
    "unspecified",
    "certain",
    "also",
    "undefined",
    "ordinary",
    "least",
    "squares",
    "FINNGEN",
    "elsewhere",
    "more",
    "excluded",
    "classified",
    "classifeid",
    "unspcified",
    "unspesified",
    "specified",
    "acquired",
    "combined",
    "unspeficied",
    "elsewhere",
    "not",
    "by",
    "strict",
    "wide",
    "definition",
    "definitions",
    "confirmed",
    "chapter",
    "chapters",
    "controls",
    "characterized",
    "main",
    "diagnosis",
    "hospital",
    "admissions",
    "other",
    "resulting",
    "from",
}

TEMPORAL_WORDS = {
    "age",
    "time",
    "times",
    "date",
    "initiation",
    "cessation",
    "progression",
    "duration",
    "early",
    "late",
    "later",
    "trimester",
}

QUANTITY_WORDS = {
    "hourly",
    "daily",
    "weekly",
    "monthly",
    "yearly",
    "frequently",
    "per",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "years",
    "total",
    "quantity",
    "amount",
    "level",
    "levels",
    "volume",
    "count",
    "counts",
    "percentage",
    "abundance",
    "proportion",
    "content",
    "average",
    "prevalence",
    "mean",
    "ratio",
}

BOLD = "\033[1m"
END = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def remove_duplicates(values):
    seen = set()
    return [x for x in values if not (x in seen or seen.add(x))]


def combine_if_different(list_1, list_2, separator="|") -> str:
    """
    Combines two lists with a separator if they are different. Returns a concatenated string if different, else a single value.
    """
    joined_1 = join_or_single(list_1)
    joined_2 = join_or_single(list_2)

    if joined_1 != joined_2:
        # Combine them with a separator if they are different
        return (
            f"{joined_1}{separator}{joined_2}"
            if joined_1 and joined_2
            else joined_1 or joined_2
        )
    else:
        # If they are the same, return one of them
        return joined_1


def join_or_single(items, seperator="|") -> str:
    if items is None:
        return ""
    if isinstance(items, str):
        return items
    unique_items = remove_duplicates(filter(None, items))
    return (
        "|".join(unique_items)
        if len(unique_items) > 1
        else unique_items[0]
        if unique_items
        else ""
    )


def create_processed_result(result_object: ProcessedResultsModel = None) -> dict:
    """
    Processes the input query and maps it to standardized concepts, returning the processed result.

    Parameters:
    -----------
    result_object : ProcessedResultsModel
        A dictionary containing processed documents with information like standard labels, codes, and concept IDs.

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
    if result_object is None:
        return {
            "VARIABLE NAME": None,
            "VARIABLE LABEL": None,
            "Domain": None,
            "Variable Concept Label": None,
            "Variable Concept Code": None,
            "Variable Concept OMOP ID": None,
            "Additional Context Concept Label": None,
            "Additional Context Concept Code": None,
            "Additional Context OMOP ID": None,
            "Primary to Secondary Context Relationship": None,
            "Categorical Values Concept Label": None,
            "Categorical Values Concept Code": None,
            "Categorical Values OMOP ID": None,
            "Unit Concept Label": None,
            "Unit Concept Code": None,
            "Unit OMOP ID": None,
        }
    additional_entities = result_object.additional_entities
    additional_entities_matches = result_object.additional_entities_matches
    categorical_values = result_object.categories
    categorical_values_matches = result_object.categories_matches
    # main_term = result_object.base_entity
    main_term_matches = result_object.base_entity_matches
    # query_text = result_object.original_query

    (
        additional_entities_labels,
        additional_entities_codes,
        additional_entities_omop_ids,
    ) = format_categorical_values(
        additional_entities_matches, additional_entities, type="additional"
    )
    categorical_values_labels, categorical_values_codes, categorical_values_omop_ids = (
        format_categorical_values(categorical_values_matches, categorical_values)
    )
    main_term_labels = (
        main_term_matches[0].standard_label if len(main_term_matches) >= 1 else None
    )
    main_term_codes = (
        main_term_matches[0].standard_code if len(main_term_matches) >= 1 else None
    )
    main_term_omop_id = (
        main_term_matches[0].standard_omop_id if len(main_term_matches) >= 1 else None
    )
    # if additional_entities:
    #     combined_labels=combine_if_different(main_term_labels, additional_entities_labels, separator="|")
    #     combined_codes=combine_if_different(main_term_codes, additional_entities_codes, separator="|")
    #     combined_omop_ids=combine_if_different(main_term_omop_id, additional_entities_omop_ids, separator="|")

    # else:
    #     combined_labels = join_or_single(main_term_labels)
    #     combined_codes = join_or_single(main_term_codes)
    #     combined_omop_ids = join_or_single(main_term_omop_id)
    results = {
        "VARIABLE NAME": result_object.variable_name,
        "VARIABLE LABEL": result_object.original_query,
        "Domain": result_object.domain,
        "Variable Concept Label": main_term_labels,
        "Variable Concept Code": main_term_codes,
        "Variable Concept OMOP ID": main_term_omop_id,
        "Additional Context Concept Label": additional_entities_labels,
        "Additional Context Concept Code": additional_entities_codes,
        "Additional Context OMOP ID": additional_entities_omop_ids,
        "Primary to Secondary Context Relationship": result_object.primary_to_secondary_rel,
        "Categorical Values Concept Label": categorical_values_labels,
        "Categorical Values Concept Code": categorical_values_codes,
        "Categorical Values OMOP ID": categorical_values_omop_ids,
        "Unit Concept Label": result_object.unit_matches[0].standard_label
        if len(result_object.unit_matches) >= 1
        else None,
        "Unit Concept Code": result_object.unit_matches[0].standard_code
        if len(result_object.unit_matches) >= 1
        else None,
        "Unit OMOP ID": result_object.unit_matches[0].standard_omop_id
        if len(result_object.unit_matches) >= 1
        else None,
    }
    # print(f"Processed Result={results}")
    return results


def create_result_dict(tuple_item):
    result = {
        "VARIABLE NAME": tuple_item[0],
        "VARIABLE LABEL": tuple_item[1],
        "Domain": tuple_item[2],
        "Variable Concept Label": tuple_item[3],
        "Variable Concept Code": tuple_item[4],
        "Variable Concept OMOP ID": tuple_item[5],
        "Additional Context Concept Label": tuple_item[6],
        "Additional Context Concept Code": tuple_item[7],
        "Additional Context OMOP ID": tuple_item[8],
        "Primary to Secondary Context Relationship": tuple_item[9],
        "Categorical Values Concept Label": tuple_item[10],
        "Categorical Values Concept Code": tuple_item[11],
        "Categorical Values OMOP ID": tuple_item[12],
        "Unit Concept Label": tuple_item[13],
        "Unit Concept Code": tuple_item[14],
        "Unit OMOP ID": tuple_item[15],
    }
    print(f"EXISTING MAPPING RESULT={result}")
    return result


def format_categorical_values(
    values_matches_documents: Dict[str, List[RetrieverResultsModel]],
    values_list: List[str],
    type="categorical",
) -> Tuple[str, str, str]:
    labels = []
    codes = []
    ids = []
    # print(f"values: {status}\nDocs: {status_docs}")
    if values_matches_documents is None or len(values_matches_documents) == 0:
        return None, None, None

    # Normalize the keys of status_docs to lower case for case-insensitive matching
    normalized_status_docs = {k.lower(): v for k, v in values_matches_documents.items()}
    for v_ in values_list:
        v_ = v_.strip().lower()
        if v_ in normalized_status_docs:
            docs = normalized_status_docs[v_]
            # if len(docs) > 1:
            #     labels.append(' and/or '.join(remove_duplicates([doc.standard_label for doc in docs])))
            #     codes.append(' and/or '.join(remove_duplicates([doc.standard_code for doc in docs])))
            #     ids.append(' and/or '.join(remove_duplicates(doc.standard_concept_id)) for doc in docs])))
            # elif len(docs) == 1:
            labels.append(docs[0].standard_label)
            codes.append(docs[0].standard_code)
            ids.append(docs[0].standard_omop_id)
        elif type == "categorical":
            labels.append("na")
            codes.append("na")
            ids.append("na")

    return "|".join(labels), "|".join(codes), "|".join(ids)


def process_synonyms(synonyms_text: str) -> List[str]:
    """
    Processes the synonyms text to split by ';;' if it exists, otherwise returns the whole text as a single item list.
    If synonyms_text is empty, returns an empty list.
    """
    if synonyms_text:
        if ";;" in synonyms_text:
            return synonyms_text.split(";;")
        else:
            return [synonyms_text]
    return []


def parse_tokens(
    text: str, semantic_type: bool = False, domain: str = None
) -> Tuple[str, List[str], str]:
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
    if (
        "concept name:" in text_
        and "synonyms:" in text_
        and "domain:" in text_
        and "concept class:" in text_
        and "vocabulary:" in text_
    ):
        parts = text_.split(", ")
        concept_dict = {}
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                concept_dict[key.strip()] = value.strip()
            else:
                concept_dict["concept name"] = part.strip()
        entity = concept_dict.get("concept name", None)
        synonyms_str = concept_dict.get("synonyms", "")
        synonyms = process_synonyms(synonyms_str)
        if semantic_type:
            if domain:
                semantic_type_of_entity = concept_dict.get("domain", "")
                entity = f"{entity}||{semantic_type_of_entity}"
        global_logger.info(
            f"Entity: {entity}, Synonyms: {synonyms}, Semantic Type of Entity: {semantic_type_of_entity}"
        )
    else:
        # Fallback to the old format handling
        if "<ent>" in text_ and "<syn>" in text_:
            entity = text_.split("<ent>")[1].split("</ent>")[0]
            synonyms_text = text_.split("<syn>")[1].split("</syn>")[0]
            synonyms = process_synonyms(synonyms_text)
            global_logger.info(f"Entity: {entity}, Synonyms: {synonyms}")
        else:
            entity = text_
            synonyms = []

    return entity, synonyms, None


def combine_ent_synonyms(text: str, semantic_type=False, domain=None) -> str:
    """Extract entity and synonyms from the formatted text, if present."""
    text_ = text.strip().lower()
    if "<desc>" in text_:
        description = text_.split("<desc>")[1].split("</desc>")[0]
        global_logger.info(f"Description: {description}")
        text = text_.split("<desc>")[0]
    return text_


def print_docs(docs):
    for res, i in zip(docs, range(len(docs))):
        print("----" + str(i + 1) + "----")
        # print("LABEL: " + res.metadata["label"])
        print(f"{res.metadata['label']}:{res.metadata['domain']}")


def filter_irrelevant_domain_candidates(docs, domain) -> List[RetrieverResultsModel]:
    select_vocabs = select_vocabulary(domain=domain)

    # Filter documents based on selected vocabularies
    docs_ = [doc for doc in docs if doc.metadata["vocab"] in select_vocabs]
    # If no documents match, add 'snomed' to the vocabularies and filter again
    if not docs_:
        combined_vocabs = chain(["snomed"], select_vocabs)
        docs_ = [doc for doc in docs if doc.metadata["vocab"] in combined_vocabs]
    return docs_


def pretty_print_docs(docs) -> None:
    # Use a single print statement to reduce I/O operations
    print(
        "\n".join(
            [f"****{doc.metadata['label']}:{doc.metadata['vocab']}****" for doc in docs]
        )
    )


# def estimate_token_cost(text,filename):
#     # Get the encoder for the GPT-4 model
#     enc = tiktoken.encoding_for_model("gpt-4")

#     # Encode the text and count the tokens
#     n_tokens = len(enc.encode(text))

#     # Save the token count to a text file
#     with open(filename, 'w') as f:
#         f.write(f"Token count: {n_tokens}")

#     print(f"Token count saved to {filename}")


def append_results_to_csv(
    input_file, results, output_suffix="_mapped.csv", llm_id: str = ""
) -> None:
    """
    Reads the input CSV file, uses the number of rows according to `results` length, appends new columns,
    and saves it with a new name with the suffix '_mapped.csv'.

    Parameters:
    -----------
    input_file : str
        The path to the input CSV file.
    results : list of dict
        A list of dictionaries containing processed data. Each dictionary corresponds to a row.
    output_suffix : str, optional
        The suffix to append to the output CSV file name. Default is '_mapped.csv'.

    Returns:
    --------
    None
    """
    output_suffix = f"_{llm_id}_mapped.csv"
    # Step 1: Load the input CSV file
    df = pd.read_csv(input_file)

    # Step 2: Use only the number of rows that match the length of `results`
    if len(df) != len(results):
        df = df.iloc[: len(results)]

    # Step 3: Extract data from `results` to create new columns
    new_columns_data = {
        "Variable Concept Label": [
            result.get("Variable Concept Label", None) for result in results
        ],
        "Variable Concept Code": [
            result.get("Variable Concept Code", None) for result in results
        ],
        "Variable Concept OMOP ID": [
            result.get("Variable Concept OMOP ID", None) for result in results
        ],
        "Domain": [result.get("Domain", None) for result in results],
        "Additional Context Concept Label": [
            result.get("Additional Context Concept Label", None) for result in results
        ],
        "Additional Context Concept Code": [
            result.get("Additional Context Concept Code", None) for result in results
        ],
        "Additional Context OMOP ID": [
            result.get("Additional Context OMOP ID", None) for result in results
        ],
        "Primary to Secondary Context Relationship": [
            result.get("Primary to Secondary Context Relationship", None)
            for result in results
        ],
        "Categorical Values Concept Label": [
            result.get("Categorical Values Concept Label", None) for result in results
        ],
        "Categorical Values Concept Code": [
            result.get("Categorical Values Concept Code", None) for result in results
        ],
        "Categorical Values Concept OMOP ID": [
            result.get("Categorical Values Concept OMOP ID", None) for result in results
        ],
        "Unit Concept Label": [
            result.get("Unit Concept Label", None) for result in results
        ],
        "Unit Concept Code": [
            result.get("Unit Concept Code", None) for result in results
        ],
        "Unit OMOP ID": [result.get("Unit OMOP ID", None) for result in results],
        "Reasoning": [result.get("reasoning", None) for result in results],
        "Prediction": [result.get("prediction", None) for result in results],
    }

    # Step 4: Append the new columns to the dataframe
    for column_name, column_data in new_columns_data.items():
        df[column_name] = column_data

    # Step 5: Save the updated dataframe to a new CSV file
    file_name, _ = os.path.splitext(input_file)
    output_file = f"{file_name}{output_suffix}"
    df.to_csv(output_file, index=False)

    print(f"File saved: {output_file}")
    return df


def remove_repeated_phrases(text):
    words = text.split()
    result = []
    seen_phrases = set()

    i = 0
    while i < len(words):
        phrase = words[i]
        # Check if we are at the last word or the next word is not the same phrase
        if phrase not in seen_phrases:
            result.append(phrase)
            seen_phrases.add(phrase)
        i += 1

    return " ".join(result)


def extract_visit_number(visits_string):
    num_str = ""
    for char in visits_string:
        if char.isdigit():
            num_str += char
        elif (
            num_str
        ):  # Break as soon as we encounter the first non-digit after a number
            break
    return int(num_str) if num_str else None


def check_k(queries):
    # return len(queries[0]["mentions"][0]["candidates"])
    return 10


def evaluate_recall_mrr(data, top_k=10):
    """
    Evaluate recall@k and Mean Reciprocal Rank (MRR) for top-k predictions.
    """
    queries = data["queries"]

    # Initialize accumulators for recall@k and MRR
    recall_at_k = {f"recall@{i+1}": [] for i in range(top_k)}
    mrr_scores = []

    for query in queries:
        mentions = query["mentions"]
        for mention in mentions:
            candidates = mention["candidates"]
            # Identify the indices of relevant items
            relevant_indices = [
                idx
                for idx, candidate in enumerate(candidates)
                if candidate["label"] == 1
            ]
            total_relevant = len(relevant_indices)
            retrieved_relevant = 0
            reciprocal_rank = 0

            # Initialize relevant_found_at_k only up to the number of candidates
            relevant_found_at_k = [0] * min(top_k, len(candidates))

            for idx in range(len(candidates)):
                candidate = candidates[idx]
                if candidate["label"] == 1:
                    retrieved_relevant += 1
                    # Update reciprocal rank if it's the first relevant item
                    if reciprocal_rank == 0:
                        reciprocal_rank = 1 / (idx + 1)
                # Update cumulative count of relevant items found up to each k
                if idx < top_k:
                    relevant_found_at_k[idx] = retrieved_relevant

            # Fill in the recall values only up to the number of retrieved candidates
            for k in range(len(relevant_found_at_k)):  # Limit to available candidates
                if total_relevant > 0:
                    recall = relevant_found_at_k[k] / total_relevant
                else:
                    recall = 0
                recall_at_k[f"recall@{k+1}"].append(recall)

            # Append reciprocal rank for MRR calculation
            mrr_scores.append(reciprocal_rank)

    # Calculate average recall@k and MRR
    recall_at_k_avg = {k: np.mean(v) for k, v in recall_at_k.items() if v}
    # make it in percentage from 0-100 and round to 2 decimal places
    mrr = np.mean(mrr_scores)
    mrr = round(mrr, 4) if mrr else 0
    # make it in percentage from 0-100 and round to 2 decimal places
    recall_at_k_avg = {k: round(v * 100, 2) for k, v in recall_at_k_avg.items()}
    # Print results
    for k, recall in recall_at_k_avg.items():
        print(f"{k}: {recall:.4f}")
    print(f"MRR: {mrr:.4f}")

    # Update data with new metrics
    data.update(recall_at_k_avg)
    data["MRR"] = mrr
    return data


def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data["queries"]
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query["mentions"]
            mention_hit = 0
            for mention in mentions:
                candidates = mention["candidates"][: i + 1]  # to get acc@(i+1)
                mention_hit += np.any([candidate["label"] for candidate in candidates])

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit
            if mention_hit == len(mentions):
                hit += 1

        # make it in percentage from 0-100 and round to 2 decimal places
        data["acc{}".format(i + 1)] = round(hit / len(queries) * 100, 2)
        # data["acc{}".format(i + 1)] = hit / len(queries)
    # print accuracy  from 01-100 and round to 2 decimal points
    for i in range(0, k):
        print(f"acc{i+1}: {data['acc{}'.format(i+1)]:.2f}")
    data = evaluate_precision_at_k(data)
    data = evaluate_recall_mrr(data)
    return data


def evaluate_precision_at_k(data):
    top_k = check_k(data["queries"])
    """
    Evaluate precision@k for top-k predictions.
    """
    queries = data["queries"]
    precision_at_k = defaultdict(list)

    for query in queries:
        mentions = query["mentions"]
        for mention in mentions:
            candidates = mention["candidates"]
            relevant_retrieved = 0

            # Calculate precision at each k
            for idx in range(min(top_k, len(candidates))):
                candidate = candidates[idx]
                if candidate["label"] == 1:
                    relevant_retrieved += 1

                # Precision@k is the proportion of relevant items in top-k results
                precision_at_k[f"precision@{idx+1}"].append(
                    relevant_retrieved / (idx + 1)
                )

            # Fill in remaining k values with 0 if fewer than top_k candidates
            for j in range(len(candidates), top_k):
                precision_at_k[f"precision@{j+1}"].append(0)

    # Calculate average precision@k
    precision_at_k_avg = {k: np.mean(v) for k, v in precision_at_k.items()}
    # make it in percentage from 0-100 and round to 2 decimal places
    precision_at_k_avg = {k: round(v * 100, 2) for k, v in precision_at_k_avg.items()}
    # Print precision@k results
    for k, precision in precision_at_k_avg.items():
        print(f"{k}: {precision:2f}")

    # Update data with new metrics
    data.update(precision_at_k_avg)
    return data


def dcg(relevances, k):
    """Compute the Discounted Cumulative Gain (DCG) up to position k."""
    relevances = relevances[:k]
    return sum(
        rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)
    )  # idx + 2 because log2 starts at 2


def idcg(relevances, k):
    """Compute the Ideal DCG (IDCG) up to position k."""
    sorted_relevances = sorted(relevances, reverse=True)
    return dcg(sorted_relevances, k)


def calculate_ncgd(candidates, k):
    """Calculate NCGD at k."""
    relevances = [candidate["label"] for candidate in candidates]
    dcg_score = dcg(relevances, k)
    idcg_score = idcg(relevances, k)
    return dcg_score / idcg_score if idcg_score > 0 else 0


def evaluate_ncgd(data):
    top_k_values = [1, 3, 5, 10]
    queries = data["queries"]
    ncgd_at_k = {f"ncgd@{k}": [] for k in top_k_values}

    for query in queries:
        mentions = query["mentions"]
        for mention in mentions:
            candidates = mention["candidates"]
            for k in top_k_values:
                ncgd_score = calculate_ncgd(candidates, k)
                ncgd_at_k[f"ncgd@{k}"].append(ncgd_score)

    # Calculate average NCGD@k
    ncgd_at_k_avg = {k: np.mean(v) for k, v in ncgd_at_k.items()}
    # make it in percentage from 0-100 and round to 2 decimal places
    ncgd_at_k_avg = {k: round(v * 100, 2) for k, v in ncgd_at_k_avg.items()}
    # Print NCGD@k results
    for k, ncgd in ncgd_at_k_avg.items():
        print(f"{k}: {ncgd:.2f}")

    # Update data with new metrics
    data.update(ncgd_at_k_avg)
    return data

def count_tokens(prompt_text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in the given prompt text for the specified model.

    :param prompt_text: The text prompt for which to count tokens.
    :param model_name: The name of the model (e.g., "gpt-3.5-turbo", "llama3.1").
    :return: The total number of tokens in the prompt.
    """
    try:
        # For models that work with tiktoken (like GPT)
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(prompt_text))
    except Exception as e:
        print(f"tiktoken failed for model {model_name}: {e}")

    try:
        # For models like LLaMA that may use a different tokenizer
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        return len(tokenizer.encode(prompt_text))
    except Exception as e:
        print(f"Hugging Face tokenizer failed for model {model_name}: {e}")

    # Fallback if no tokenizer is available
    print("Warning: Using a simple word count as a fallback method.")
    return len(prompt_text.split())