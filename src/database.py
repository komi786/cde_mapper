from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
from .param import VECTOR_PATH, QDRANT_PORT
import csv
from .utils import parse_query_text, global_logger as logger
from pathlib import Path
class QdrantClientSingleton:
    _instance = None
    @classmethod
    def get_instance(cls, url=VECTOR_PATH, port=QDRANT_PORT, prefer_grpc=False):
        if cls._instance is None:
            cls._instance = QdrantClient(url=url, port=port, https=True,timeout=500)
            cls._instance.get_collections()
        return cls._instance
def add_domain_specific_conditions(domain, vocabs, should_conditions, must_conditions, query=None):
    """
    Adds domain-specific conditions to the should_conditions and must_conditions lists.
    
    Args:
        domain (str): The domain for which the retriever is being defined.
        vocabs (list): List of vocabularies to use for filtering.
        should_conditions (list): List of 'should' conditions for filtering.
        must_conditions (list): List of 'must' conditions for filtering.
    """
    if domain == 'unit':
        must_conditions.extend([
                models.FieldCondition(
                    key="metadata.vocab",
                    match=models.MatchText(text="ucum")
                ),
                models.FieldCondition(
                    key="metadata.scode",
                    match=models.MatchText(text=query)
                )
            ]
        )
    # if domain == 'drug':
    #     should_conditions.extend([
    #         models.FieldCondition(
    #             key="metadata.vocab",
    #             match=models.MatchAny(any=vocabs),
    #         ),
    #         # models.FieldCondition(
    #         #     key="metadata.concept_class",
    #         #     match=models.MatchAny(any=["atc 5th","atc 4th","disposition","clinical drug","clinical drug form",'clinical dose group','branded drug form','clinical drug comp'])
    #         # ),
    #         # models.FieldCondition(
    #         #     key="metadata.domain",
    #         #     match=models.MatchText(text=domain),
    #         # ),
    #         models.FieldCondition(
    #             key="metadata.is_standard",
    #             match=models.MatchAny(any=["S","C"]),
    #         )
    #     ])
    elif domain == 'measurement':
        must_conditions.extend([
            models.FieldCondition(
                key="metadata.domain",
                match=models.MatchText(text=domain),
            ),
            models.FieldCondition(
                key="metadata.vocab",
                match=models.MatchAny(any=vocabs),
            )
        ])
    # elif domain == 'visit':
    #     should_conditions.extend([
    #         models.FieldCondition(
    #             key="metadata.domain",
    #             match=models.MatchText(text="observation"),
    #         ),
    #         # models.FieldCondition(
    #         #     key="metadata.vocab",
    #         #     match=models.MatchAny(any=vocabs),
    #         # ),
    #          models.FieldCondition(
    #             key="metadata.is_standard",
    #             match=models.MatchAny(any=["S","C"]),
    #         )
    #     ])
    else:
        must_conditions.extend([
            models.FieldCondition(
            key="metadata.vocab",
            match=models.MatchAny(any=vocabs)
        )])

def update_search_param(retriever, must_conditions, should_conditions, score_threshold=0.5, dense=False, topk=10):
    logger.info(f"must={must_conditions}\nshould={should_conditions}, dense={dense}")
    # logger.info(f"must={must_conditions}\nshould={should_conditions}, dense={dense}")
    if dense:
            if len(must_conditions) > 0:
                        retriever.search_kwargs.update({
                            "k": topk,
                            "score_threshold": score_threshold,
                            "filter": models.Filter(must=must_conditions)
                        })
            elif len(should_conditions) > 0:
                retriever.search_kwargs.update({
                    "k": topk,
                    "score_threshold": score_threshold,
                    "filter": models.Filter(should=should_conditions)
                })
    else:
        if len(must_conditions) > 0:
            retriever.search_options =  {
                    # "search_params": {
                    #      "hnsw_ef": 128,
                    # "exact": False,
                    # },
                    "score_threshold": 0.5
                    
                }
            retriever.filter = models.Filter(must=must_conditions)
        elif len(should_conditions) > 0:
            retriever.search_options =  {
                    # "search_params": {
                    #      "hnsw_ef": 128,
                    # "exact": False
                    # # "quantization" : {
                    # #     "rescore" : True
                    # # }
                    # },
                    "score_threshold": 0.5
                    
                }
            retriever.filter = models.Filter(should=should_conditions)
    return retriever

import csv
import os
from rag.param import MAPPING_CACHE_FILE
def initialize_csv_cache(cache_file=MAPPING_CACHE_FILE):
    if not os.path.exists(cache_file):
        with open(cache_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['query_text', 'revised_query', 'domain', 'standard_label', 'standard_code','standard_concept_id',
                             'additional_context', 'additional_context_codes','additional_context_concept_ids', 'categorical_values',
                             'categorical_codes', 'categorical_concept_ids','unit', 'unit_code','unit_concept_id'])

def store_in_csv_cache(query_text, result, cache_file=MAPPING_CACHE_FILE):
    with open(cache_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            query_text, result['revised_query'], result['domain'], result['standard_label'],
            result['standard_code'],result['standard_concept_id'], result['additional_context'], result['additional_context_codes'],result['additional_context_concept_ids'],
            result['categorical_values'], result['categorical_codes'], result['categorical_concept_ids'],result['unit'], result['unit_code'],result['unit_concept_id']
        ])

def get_from_csv_cache(query_text, cache_file=MAPPING_CACHE_FILE):
    if not os.path.exists(cache_file):
        return None
    
    with open(cache_file, mode='r') as file:
        reader = csv.DictReader(file)
        #query_text, status:[], unit:''
        # query_components = parse_query_text(query_text)
        #idea is that we need to check if each component of query is present in the cache using check_value_in_csv_by_type
        for row in reader:
            if row['query_text'] == query_text:
                return row
            # elif query_components:
            #     return get_component_from_csv_cache(None, query_components[1], query_components[2], main_query=query_components[0])
    return None

import csv
from  langchain.schema import Document
#query_text,revised_query,domain,standard_label,standard_code,additional_context,additional_context,categorical_values,categorical_codes,unit,unit_code

def check_value_in_csv_by_type(value_to_check, search_type, csv_file_path=MAPPING_CACHE_FILE):
    
    """
    Searches for a specific value in designated columns of a CSV file based on the type of search.
    Returns the corresponding code if the value is found.

    Args:
    value_to_check (str): The value to search for in the CSV.
    search_type (str): The type of the value to search ('categorical', 'unit', 'main_query').
    csv_file_path (str): Path to the CSV file.

    Returns:
    (str, str): Tuple where the first element is the code if the value is found, None otherwise.
                The second element is the column name where the value was found or None.
    """
    # Define the columns to search and their corresponding code columns based on the type
    type_to_columns = {
        'status': {'search_columns': ['categorical_values'], 'code_column': 'categorical_codes', 'id_column': 'categorical_concept_ids'},
        'additional': {'search_columns': ['additional_context'], 'code_column': 'additional_context_codes', 'id_column': 'additional_context_concept_ids'},
        'unit': {'search_columns': ['unit'], 'code_column': 'unit_code','id_column': 'unit_concept_id'},
        'main_query': {'search_columns': ['standard_label'], 'code_column': 'standard_code','id_column': 'standard_concept_id'}
    }
    logger.info(f"value_to_check={value_to_check}, search_type={search_type}")

    if search_type not in type_to_columns:
        return None, None, None

    try:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            search_columns = type_to_columns[search_type]['search_columns']
            code_column = type_to_columns[search_type]['code_column']
            id_column = type_to_columns[search_type]['id_column']
            for row in reader:
                for column_name in search_columns:
                    if column_name in row:
                        # Split the values in the cell and check each one
                        cell_values = row[column_name].split('|') if row.get(column_name) else []
                        cell_codes = row[code_column].split('|') if row.get(code_column) else []
                        cell_ids = row[id_column].split('|') if row.get(id_column) else []
                        for i, cell_value in enumerate(cell_values):
                                if cell_value.strip() == value_to_check:
                                    code = cell_codes[i].strip() if i < len(cell_codes) else None
                                    id_ = cell_ids[i].strip() if i < len(cell_ids) else None
                                    # logger.info(f"Value '{value_to_check}' found in column '{column_name}', Code: '{code}'")
                                    return code, cell_value, id_
        logger.info(f"Value '{value_to_check}' not found in specified columns.")
        return None, None, None
    except FileNotFoundError:
        logger.info(f"The file {csv_file_path} does not exist.")
        return None, None, None
    except Exception as e:
        logger.info(f"An error occurred in Check_Value_in_CSV: {str(e)}")
        return None, None, None
    
    
def get_component_from_csv_cache(context, status, unit, cache_file=MAPPING_CACHE_FILE,main_query=None):
    if not os.path.exists(cache_file):
        logger.info(f"Cache file {cache_file} does not exist.")
        return None
    component_results = {'main_query':None,'context': None, 'status': None, 'unit': None}

    with open(cache_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if main_query and row['standard_label'] == main_query:
                component_results['main_query'] = row 
            if context and row['additional_context'] == context:
                component_results['context'] = row
            if status and row['categorical_values'] == status:
                component_results['status'] = row
            if unit and row['unit'] == unit:
                component_results['unit'] = row

    return component_results


import sqlite3
import os
from rag.param import MAPPING_CACHE_FILE

DB_FILE = MAPPING_CACHE_FILE.replace('.csv', '.db')

def initialize_sqlite_cache(db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY,
            query_text TEXT,
            revised_query TEXT,
            domain TEXT,
            standard_label TEXT,
            standard_code TEXT,
            additional_context TEXT,
            additional_context_codes TEXT,
            categorical_values TEXT,
            categorical_codes TEXT,
            unit TEXT,
            unit_code TEXT
        )
    ''')
    conn.commit()
    conn.close()

def store_in_sqlite_cache(query_text, result, db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO cache (
            query_text, revised_query, domain, standard_label, standard_code,
            additional_context, additional_context_codes, categorical_values,
            categorical_codes, unit, unit_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        query_text, result['revised_query'], result['domain'], result['standard_label'],
        result['standard_code'], result['additional_context'], result['additional_context_codes'],
        result['categorical_values'], result['categorical_codes'], result['unit'], result['unit_code']
    ))
    conn.commit()
    conn.close()

def get_from_sqlite_cache(query_text, db_file=DB_FILE):
    if not os.path.exists(db_file):
        return None

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM cache WHERE query_text = ?
    ''', (query_text,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'query_text': row[0],
            'revised_query': row[1],
            'domain': row[2],
            'standard_label': row[3],
            'standard_code': row[4],
            'additional_context': row[5],
            'additional_context_codes': row[6],
            'categorical_values': row[7],
            'categorical_codes': row[8],
            'unit': row[9],
            'unit_code': row[10]
        }
    return None

def get_component_from_sqlite_cache(context, status, unit, db_file=DB_FILE):
    if not os.path.exists(db_file):
        return None

    component_results = {'context': None, 'status': None, 'unit': None}

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    if context:
        cursor.execute('''
            SELECT * FROM cache WHERE additional_context = ?
        ''', (context,))
        row = cursor.fetchone()
        if row:
            component_results['context'] = {
                'query_text': row[0],
                'revised_query': row[1],
                'domain': row[2],
                'standard_label': row[3],
                'standard_code': row[4],
                'additional_context': row[5],
                'additional_context_codes': row[6],
                'categorical_values': row[7],
                'categorical_codes': row[8],
                'unit': row[9],
                'unit_code': row[10]
            }

    if status:
        cursor.execute('''
            SELECT * FROM cache WHERE categorical_values = ?
        ''', (status,))
        row = cursor.fetchone()
        if row:
            component_results['status'] = {
                'query_text': row[0],
                'revised_query': row[1],
                'domain': row[2],
                'standard_label': row[3],
                'standard_code': row[4],
                'additional_context': row[5],
                'additional_context_codes': row[6],
                'categorical_values': row[7],
                'categorical_codes': row[8],
                'unit': row[9],
                'unit_code': row[10]
            }

    if unit:
        cursor.execute('''
            SELECT * FROM cache WHERE unit = ?
        ''', (unit,))
        row = cursor.fetchone()
        if row:
            component_results['unit'] = {
                'query_text': row[0],
                'revised_query': row[1],
                'domain': row[2],
                'standard_label': row[3],
                'standard_code': row[4],
                'additional_context': row[5],
                'additional_context_codes': row[6],
                'categorical_values': row[7],
                'categorical_codes': row[8],
                'unit': row[9],
                'unit_code': row[10]
            }

    conn.close()
    return component_results

# Initialize the database only once
# initialize_sqlite_cache()
