from rag.vector_index import update_merger_retriever
import time
from rag.utils import global_logger as logger
from .llm_chain import *
from rag.eval import evaluate_with_multiple_mappings
import qdrant_client.http.models as rest
from rag.athena_api_retriever import AthenaFilters

def retriever_docs(query,  retriever, domain='all',is_omop_data=False):
    if is_omop_data: retriever= update_merger_retriever(retriever,domain)
    try:
        results =  retriever.invoke(query)
        unique_results = filter_results(query,results)[:20]
        # print(f"Unique Results:\n {[res.metadata['label'] for res in unique_results]}")
        return unique_results
    except Exception as e:
        logger.error(f"Error retrieving docs: {e}")
        return None

def map_data(data, retriever, custom_data=False, output_file=None, 
             llm_name='llama', topk=5,do_eval=False,is_omop_data=True):

    global RETRIEVER_CACHE
    start_time = time.time()
    max_queries = len(data)   #282
    results = []
    index = 0 
    for _, query in enumerate(data[:5]):
        name = query[1]
        # cui = query[0]
        query_result =  full_query_processing(name,retriever, llm_name, topk,custom_data=custom_data, is_omop_data=is_omop_data)
        if query_result:
            results.append(query_result) 
        else: print(f"NO RESULT FOR {query}")
        index +=  1
        if (index + 1) % 15 == 0:
            time.sleep(0.05)  # Adjusted to be more appropriate than 0.0005

    end_time = time.time()
    total_time = end_time - start_time
    save_results(results, output_file)
    if do_eval:
        return evaluate_with_multiple_mappings(data[:max_queries],  results,model_name=llm_name)
    logger.info(f"Total execution time for {max_queries} queries is {total_time} seconds.")
    return results

def full_query_processing(query_text,retriever, llm_name,topk,custom_data=False, is_omop_data=True):
    try:
        print(f"Processing query: {query_text}")
        # normalized_query_text = rule_base_decomposition(query_text)
        # processes_results = []
        # if not is_omop_data:
        #     query_decomposed = find_domain(query_text)
        #     query_decomposed['domain'] = 'all'
        #     processes_results = temp_process_query_details(query_decomposed, retriever, llm_name, topk, query_text)
        # else:
        # if not custom_data:
        #     query_decomposed = find_domain(query_text)
        #     processes_results = temp_process_query_details(query_decomposed, retriever, llm_name, topk, query_text)
        # else:
        query_decomposed = check_extracted_details(  extract_information(query_text, llm_name))
        processes_results =  temp_process_query_details(query_decomposed, retriever, llm_name, topk, query_text, is_omop_data)
        # print("Mapping result:", processes_results)
        return processes_results
    except Exception as e:
        logger.error(f"Error full processing query: {e}")
        return None



def temp_process_query_details(query_decomposed, retriever_cache, llm_name, topk, query_text,is_omop_data=False):
    try:
        if query_decomposed:
            original_query_matches, label_matches, additional_entities_matches,status_docs,unit_docs = None, None, None, None, None
            original_query_matches, found_match=  process_retrieved_docs(query_text,  retriever_docs(query_text, retriever_cache, domain='all',is_omop_data=is_omop_data),None, 'all')
            print(f"original_query_matches match_found={found_match}")
            if found_match:
                return create_processed_result(query_text, query_text, original_query_matches, domain='all')
            
            main_term = query_decomposed.get('base_entity', query_text)
            context = query_decomposed.get('additional_entities',None) 
            status = query_decomposed.get('status', None)
            domain = query_decomposed.get('domain','all').strip().lower()
            unit = query_decomposed.get('unit',None)
            print(f"main_term={main_term}, context={context}, status={status}, domain={domain}, unit={unit}")
            if main_term:
                # print(f"main_query_retriever={retriever_cache}")
                label_matches, _ =  process_retrieved_docs(main_term,  retriever_docs(main_term, retriever_cache, domain=domain), llm_name, domain)
            if context:
                additional_entities_matches =  process_values(context, retriever_cache, llm_name, domain=domain, values_type='additional') if context else []
            if status:
                status_docs =  process_values(status,retriever_cache , llm_name, domain='all', values_type='status') if status else []
            if unit:
                unit_docs =  process_unit(unit, retriever_cache, llm=llm_name, domain='unit') if unit else []
            mapping_result = create_processed_result(query_text, main_term, label_matches, domain=domain, 
                                    values_docs=additional_entities_matches,status_docs=status_docs, unit_docs=unit_docs,
                                    context=context, status=status, unit=unit)
            return mapping_result
    except Exception as e:
        logger.error(f"Error full processing query: {e}")
        return None


def process_retrieved_docs(query, docs, llm_name=None, domain = None):
    # print_docs(docs)
    if docs:
        if matched_docs := exact_match_found(query_text=query, documents=docs, domain=domain):
            return post_process_candidates(matched_docs, max=2), True
        if llm_name:
            print(f"no exact match found for {query}")
            domain_specific_docs = filter_irrelevant_domain_candidates(docs, domain)
            llm_ranks, match_found =  pass_to_chat_llm_chain(query, domain_specific_docs, llm_name=llm_name,domain=domain)
            return post_process_candidates(llm_ranks, max=1), match_found
        else:
            return docs, False
    else:
        logger.info(f"No docs found for query={query}")
    return [], False


def process_context(context,retriever,llm, domain = None):
    context = normalize(context)
    if context:
        if docs :=  retriever_docs(context,retriever, domain='all'):
            if matched_docs := exact_match_found(query_text=context, documents=docs, domain=domain):
                return post_process_candidates(matched_docs, max=2)
            llm_results,_ =  pass_to_chat_llm_chain(context, docs,llm_name=llm,domain=domain)
            return post_process_candidates(llm_results, max=2)
    return []     

def process_values(values,retriever,llm,domain =None, values_type='additional'):
    if isinstance(values, str):
        values = [values]
    logger.info(f"processing values={values}")
    all_values = {}
    if not values:  # If the values list itself is empty or None, exit early
        return all_values
        # Process each value, similar to context
    for q_value in values:
            q_value = normalize(q_value)
            if q_value and q_value != 'unknown':
                # updated_q_value = rule_base_decomposition(q_value)
                # print(f"q_value={updated_q_value}")
                # code, value, vid=check_value_in_csv_by_type(updated_q_value,values_type)
                # if code and value:
                #     print(f"foudnd code={code} and value={value}")
                #     all_values[updated_q_value] = [{'standard_label':value,'standard_code':code, 'concept_id':vid}]
                #     continue
                if categorical_value_results:=  retriever_docs(q_value, retriever, domain='all'):
                    print(f"retrieved categorical_value_results type={type(categorical_value_results)}")
                    if matched_docs:= exact_match_found(query_text=q_value, documents=categorical_value_results, domain=domain):
                        max_results = 2 if ('or' in q_value or 'and' in q_value) else 1
                        # print(f"max_results={max_results} for {updated_q_value}")
                        all_values[q_value] = post_process_candidates(matched_docs, max=max_results)
                    elif len(categorical_value_results) > 0:
                        updated_results,_ =  pass_to_chat_llm_chain(q_value, categorical_value_results, llm_name=llm,domain=domain)
                        if updated_results:
                            max_results = 2 if ('or' in q_value or 'and' in q_value) else 1
                            # logger.info(f"max_results={max_results} for {updated_q_value}")
                            all_values[q_value] = post_process_candidates(updated_results, max=max_results)
    # print(f"all_values={all_values}")
    return all_values


def process_unit(unit,retriever, llm:Any,domain:str):
    unit = normalize(unit)
    unit_results = []
    if unit and unit != 'unknown':
        # Process unit
        # code, value,uid=check_value_in_csv_by_type(unit, 'unit')
        # if code and value:
        #     return {'standard_label':value,'standard_code':code, 'concept_id':uid}
        unit_results =  retriever_docs(unit,retriever, domain='unit')
        if unit_results:
           exact_units = exact_match_found(unit, unit_results, domain='unit')
           if len(exact_units) > 0:
               unit_results = post_process_candidates(exact_units, max=1)
           else:
                llm_results , _ =  pass_to_chat_llm_chain(unit, unit_results, llm_name=llm,domain=domain)
                unit_results = post_process_candidates(llm_results,max=1)
    return unit_results 


def save_results(results, file_path):
    if results:
    # if file_path.endswith('.csv'):
        save_to_csv(results, file_path)
    else:
        logger.info("No results to save.")
        
        
def update_api_search_filter(api_retriever, domain='observation'):
    
    if domain == 'unit':
         api_retriever.filters = AthenaFilters(domain=None, vocabulary=['UCUM'], standard_concept=['Standard'])
    elif domain == 'condition' or domain == 'anatomic site':
        api_retriever.filters  = AthenaFilters(domain=['Condition','Meas Value','Spec Anatomic Site'], vocabulary=['SNOMED'],standard_concept=['Standard'])
    elif domain == 'measurement':
        api_retriever.filters  = AthenaFilters(domain=['Measurement','Meas Value','Observation'], vocabulary=['LOINC','MeSH','SNOMED'],standard_concept=['Standard'])
    elif domain == 'drug':
        api_retriever.filters  = AthenaFilters(domain=['Drug'], vocabulary=['RxNorm','ATC','SNOMED'], standard_concept=['Standard','Classification'])
    elif domain == 'observation':
        api_retriever.filters  = AthenaFilters(domain=['Observation','Meas Value'], vocabulary=['SNOMED','LOINC','OMOP Extension'])
    elif domain == 'visit':
        api_retriever.filters  = AthenaFilters(domain=['Visit','Observation'], vocabulary=['SNOMED','LOINC','OMOP Extension'])
    elif domain == 'demographics':
        api_retriever.filters  = AthenaFilters(domain=['Observation','Meas Value'], vocabulary=['SNOMED','LOINC','OMOP Extension','Gender','Race','Ethnicity'])
    else:
        api_retriever.filters  = AthenaFilters()
    return api_retriever


def update_qdrant_search_filter(retriever, domain='unknown'):
    if domain == 'unit':
        retriever.search_kwargs['filter'] = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.vocab",
                            match=rest.MatchValue(value='ucum')
                        )
                    ]
                )
    elif domain == 'condition' or domain == 'anatomic site':
        retriever.search_kwargs['filter'] = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.vocab",
                            match=rest.MatchAny(any=['snomed'])
                        
                        )
                    ]
                )
    elif domain == 'demographics':
        retriever.search_kwargs['filter'] = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.vocab",
                            match=rest.MatchAny(any=['snomed','loinc'])
                        ),
                        rest.FieldCondition(
                            key="metadata.domain",
                            match=rest.MatchAny(any=['observation','meas value'])
                        )      
                    ]
                )
    elif domain == 'measurement':
        retriever.search_kwargs['filter'] = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.vocab",
                            match=rest.MatchAny(any=['loinc','mesh','snomed'])
                        ),
                        rest.FieldCondition(
                            key="metadata.domain",
                            match=rest.MatchAny(any=['measurement','meas value','observation'])
                        )
                    ]
                )
    elif domain == 'drug':
        retriever.search_kwargs['filter'] = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.vocab",
                            match=rest.MatchAny(any=['rxnorm','atc','snomed'])
                        ),
                        rest.FieldCondition(
                            key="metadata.is_standard",
                            match=rest.MatchAny(any=['S','C'])
                        )
                    ]
                )
    elif domain == 'observation' or domain == 'visit' or domain == 'demographics' or domain == 'history of events' or domain == 'life style':
        retriever.search_kwargs['filter'] = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.vocab",
                            match=rest.MatchAny(any=['snomed','loinc','omop extension'])
                        ),
                        rest.FieldCondition(
                            key="metadata.domain",
                            match=rest.MatchAny(any=['observation','meas value'])
                        )
                    ]
        )
    elif domain == 'procedure':
        retriever.search_kwargs['filter'] = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.vocab",
                            match=rest.MatchAny(any=['snomed'])
                        )
                    ]
                )
    else:
        retriever.search_kwargs['filter'] = None
    return retriever


def filter_results(query, results):
    prioritized = []
    non_prioritized = []
    query = query.strip().lower()  # Normalize the query for comparison
    # print(f"Original results before filtering = {results}")

    # First pass: collect prioritized and non-prioritized results
    for res in results:
        label = res.metadata['label'].strip().lower()  # Normalize the label for comparison
        if label == query:
            prioritized.append(res)  # Add to prioritized list if label matches the query
        else:
            non_prioritized.append(res)  # Add to non-prioritized list if label does not match

    # Combine prioritized and non-prioritized lists
    combined_results = prioritized + non_prioritized
    
    # print(f"Original results after filtering = {combined_results}")
    return combined_results
