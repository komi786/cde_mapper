from .vector_index import update_compressed_merger_retriever, update_merger_retriever
import time
from .utils import global_logger as logger
from .utils import (
  
    # save_to_csv,
    convert_row_to_entities,
    # create_result_dict,
    convert_db_result,
    create_processed_result,
    post_process_candidates,
    exact_match_found,
    filter_irrelevant_domain_candidates,
    add_result_to_training_data
)
from .evalmap import perform_mapping_eval_for_variable
from .llm_chain import pass_to_chat_llm_chain, extract_information
# from .eval import evaluate_with_multiple_mappings
from .py_model import QueryDecomposedModel, ProcessedResultsModel, RetrieverResultsModel
from .sql import DataManager
from .param import DB_FILE, MAPPING_FILE
from typing import Any

# Cache for retrievers based on domain
RETRIEVER_CACHE = {}

# def get_cached_retriever(retriever, domain, topk=10):
#     cache_key = (domain, topk)
#     if cache_key not in RETRIEVER_CACHE:
#         if domain != 'all':
#             retriever = update_merger_retriever(retriever, domain, topk=topk)
#         RETRIEVER_CACHE[cache_key] = retriever
#     return RETRIEVER_CACHE[cache_key]


# def retriever_docs(query, retriever, domain='all', is_omop_data=False, topk=10):
#     print(f"selected domain={domain}")
#     retriever = get_cached_retriever(retriever, domain, topk)
#     try:
#         results = retriever.invoke(query)
#         unique_results = filter_results(query, results)[:10]
#         print(f"length of unique results={len(unique_results)}")
#         return unique_results
#     except Exception as e:
#         logger.error(f"Error retrieving docs: {e}")
#         return None


def retriever_docs(query, retriever, domain="all", is_omop_data=False, topk=10):
    if is_omop_data:
        if COMPRESSED_RETRIEVER:
            retriever = update_compressed_merger_retriever(retriever, domain, topk=topk)
        else:
            retriever = update_merger_retriever(retriever, domain, topk=topk)
    try:
        results = retriever.invoke(query)
        unique_results = filter_results(query, results)
        if unique_results:
            unique_results = unique_results[:10]
        else:
            unique_results = []
        print(f"length of unique results={len(unique_results)}")
        return unique_results
    except Exception as e:
        logger.error(f"Error retrieving docs: {e}")
        return []


COMPRESSED_RETRIEVER = False

def map_data(
    data,
    retriever,
    custom_data=False,
    output_file=None,
    llm_name="llama",
    topk=10,
    do_eval=False,
    is_omop_data=True,
    compressed_retriever=False,
):
    COMPRESSED_RETRIEVER = compressed_retriever  # to use global variable

    db = DataManager(DB_FILE, initial_json=MAPPING_FILE)
    global RETRIEVER_CACHE
    # start_time = time.time()
    # max_queries = len(data)  # 282
    results = []
    training_examples= []
    for _, item in enumerate(data):
        query_obj = item[1]
        # logger.info(f"Processing object: {query_obj}")
        # query_result = full_query_processing(
        #     query_text=query_obj,
        #     retriever=retriever,
        #     llm_name=llm_name,
        #     is_omop_data=is_omop_data,
        #     topk=topk,
        # )
        if query_obj:
            query_result, decomposed_query_object = full_query_processing_db(
                query_text=query_obj,
                retriever=retriever,
                llm_name=llm_name,
                is_omop_data=is_omop_data,
                topk=topk,
                datamanager=db,
            )
            if query_result:
                
                query_result = perform_mapping_eval_for_variable(
                    var_=query_result, llm_id=llm_name
                )
                results.append(query_result)
                query_result["variable label"] = decomposed_query_object.base_entity
                query_result["variable name"] = decomposed_query_object.name
                query_result["categorical"] = "|".join(decomposed_query_object.categories) if decomposed_query_object.categories else None
                query_result["visits"] = decomposed_query_object.visit if decomposed_query_object.visit else None
                query_result["additional entities"] = "|".join(decomposed_query_object.additional_entities) if decomposed_query_object.additional_entities else None
                query_result["units"] = decomposed_query_object.unit if query_obj.unit else None
                if query_result["prediction"].strip().lower() == "correct":
                    training_examples.append(
                        {
                            "input": query_obj,
                            "output": query_result
                        }
                    )
                    db.insert_many(convert_row_to_entities(query_result))
                logger.info(f"Query result after processing: {query_result}")
                
        else:
            logger.info(f"No query object found for item: {item}")
            results.append(create_processed_result())
        # time.sleep(0.05)  # Adjusted to be more appropriate than 0.0005

    # end_time = time.time()
    # total_time = end_time - start_time
    # save_results(results, output_file)
    # if do_eval:
    #     return evaluate_with_multiple_mappings(
    #         data[:max_queries], results, model_name=llm_name
    #     )
    # logger.info(
    #     f"Total execution time for {max_queries} queries is {total_time} seconds."
    # )
    add_result_to_training_data(training_examples, MAPPING_FILE)
    db.close_connection()
    return results


def full_query_processing_db(
    query_text: QueryDecomposedModel,
    retriever: Any,
    llm_name: str,
    topk: int,
    is_omop_data=True,
    datamanager: DataManager = None,
):
    try:
        logger.info(f"Processing query: {query_text}")

        if query_text is None:
            return None
        else:
            print(f"variable name ={query_text.full_query}")
            # results, mode = datamanager.query_variable(. It was inconsisten logic 
            #     query_text.original_label, var_name=query_text.name
            # )
            # if mode == "full" and len(results) >= 4:
            #     logger.info(f"Found results for {query_text} in RESERVOIR")
            #     return create_result_dict(results)

            query_decomposed = extract_information(query_text, llm_name)
            logger.info(f"Query decomposed:{query_decomposed}")
            processes_results = temp_process_query_details_db(
                llm_query_obj=query_decomposed,
                retriever_cache=retriever,
                llm_name=llm_name,
                topk=topk,
                original_query_obj=query_text,
                is_omop_data=is_omop_data,
                db=datamanager,
            )

            
            return processes_results, query_decomposed
    except Exception as e:
        logger.error(f"Error full processing query: {e}", exc_info=True)
        return {}, query_decomposed


def find_entity_in_db(entity_str:str, data_manager: DataManager) -> list:
        entity_str = entity_str.strip().lower() 
        result = data_manager.find_by_variable(entity_str)

        if result:
            logger.info(f"Found entity {result} by variable name:{entity_str}")
            return [convert_db_result(result)]
        else:
            result = data_manager.find_by_label(entity_str)
            logger.info(f"Found entity {result} by label:{entity_str}")
            if result:
                return [convert_db_result(result)]
            else:
                logger.info(f"No entity found for:{entity_str}")
                return []

def temp_process_query_details_db(
    llm_query_obj: QueryDecomposedModel,
    retriever_cache: Any,
    llm_name: str,
    topk: int,
    original_query_obj: QueryDecomposedModel,
    is_omop_data=False,
    db: DataManager = None,
):
    try:
        if llm_query_obj:
            original_query_obj.domain = llm_query_obj.domain
            original_query_obj.rel = llm_query_obj.rel
            original_query_obj.unit = llm_query_obj.unit
            original_query_obj.categories = llm_query_obj.categories
            original_query_obj.visit = llm_query_obj.visit 
            logger.info(f"original query Obj in temp process:{original_query_obj}")
            (
                variable_label_matches,
                additional_entities_matches,
                categories_matches,
                unit_matches,
                visit_matches
            ) = None, None, None, None, None
            base_entity = llm_query_obj.base_entity
            # domain = original_query_obj.domain
            
            # base_entity_in_db = find_entity_in_db(base_entity, db)
            # if len(base_entity_in_db) > 0:
            #     variable_label_matches, found_match = (
            #         base_entity_in_db, True
            #     )
            # else:
            #     domain = "all"
            #         variable_label_matches, found_match = process_retrieved_docs(
            #             base_entity,
            #             retriever_docs(
            #                 original_query_obj.base_entity,
            #                 retriever_cache,
            #                 domain=domain,
            #                 is_omop_data=is_omop_data,
            #                 topk=topk,
            #             ),
            #             llm_name,
            #             domain,
            #             belief_threshold=0.85,
            #         )
            # if found_match and len(variable_label_matches) > 0:
            #     main_term = (
            #         base_entity  # Assign base_entity to main_term if a match is found
            #     )
            #     llm_query_obj = original_query_obj
            # else:
            # first check if the base_entity is in the RESERVOIR
            main_term = base_entity
            base_entity_in_db = find_entity_in_db(base_entity, db)
            if len(base_entity_in_db) > 0:
                variable_label_matches = base_entity_in_db
                print("found base entity in RESERVOIR")
          
            else:
                    # else proceed to RAG based search 
                    print("proceeding to structured completion format")
                    main_term = (
                        llm_query_obj.base_entity if llm_query_obj else base_entity
                    )
                    variable_label_matches, _ = process_retrieved_docs(
                        main_term,
                        retriever_docs(
                            main_term,
                            retriever_cache,
                            domain=llm_query_obj.domain,
                            is_omop_data=is_omop_data,
                            topk=topk,
                        ),
                        llm_name,
                        llm_query_obj.domain,
                    )
                    # llm_query_obj.name = original_query_obj.name
            # additional_entities = llm_query_obj.additional_entities
            # categories = llm_query_obj.categories
            # domain = llm_query_obj.domain
            # unit = llm_query_obj.unit
            rel = llm_query_obj.rel
            logger.info(
                f"main_term={main_term}, context={llm_query_obj.additional_entities}, categories={llm_query_obj.categories}, domain={llm_query_obj.domain}, unit={llm_query_obj.unit}"
            )
            if llm_query_obj.additional_entities:
                logger.info(f"Processing additional entities: {llm_query_obj.additional_entities}")
                # first check if the additional entities are in the RESERVOIR
                
                additional_entities_matches = (
                    process_values_db(
                        main_term,
                        llm_query_obj.additional_entities,
                        retriever_cache,
                        llm_name,
                        domain="observation",
                        values_type="additional",
                        is_omop_data=is_omop_data,
                        topk=topk,
                        db=db,
                    )
                    if llm_query_obj.additional_entities
                    else {}
                )
            if llm_query_obj.categories:
                logger.info(f"Processing categories: {llm_query_obj.categories}")
                categories_matches = (
                    process_values_db(
                        main_term,
                        llm_query_obj.categories,
                        retriever_cache,
                        llm_name,
                        domain="all",
                        values_type="categories",
                        is_omop_data=is_omop_data,
                        topk=topk,
                        db=db,
                    )
                    if llm_query_obj.categories
                    else {}
                )
            if llm_query_obj.visit:
                logger.info(f"Processing visit: {llm_query_obj.visit}")
                visit_matches = process_values_db(
                        main_term,
                        [llm_query_obj.visit],
                        retriever_cache,
                        llm_name,
                        domain="visit",
                        values_type="visit",
                        is_omop_data=is_omop_data,
                        topk=topk,
                        db=db,
                    
                    
                )  # its dict
                if visit_matches:
                    logger.info(f"Visit matches found: {visit_matches[llm_query_obj.visit]}")
                    visit_matches = visit_matches[llm_query_obj.visit] if llm_query_obj.visit in visit_matches else []
                    # logger.info(f"Visit matches after processing: {visit_matches}")
                
            # if rel:
            #     rel_docs = process_values([rel], retriever_cache, llm_name, domain='all', values_type='rel')[rel] if rel else []
            if llm_query_obj.unit:
                logger.info(f"Processing unit: {llm_query_obj.unit}")
                unit_matches = (
                    process_unit_db(
                        llm_query_obj.unit,
                        retriever_cache,
                        llm=llm_name,
                        domain="unit",
                        is_omop_data=is_omop_data,
                        topk=topk,
                        db=db,
                    )
                    if llm_query_obj.unit
                    else []
                )
            mapping_result = create_processed_result(
                ProcessedResultsModel(
                    variable_name=llm_query_obj.name,
                    base_entity=main_term,
                 
                    base_entity_matches=variable_label_matches,
                    categories=llm_query_obj.categories,
                    categories_matches=categories_matches,
                    unit=llm_query_obj.unit,
       
                    unit_matches=unit_matches,
                    original_query=llm_query_obj.original_label,
                    additional_entities=llm_query_obj.additional_entities,
                    primary_to_secondary_rel=rel,
                    additional_entities_matches=additional_entities_matches,
                    domain= llm_query_obj.domain,
                    visit_matches=visit_matches
                )
            )
            logger.info(f"Mapping result temp process query details db= {mapping_result}")
            return mapping_result
    except Exception as e:
        logger.error(f"Error full processing query: {e}", exc_info=True)
        return {}




def process_retrieved_docs(
    query, docs, llm_name=None, domain=None, belief_threshold=0.8
):
    # logger.info_docs(docs)
    if docs and len(docs) > 0:
        if matched_docs := exact_match_found(
            query_text=query, documents=docs, domain=domain
        ):
            return post_process_candidates(matched_docs, max=1), True
        if llm_name:
            logger.info(f"No string match found for {query} pass to {llm_name}")
            domain_specific_docs = filter_irrelevant_domain_candidates(docs, domain)
            if domain_specific_docs is None or len(domain_specific_docs) == 0:
                return [], False
            llm_ranks, match_found = pass_to_chat_llm_chain(
                query,
                domain_specific_docs,
                llm_name=llm_name,
                domain=domain,
                threshold=belief_threshold,
            )
            if llm_ranks and len(llm_ranks) > 0:
                print(
                    f"number of candidates={llm_ranks} and exact match={match_found}"
                )
                return post_process_candidates(llm_ranks, max=1), match_found
            else:
                return [], False
        else:
            return docs, False
    else:
        logger.info(f"No docs found for query={query}")
    return [], False


def process_context(context, retriever, llm, domain=None, topk=10):
    # context = normalize(context)
    if context:
        if docs := retriever_docs(context, retriever, domain="all", topk=topk):
            if matched_docs := exact_match_found(
                query_text=context, documents=docs, domain=domain
            ):
                return post_process_candidates(matched_docs, max=1)
            llm_results, _ = pass_to_chat_llm_chain(
                context, docs, llm_name=llm, domain=domain
            )
            return post_process_candidates(llm_results, max=2)
    return []


def process_values_db(
    main_term,
    values,
    retriever,
    llm,
    domain=None,
    values_type="additional",
    is_omop_data=False,
    topk=10,
    db: DataManager = None,
):
    if isinstance(values, str):
        values = [values]
    logger.info(f"processing values={values}")
    all_values = {}
    if not values:
        return all_values
    for q_value in values:
        # q_value = str(q_value).strip().lower()
        if q_value:
            result = find_entity_in_db(q_value, db)
            if len(result) > 0:
                logger.info(f"found value in RESERVOIR={q_value} = {result}")
                all_values[q_value] = result
            else:
                contextaware_value_results = []
                value_results = retriever_docs(
                    q_value,
                    retriever,
                    domain=domain,
                    is_omop_data=is_omop_data,
                    topk=topk,
                )
                if values_type == "additional":
                    contextaware_value_results = retriever_docs(
                        f"{q_value},{main_term}",
                        retriever,
                        domain=domain,
                        is_omop_data=is_omop_data,
                        topk=topk,
                    )[:5]
                if value_results:
                    value_results += contextaware_value_results
                    # pretty_print_docs(categorical_value_results)
                    if matched_docs := exact_match_found(
                        query_text=q_value,
                        documents=value_results,
                        domain=domain,
                    ):
                        # max_results = 2 if ('or' in q_value or 'and' in q_value) else 1
                        logger.info(f"Exact Matched docs for {q_value}: {matched_docs}")
                        all_values[q_value] = post_process_candidates(
                            matched_docs, max=1
                        )
                    elif (
                        value_results and len(value_results) > 0
                    ):
                        if values_type == "additional":
                            q_value_ = f"{q_value}, context: {main_term}"
                        else:
                            q_value_ = q_value
                        updated_results, _ = pass_to_chat_llm_chain(
                            q_value_,
                            value_results,
                            llm_name=llm,
                            domain=domain,
                        )
                        if updated_results:
                            # max_results = 2 if ('or' in q_value or 'and' in q_value) else 1
                            # logger.info(f"max_results={max_results} for {updated_q_value}")
                            all_values[q_value] = post_process_candidates(
                                updated_results, max=1
                            )
                        else:
                            if values_type == "categories":
                                all_values[q_value] = [
                                    RetrieverResultsModel(
                                        label="na",
                                        code="na",
                                        omop_id="na",
                                        vocab=None,
                                    )
                                ]
        # elif values_type == "categories":
        #     all_values[q_value] = [
        #         RetrieverResultsModel(
        #             label="na",
        #             code="na",
        #             omop_id="na",
        #             vocab="na",
        #         )
        #     ]
        
        else:
            all_values[q_value] = [
                                    RetrieverResultsModel(
                                        label="na",
                                        code="na",
                                        omop_id="na",
                                        vocab=None,
                                    )
                                ]
    return all_values

def process_unit_db(
    unit,
    retriever,
    llm: Any,
    domain: str,
    is_omop_data: bool = False,
    topk: int = 10,
    db: DataManager = None,
):
    unit_results = []
    if unit and unit != "unknown":
        result  = find_entity_in_db(unit, db)
        if len(result) > 0:
            logger.info(f"Found unit in RESERVOIR: {unit} = {result}")
            unit_results = result
        else:
            unit_results = retriever_docs(
                unit, retriever, domain="unit", is_omop_data=is_omop_data, topk=topk
            )
            if unit_results:
                exact_units = exact_match_found(unit, unit_results, domain="unit")
                if len(exact_units) > 0:
                    unit_results = post_process_candidates(exact_units, max=1)
                elif len(unit_results) > 0:
                    llm_results, _ = pass_to_chat_llm_chain(
                        unit, unit_results, llm_name=llm, domain=domain
                    )
                    unit_results = post_process_candidates(llm_results, max=1)
                else:
                    unit_results = [
                        RetrieverResultsModel(
                            label="na",
                            code=None,
                            omop_id=None,
                            vocab=None,
                        )
                    ]

    return unit_results




def filter_results(query, results):
    # pretty_print_docs(results)
    prioritized = []
    non_prioritized = []
    seen_metadata = []  # Use a list to track seen metadata and preserve order
    query = query.strip().lower()  # Normalize the query for comparison

    # First pass: collect prioritized and non-prioritized results
    for res in results:
        label = (
            res.metadata["label"].strip().lower()
        )  # Normalize the label for comparison
        metadata_str = f"{label}|{res.metadata['vocab']}|{res.metadata['scode']}|{res.metadata['sid']}"

        # Check if metadata has already been seen
        if metadata_str not in seen_metadata:
            seen_metadata.append(metadata_str)  # Mark metadata as seen

            if label == query:
                prioritized.append(
                    res
                )  # Add to prioritized list if label matches the query
            else:
                non_prioritized.append(
                    res
                )  # Add to non-prioritized list if label does not match

    # Combine prioritized and non-prioritized lists while preserving their original order
    combined_results = prioritized + non_prioritized
    # print("unique docs")
    # pretty_print_docs(combined_results)
    return combined_results
