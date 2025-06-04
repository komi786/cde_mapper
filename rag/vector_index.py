# !/usr/bin/env python3
from rag.bi_encoder import SAPEmbeddings
from rag.embeddingfilter import MyEmbeddingsFilter
from rag.compress import CustomCompressionRetriever, CustomMergeRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import json

# import random
from rag.utils import global_logger as logger
from rag.data_loader import load_data
from rag.manager import LLMManager

# from langchain.schema import Document
import argparse
from rag.utils import (
    load_docs_from_jsonl,
    load_custom_docs_from_jsonl,
    evaluate_topk_acc,
    evaluate_ncgd,
)

from langchain_qdrant import FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
import qdrant_client.http.models as rest
from rag.param import VECTOR_PATH, QDRANT_PORT, EMB_MODEL_NAME, LLM_ID
import time
from rag.qdrant import CustomQdrantVectorStore
from rag.llm_chain import pass_to_chat_llm_chain
from rag.athena_api_retriever import RetrieverAthenaAPI, AthenaFilters
from langchain_community.retrievers import BM25Retriever


def create_bm25_sparse_retriever(docs_file):
    docs = load_custom_docs_from_jsonl(docs_file)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10
    return bm25_retriever


def filter_results(query, results):
    prioritized = []
    non_prioritized = []
    seen_labels = set()

    for res in results:
        label = res.metadata["label"]
        # is_standard = res.metadata.get('is_standard', None)
        if query.lower() in label.lower():
            if label not in seen_labels:
                seen_labels.add(label)

            else:
                non_prioritized.append(res)
        else:
            non_prioritized.append(res)
    docs = prioritized + non_prioritized
    logger.info(f"Prioritized: {[res.metadata['label'] for res in prioritized]}")
    return docs


def get_collection_vectors(client, collection_name):
    try:
        collection_status = client.get_collection(collection_name)
        vectors_count = collection_status.points_count
        logger.info(f"{collection_name} has {vectors_count} vectors.")
        return vectors_count
    except Exception as e:
        logger.info(f"Error fetching collection {collection_name} from Qdrant: {e}")
        return 0


def _create_payload_index(client, collection_name):
    # client.payload_index_exists(collection_name=collection_name, field_name="metadata.label")
    # client.create_payload_index(
    #                 collection_name=collection_name,
    #                 field_schema= "keyword",
    #                 field_name="metadata.label",
    #                 wait = True
    #                 )
    client.create_payload_index(
        collection_name=collection_name,
        field_schema="keyword",
        field_name="metadata.domain",
        wait=True,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_schema="keyword",
        field_name="metadata.vocab",
        wait=True,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_schema="keyword",
        field_name="metadata.is_standard",
        wait=True,
    )
    # client.create_payload_index(
    #                 collection_name=collection_name,
    #                 field_schema= "keyword",
    #                 field_name="metadata.concept_class",
    #                 wait=True
    #             )

def check_collection_exists(collection_name:str):
    """Check if a collection exists in Qdrant."""
    client = QdrantClient(url=VECTOR_PATH, port=QDRANT_PORT, https=True, timeout=300)
    try:
        exists = client.collection_exists(collection_name)
        logger.info(f"Collection '{collection_name}' exists: {exists}")
        return exists
    except Exception as e:
        logger.error(f"Error checking collection existence: {e}")
        return False

def generate_vector_index(
    dense_embedding,
    sparse_embedding=None,
    url=VECTOR_PATH,
    port=QDRANT_PORT,
    docs_file="/Users/komalgilani/Desktop/cde_mapper/data/output/concepts.jsonl",
    mode="inference",
    collection_name="concept_mapping",
    topk=10,
):
    if sparse_embedding is None:
        # Qdrant/bm42-all-minilm-l6-v2-attentions
        sparse_embedding = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")
    client = QdrantClient(url=url, port=port, https=True, timeout=300)
    # client = QdrantClient(":memory:")
    logger.info(f"collection exist: {client.collection_exists(collection_name)}")
    if client.collection_exists(collection_name):
        vector_count = get_collection_vectors(client, collection_name=collection_name)
    else:
        vector_count = 0
    if vector_count == 0 or mode == "recreate":
        docs = load_docs_from_jsonl(docs_file) if mode == "recreate" else None
        logger.info(f"Docs: {len(docs)}")
        client.delete_collection(collection_name=collection_name)
        vector_store = CustomQdrantVectorStore.from_documents(
            docs,
            embedding=dense_embedding,
            batch_size=64,
            url=url,
            port=port,
            https=True,
            vector_name="omop_dense_vector",
            sparse_vector_name="omop_sparse_vector",
            sparse_embedding=sparse_embedding,
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.DENSE,
            vector_params={
                "size": 768,
                "distance": Distance.COSINE,
                # "hnsw_config": rest.HnswConfigDiff(
                #     m=32,
                #     ef_construct=48,
                #     full_scan_threshold=20000,
                #     max_indexing_threads=8,
                #     payload_m=32,
                # ),
                # "quantization_config": rest.ScalarQuantization(
                #     scalar=rest.ScalarQuantizationConfig(
                #         type=rest.ScalarType.INT8,
                #         quantile=0.99,
                #         always_ram=True,
                #     ),
                # ),
                "on_disk": True,
                # "on_disk_payload": True,
            },
            sparse_vector_params={
                "modifier": rest.Modifier.IDF,
                "index": {"full_scan_threshold": 20000, "on_disk": True},
            },
            force_recreate=True,
        )
        _create_payload_index(client, collection_name)

    else:
        # _create_payload_index(client, collection_name)
        vector_store = CustomQdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            vector_name="omop_dense_vector",
            sparse_vector_name="omop_sparse_vector",
            distance=rest.Distance.COSINE,
            retrieval_mode=RetrievalMode.DENSE,
            validate_collection_config=True,
        )
        if mode == "update":
            docs = load_custom_docs_from_jsonl(docs_file)
            if vector_count > 0 and vector_count < len(docs):
                vector_store.add_documents(docs)
                vcount = get_collection_vectors(
                    client=client, collection_name=collection_name
                )
                logger.info(f"Added {vcount - vector_count} vectors to collection")

    # similarity_score_threshold, 'fetch_k':100},
    #                       search_kwargs={'k': 10,'lambda_mult': 0.4},
    #     docsearch.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={'k': 6, 'lambda_mult': 0.25}

    # )
    # return vector_store.as_retriever(
    #             search_type="mmr",
    #             search_kwargs={'k':10, 'lambda_mult': 0.3}
    #         )

    return vector_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    )

    # return vector_store.as_retriever(search_kwargs={"k": topk})


def initiate_api_retriever(compress=True):
    #'LOINC','UCUM','OMOP Extension','ATC','RxNorm','Gender','Race','Ethnicity', excluded for BC5CDR-D and NCBI
    # FOR AAP datset: 'SNOMED','MeSH','MedDRA','LOINC
    # domain=['Condition','Observation']
    athena_api_retriever = RetrieverAthenaAPI(
        filters=AthenaFilters(
            vocabulary=[
                "SNOMED",
                "LOINC",
                "UCUM",
                "OMOP Extension",
                "ATC",
                "RxNorm",
                "Gender",
                "Race",
                "Ethnicity",
            ],
            standard_concept=["Standard", "Classification"],
        ),
        k=15,
    )
    if compress:
        athena_api_retriever = set_compression_retriever(athena_api_retriever)
    return athena_api_retriever


def initiate_api_retriever_all_concepts():
    #'LOINC','UCUM','OMOP Extension','ATC','RxNorm','Gender','Race','Ethnicity', excluded for BC5CDR-D and NCBI
    # FOR AAP datset: 'SNOMED','MeSH','MedDRA','LOINC
    # domain=['Condition','Observation']
    athena_api_retriever = RetrieverAthenaAPI(
        filters=AthenaFilters(
            vocabulary=[
                "SNOMED",
                "LOINC",
                "UCUM",
                "OMOP Extension",
                "ATC",
                "RxNorm",
                "Gender",
                "Race",
                "Ethnicity",
                "MeSH",
            ]
        ),
        k=15,
    )
    compression_retriever = set_compression_retriever(athena_api_retriever)
    return compression_retriever


def update_api_search_filter(api_retriever, domain="observation", topk=10):
    if domain == "unit":
        api_retriever.filters = AthenaFilters(
            domain=None, vocabulary=["UCUM"], standard_concept=["Standard"]
        )
    elif domain == "condition" or domain == "anatomic site":
        api_retriever.filters = AthenaFilters(
            domain=["Condition", "Meas Value", "Spec Anatomic Site"],
            vocabulary=["SNOMED"],
            standard_concept=["Standard"],
        )
    elif domain == "measurement":
        api_retriever.filters = AthenaFilters(
            domain=["Measurement", "Meas Value", "Observation", "Spec Anatomic Site"],
            vocabulary=["LOINC", "SNOMED"],
            standard_concept=["Standard"],
        )
    elif domain == "drug":
        api_retriever.filters = AthenaFilters(
            domain=["Drug"],
            vocabulary=["RxNorm", "ATC", "SNOMED"],
            standard_concept=["Standard", "Classification"],
        )
    elif domain == "observation":
        api_retriever.filters = AthenaFilters(
            domain=["Observation", "Meas Value"],
            vocabulary=["SNOMED", "LOINC", "OMOP Extension"],
        )
    elif domain == "visit":
        api_retriever.filters = AthenaFilters(
            domain=["Visit", "Observation"],
            vocabulary=["SNOMED", "LOINC", "OMOP Extension"],
        )
    elif domain == "demographics":
        api_retriever.filters = AthenaFilters(
            domain=["Observation", "Meas Value"],
            vocabulary=[
                "SNOMED",
                "LOINC",
                "OMOP Extension",
                "Gender",
                "Race",
                "Ethnicity",
            ],
        )
    else:
        api_retriever.filters = AthenaFilters(
            vocabulary=["SNOMED", "LOINC", "OMOP Extension", "RxNorm", "ATC"],
            standard_concept=["Standard", "Classification"],
        )
    api_retriever.k = topk
    return api_retriever


def update_qdrant_search_filter(retriever, domain="all", topk=10):
    if domain == "unit":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.vocab", match=rest.MatchValue(value="ucum")
                ),
                rest.FieldCondition(
                    key="metadata.is_standard", match=rest.MatchAny(any=["S"])
                ),
            ]
        )
    elif domain == "condition" or domain == "anatomic site":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.vocab",
                    match=rest.MatchAny(any=["snomed", "cancer modifier"]),
                ),
                rest.FieldCondition(
                    key="metadata.is_standard", match=rest.MatchAny(any=["S"])
                ),
            ]
        )
    elif domain == "demographics":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.vocab", match=rest.MatchAny(any=["snomed", "loinc"])
                ),
                rest.FieldCondition(
                    key="metadata.domain",
                    match=rest.MatchAny(any=["observation", "meas value"]),
                ),
                rest.FieldCondition(
                    key="metadata.is_standard", match=rest.MatchAny(any=["S"])
                ),
            ]
        )
    elif domain == "measurement":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.vocab", match=rest.MatchAny(any=["loinc", "snomed"])
                ),
                rest.FieldCondition(
                    key="metadata.domain",
                    match=rest.MatchAny(
                        any=[
                            "measurement",
                            "meas value",
                            "observation",
                            "spec anatomic site",
                        ]
                    ),
                ),
                rest.FieldCondition(
                    key="metadata.is_standard", match=rest.MatchAny(any=["S"])
                ),
            ]
        )
    elif domain == "drug":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.vocab",
                    match=rest.MatchAny(
                        any=["rxnorm", "rxnorm extension", "atc", "snomed"]
                    ),
                ),
                rest.FieldCondition(
                    key="metadata.is_standard", match=rest.MatchAny(any=["S", "C"])
                ),
            ]
        )
    elif (
        domain == "observation"
        or domain == "visit"
        or domain == "demographics"
        or domain == "history of event"
        or domain == "history of events"
        or domain == "life style"
    ):
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.vocab",
                    match=rest.MatchAny(any=["snomed", "loinc", "omop extension"]),
                ),
                rest.FieldCondition(
                    key="metadata.domain",
                    match=rest.MatchAny(any=["observation", "meas value"]),
                ),
                rest.FieldCondition(
                    key="metadata.is_standard", match=rest.MatchAny(any=["S"])
                ),
            ]
        )
    elif domain == "procedure":
        retriever.search_kwargs["filter"] = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.vocab",
                    match=rest.MatchAny(any=["snomed", "icd9proc"]),
                )
            ]
        )
    else:
        # print("No domain specified")
        retriever.search_kwargs["filter"] = rest.Filter(
            #  must=[
            #            rest.FieldCondition(
            #                 key="metadata.vocab",
            #                  match=rest.MatchExcept(**{"except": ["meddra","icd10","icd9","icd10cm","icd9cm","uk biobank","mesh"]}),
            #             )
            #         ])
            must=[
                rest.FieldCondition(
                    key="metadata.vocab",
                    match=rest.MatchAny(
                        any=["snomed", "loinc", "atc", "rxnorm extension", "ucum"]
                    ),
                ),
                rest.FieldCondition(
                    key="metadata.is_standard", match=rest.MatchAny(any=["S", "C"])
                ),
            ]
        )
    retriever.search_kwargs["k"] = topk
    return retriever


def retriever_doc(vector_store, query, top_k=5):
    results = vector_store.invoke(query)
    for res in results:
        logger.info(f"*{res.metadata['label']}---[{res.metadata['domain']}]")


def print_results(results):
    results = {res.metadata["label"]: res for res in results}
    for res in results.values():
        if "domain" in res.metadata:
            logger.info(
                f"*{res.metadata['label']}---[{res.metadata['domain']}---[{res.metadata['sid']}----[{res.metadata['is_standard']}]"
            )
        else:
            logger.info(f"*{res.metadata['label']}")


# def hybrid_only_retriever(query,domain,hybrid_vector_retriever,topk=5,llm_name='llama') -> list:
#     vector_retriever = update_qdrant_search_filter(hybrid_vector_retriever, domain=domain)
#     vector_retriever = set_compression_retriever(hybrid_vector_retriever)
#     results = vector_retriever.invoke(query)
#    # Prioritize results with 'is_standard' set to 'S' or 'C'
#     unique_results = filter_results(query, results)
#     logger.info(f"Unique Results:\n {[res.metadata['label'] for res in unique_results]}")
#     # unique_results, _ = pass_to_chat_llm_chain(query, unique_results, llm_name=llm_name)
#     return unique_results


# def set_ensemble_retrievers(retrievers, weights=[0.55, 0.45]):
#     ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
#     return ensemble_retriever


def set_merger_retriever(retrievers):
    ensemble_retriever = CustomMergeRetriever(retrievers=retrievers)
    return ensemble_retriever


def update_compressed_merger_retriever(
    merger_retriever: CustomCompressionRetriever, domain="all", topk=10
) -> CustomCompressionRetriever:
    try:
        retrievers = merger_retriever.base_retriever.retrievers
        logger.info(f"retrievers: {retrievers}")
        dense_retriever = update_qdrant_search_filter(
            retrievers[0], domain=domain, topk=topk
        )
        api_retriever = update_api_search_filter(
            retrievers[1], domain=domain, topk=topk
        )
        merger_retriever = CustomMergeRetriever(
            retrievers=[dense_retriever, api_retriever]
        )
        return set_compression_retriever(merger_retriever)
    except Exception as e:
        logger.info(f"Error updating merger retriever: {e}")
        return merger_retriever


def update_merger_retriever(
    merger_retriever: CustomCompressionRetriever, domain="all", topk=10
) -> CustomCompressionRetriever:
    try:
        retrievers = merger_retriever.retrievers
        api_retriever = update_api_search_filter(
            retrievers[1], domain=domain, topk=topk
        )
        dense_retriever = update_qdrant_search_filter(
            retrievers[0], domain=domain, topk=topk
        )
        merger_retriever = CustomMergeRetriever(
            retrievers=[dense_retriever, api_retriever]
        )
        return merger_retriever
    except Exception as e:
        logger.info(f"Error updating merger retriever: {e}")
        return merger_retriever


# def update_ensemble_ret_search_filter(ensemble_retriever, domain='all'):

#     #api_retriever is wrapper around compression_retriever
#     api_retriever = update_api_search_filter(ensemble_retriever.retrievers[1].base_retriever, domain=domain)
#     api_retriever = set_compression_retriever(api_retriever)
#     dense_retriever = update_qdrant_search_filter(ensemble_retriever.retrievers[0], domain=domain)

#     ensemble_retriever.retrievers = [dense_retriever,api_retriever]
#     return ensemble_retriever


# def run_ensemble_retrievers(domain='all',retriever=None) -> list:
#     eRetriever=update_ensemble_ret_search_filter(ensemble_retriever=retriever, domain=domain)
#     results=eRetriever.invoke("red blood cell count in blood")[:5]
#     return filter_results(results)


def set_compression_retriever(base_retriever) -> CustomCompressionRetriever:
    embedding = SAPEmbeddings()
    embeddings_filter = MyEmbeddingsFilter(
        embeddings=embedding, similarity_threshold=0.5
    )
    # llm = LLMManager.get_instance("llama3.1")
    # compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = CustomCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=base_retriever
    )
    return compression_retriever


def log_accuracy(
    correct_dict, input_file="/Users/komalgilani/Desktop/cde_mapper/data/eval_datasets/accuracy.txt"
):
    with open(input_file, "a") as f:
        for matric, value in correct_dict.items():
            f.write(f"Metric {matric}: {value}\n")


# def evaluate_metrics(correct_dict, mrr_sum, exact_founds, total_retrieved, max_queries):
#     """Calculate and return evaluation metrics including MRR, Precision, Recall, and Accuracy at various ranks."""
#     mrr = mrr_sum / max_queries if max_queries > 0 else 0
#     recall = exact_founds / max_queries if max_queries > 0 else 0
#     precision = exact_founds / total_retrieved if total_retrieved > 0 else 0

#     metrics = {
#         "mrr": mrr,
#         "recall": recall,
#         "precision": precision,
#         "acc_at_1": correct_dict[1] / max_queries if max_queries > 0 else 0,
#         "acc_at_3": correct_dict[3] / max_queries if max_queries > 0 else 0,
#         "acc_at_5": correct_dict[5] / max_queries if max_queries > 0 else 0,
#         "acc_at_10": correct_dict[10] / max_queries if max_queries > 0 else 0,
#     }
#     return metrics


def write_results_to_file(results, query, codes_set, f):
    """Write retrieval results to the output file."""
    if results:
        for res in results:
            domain = res.metadata.get("domain", "unknown")
            label = res.metadata.get("label", "unknown")
            sid = res.metadata.get("sid", "unknown")
            f.write(f"{query}\t{label}\t{domain}\t{sid}\n")
    else:
        f.write(f"{query}\tNo results found\n")


def evaluate_metrics(correct_dict, mrr_sum, max_queries, recall_dict, precision_dict):
    """Calculate and return evaluation metrics including MRR, Recall, Precision, and Accuracy at k = 1, 3, 5, 10."""
    mrr = mrr_sum / max_queries if max_queries > 0 else 0
    # recall = exact_founds / max_queries if max_queries > 0 else 0

    metrics = {
        "mrr": mrr,
        "acc_at_1": correct_dict[1] / max_queries if max_queries > 0 else 0,
        "acc_at_3": correct_dict[3] / max_queries if max_queries > 0 else 0,
        "acc_at_5": correct_dict[5] / max_queries if max_queries > 0 else 0,
        "acc_at_10": correct_dict[10] / max_queries if max_queries > 0 else 0,
        "precision_at_1": precision_dict[1] / max_queries if max_queries > 0 else 0,
        "precision_at_3": precision_dict[3] / max_queries if max_queries > 0 else 0,
        "precision_at_5": precision_dict[5] / max_queries if max_queries > 0 else 0,
        "precision_at_10": precision_dict[10] / max_queries if max_queries > 0 else 0,
        "recall_at_1": recall_dict[1] / max_queries if max_queries > 0 else 0,
        "recall_at_3": recall_dict[3] / max_queries if max_queries > 0 else 0,
        "recall_at_5": recall_dict[5] / max_queries if max_queries > 0 else 0,
        "recall_at_10": recall_dict[10] / max_queries if max_queries > 0 else 0,
    }
    return metrics


# def process_queries(ensemble_retriever, queries, max_queries, args):
#     """Main function to process queries and output in a format compatible with evaluation functions."""
#     data = {"queries": []}

#     # with open(args.output_file, "w") as f:
#     for query in queries[:max_queries]:
#         match_found = False
#         llm_found_match = False
#         if len(query) == 3:
#             code, query_text, _ = query[0], query[1].base_entity, query[1].domain
#         else:
#             code, query_text = query[0], query[1].base_entity

#         codes_list = [str(c).strip().lower() for c in code.split("|")]
#         results = ensemble_retriever.invoke(query_text)
#         exact_results, len_matched = exact_match_found_no_vocab(query_text, results)
#         if exact_results:
#             results = exact_results  # Use exact results if available
#             match_found = True
#         if args.use_llm and not match_found:
#             print(
#                 f"Query: {query}--\nResults: {[res.metadata['label'] for res in results]}"
#             )
#             results, llm_found_match = pass_to_chat_llm_chain(
#                 query,
#                 results,
#                 llm_name=llm_id,
#                 prompt_stage=args.prompt_stage,
#                 domain="all",
#                 in_context=True,
#             )
#         # Transform results into the expected candidate format for each mention
#         candidates = []
#         for idx, res in enumerate(results):
#             sid = str(res.metadata.get("sid", "")).strip().lower()
#             is_relevant = 0
#             if match_found and len_matched > 0:
#                 if idx < len_matched:  # Only assign relevant up to len_matched
#                     is_relevant = 1
#             elif any(sid_part in codes_list for sid_part in sid.split("|")):
#                 is_relevant = 1
#             if idx == 0 and llm_found_match:
#                 # Ensure the top candidate is marked as relevant if LLM found a match
#                 is_relevant = 1
#             if is_relevant:
#                 logger.info(f"found relevant: {query_text}---{res.metadata['label']}")

#             candidate = {"label": is_relevant, "id": sid}
#             candidates.append(candidate)
#         mentions = [
#             {"candidates": candidates, "mention": query_text, "golden_cui": codes_list}
#         ]
#         data["queries"].append({"mentions": mentions})
#     data = evaluate_topk_acc(data)
#     data = evaluate_ncgd(data)
#     # write data to file
#     with open(args.output_file, "w") as f:
#         json.dump(data, f, indent=2)

#     return data


if __name__ == "__main__":
    
    check_collection_exists("concept_mapping_1")
    # check if qdrant collection exists
    
    # start_time = time.time()
    # parser = argparse.ArgumentParser(description="Load Vector Store")
    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     default=EMB_MODEL_NAME,
    #     help="Model identifier for embeddings",
    # )
    # parser.add_argument(
    #     "--llm_id", type=str, default=LLM_ID, help="Model identifier for embeddings"
    # )
    # parser.add_argument(
    #     "--mode", type=str, default="inference", help="The mode to run the model in"
    # )
    # parser.add_argument(
    #     "--compress", action="store_true", help="Use compression retriever"
    # )
    # parser.add_argument(
    #     "--collection_name",
    #     type=str,
    #     default="concept_mapping",
    #     help="Generate vector index for given collection",
    # )
    # parser.add_argument(
    #     "--document_file_path",
    #     type=str,
    #     default="/Users/komalgilani/Desktop/cde_mapper/data/output/concepts.jsonl",
    #     help="Documents to index",
    # )
    # parser.add_argument(
    #     "--dataset_name",
    #     type=str,
    #     default="icare4cvd",
    #     help="Dataset name for evaluation",
    # )
    # parser.add_argument(
    #     "--input_data",
    #     type=str,
    #     default="/Users/komalgilani/Desktop/cde_mapper/data/eval_datasets/custom_data/references.txt",
    #     help="Documents to index",
    # )
    # parser.add_argument(
    #     "--output_file",
    #     type=str,
    #     default="/Users/komalgilani/Desktop/cde_mapper/data/eval_datasets/results.txt",
    #     help="Documents to index",
    # )
    # parser.add_argument(
    #     "--prompt_stage", type=int, default=1, help="Prompt stage for LLM processing"
    # )
    # parser.add_argument("--use_llm", action="store_true", help="Use LLM for filtering")
    # args = parser.parse_args()
    # mode = args.mode
    # model_name = args.model_name
    # llm_id = args.llm_id
    # embeddings = SAPEmbeddings(model_id=model_name)
    # # in concept_mapping we used Qdrant/bm42-all-minilm-l6-v2-attentions
    # # in icare4cvd mapping we used prithivida/Splade_PP_en_v1
    # sparse_embeddings = FastEmbedSparse(
    #     model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
    # )
    # print(f"doc file path: {args.document_file_path}")
    # # if args.compress:
    # #     topk = 10
    # # else:
    # #     topk = 5

    # topk = 10
    # hybrid_vector_retriever = generate_vector_index(
    #     embeddings,
    #     sparse_embeddings,
    #     docs_file=args.document_file_path,
    #     mode=mode,
    #     collection_name=args.collection_name,
    #     topk=topk,
    # )
    # # hybrid_vector_retriever = update_qdrant_search_filter(hybrid_vector_retriever)

    # # bm25_sparse_retriever = set_compression_retriever(bm25_sparse_retriever)
    # # docs = load_docs_from_jsonl(args.document_file_path)
    # # weaivate_vector = faiss_vector_store(collection_name=args.collection_name ,docs_file=args.document_file_path,embeddings=embeddings)

    # # vector_search = VectorSearch(embedding=embeddings, documents=docs, collection_name=args.collection_name,
    # #                              topk=10)
    # # vector_search.create_qdrant_index()
    # if args.dataset_name == "icare4cvd":
    #     api_retriever = initiate_api_retriever()
    #     bm25_sparse_retriever = set_compression_retriever(api_retriever)
    # else:
    #     bm25_sparse_retriever = create_bm25_sparse_retriever(args.document_file_path)
    #     # api_retriever = initiate_api_retriever_all_concepts()

    # ensemble_retriever = CustomMergeRetriever(
    #     retrievers=[hybrid_vector_retriever, bm25_sparse_retriever]
    # )
    # if args.compress:
    #     ensemble_retriever = set_compression_retriever(ensemble_retriever)
    # queries, _ = load_data(args.input_data)
    # process_queries(ensemble_retriever, queries, len(queries), args)

    # exact_founds = 0
    # queries, _ = load_data(args.input_data)
    # correct_dict = {1: 0, 3: 0, 5: 0, 10: 0, 13: 0, 15: 0}
    # k_values = [1, 3, 5, 10, 13, 15]
    # # logger.info(f"Total queries: {queries}")
    # max_queries = len(queries)
    # # random.shuffle(queries)
    # mrr_sum = 0
    # with open(args.output_file, "w") as f:
    #     for query in queries[:max_queries]:
    #         match_found = False
    #         # logger.info(query)
    #         if len(query) == 3:
    #             code, query, domain = query[0], query[1].base_entity, query[1].domain
    #         else:
    #             print(f"Query: {type(query)}")
    #             code, query = query[0], query[1].base_entity
    #         print("Query: ", query)
    #         codes_set = str(code).strip().lower().split("|")
    #         # decompose_query =find_domain(query)
    #         # logger.info(f"Query: {query}---Domain: {decompose_query['domain'].lower()}")
    #         # hybrid_vector_retriever = update_qdrant_search_filter(hybrid_vector_retriever, domain=decompose_query['domain'.lower])
    #         results = ensemble_retriever.invoke(query)
    #         exact_results = exact_match_found_no_vocab(query, results)
    #         if len(exact_results) >= 1:
    #             results = exact_results
    #             match_found = True
    #             # Since we found an exact match, increment recall counts for all k-values
    #             for k in k_values:
    #                 correct_dict[k] += 1
    #         elif args.use_llm and not match_found:
    #             print(
    #                 f"Query: {query}--\nResults: {[res.metadata['label'] for res in results]}"
    #             )
    #             results, _ = pass_to_chat_llm_chain(
    #                 query,
    #                 results[:10],
    #                 llm_name=llm_id,
    #                 prompt_stage=2,
    #                 domain="all",
    #             )
    #         # results = filter_results(query, results)
    #         logger.info(f"length of results: {len(results)}")
    #         if results:
    #             for res in results:
    #                 if "domain" in res.metadata:
    #                     # logger.info(f"*{query}------{res.metadata['label']}---{res.metadata['domain']}---[{res.metadata['sid']}----[{res.metadata['is_standard']}]")
    #                     f.write(
    #                         f"{query}\t{res.metadata['label']}\t{res.metadata['domain']}\t{res.metadata['sid']}\n"
    #                     )
    #                 else:
    #                     # logger.info(f"*{query}------{res.metadata['label']}---[{res.metadata['sid']}]")
    #                     f.write(
    #                         f"{query}\t{res.metadata['label']}\t{res.metadata['sid']}\n"
    #                     )
    #         else:
    #             logger.info(f"*{query}------No results found")
    #             f.write(f"{query}\tNo results found\n")

    #         if not match_found:  # Only calculate rank if no exact match was found
    #             rank = next((i + 1 for i, res in enumerate(results) if res.metadata.get("sid") in codes_set), 0)
    #             if rank > 0:
    #                 mrr_sum += 1 / rank  # Reciprocal rank for the first relevant result

    #             # Recall calculation only if no exact match was found
    #             for k in k_values:
    #                 if any(sid in codes_set for res in results[:k] for sid in str(res.metadata.get("sid", "")).strip().lower().split("|")):
    #                     correct_dict[k] += 1
    #         for k in k_values:
    #             if (
    #                 any(
    #                     sid in codes_set
    #                     for res in results[:k]
    #                     for sid in str(res.metadata.get("sid", ""))
    #                     .strip()
    #                     .lower()
    #                     .replace("+", "|")
    #                     .split("|")
    #                 )
    #                 or match_found
    #             ):
    #                 correct_dict[k] += 1
    #             else:
    #                 logger.info(f"{query} not found in top {k} results")
    # mrr = sum([1 / k for k in k_values]) / max_queries
    # recall_at_1_rate = correct_dict[1] / max_queries if max_queries > 0 else 0
    # recall_at_3_rate = correct_dict[3] / max_queries if max_queries > 0 else 0
    # recall_at_5_rate = correct_dict[5] / max_queries if max_queries > 0 else 0
    # recall_at_10_rate = correct_dict[10] / max_queries if max_queries > 0 else 0
    # correct_dict["file"] = args.input_data
    # correct_dict["mrr"] = mrr
    # correct_dict["recall_at_1"] = recall_at_1_rate
    # correct_dict["recall_at_3"] = recall_at_3_rate
    # correct_dict["recall_at_5"] = recall_at_5_rate
    # correct_dict["recall_at_10"] = recall_at_10_rate
    # log_accuracy(correct_dict)

    # logger.info(f"Exact founds for file: {args.input_data} = {exact_founds}")
    # # logger.info_results(ensemble_retriever.invoke("Progressive Familial Heart Block, Type II")) #midregional pro-atrial natriuretic peptide
    # logger.info(
    #     f"Time taken fot total queries {len(queries)}: {time.time() - start_time}"
    # )

    # import faiss
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# def faiss_vector_store(collection_name,docs_file,embeddings):
#     docs = load_docs_from_jsonl(docs_file)
#     index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
#     vector_store = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )
#     uuids = [str(uuid4()) for _ in range(len(docs))]

#     vector_store.add_documents(documents=docs, ids=uuids)
#     return vector_store.as_retriever(search_kwargs={'k': 10})


# client.update_collection(collection_name='concept_mapping',
#                          vectors_config={
#                         "omop_dense_vector": rest.VectorParamsDiff(
#                             hnsw_config=rest.HnswConfigDiff(


#                                 m=16,
#                                 ef_construct=64,
#                                 payload_m = 16
#                             ),
#                             quantization_config = None,
#                             # quantization_config=rest.ScalarQuantization(
#                             #         scalar=rest.ScalarQuantizationConfig(
#                             #             type=rest.ScalarType.INT8,
#                             #             quantile=0.99,
#                             #             always_ram=True,
#                             #         ),
#                             #         ),
#                             on_disk=True,
#                         ),
#                     },
#                     sparse_vectors_config =  {
#                                         "omop_sparse_vector": rest.SparseVectorParams(
#                                             index=rest.SparseIndexParams(
#                                                 on_disk=True,
#                                                 full_scan_threshold=20000
#                                             ),
#                                             modifier=rest.Modifier.IDF
#                                         )
#                                     },
#                          )
