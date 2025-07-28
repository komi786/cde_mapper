# from langchain_qdrant import Qdrant
# from qdrant_client import QdrantClient
# import qdrant_client.http.models as rest
# from langchain.retrievers import EnsembleRetriever
# from langchain_community.retrievers import QdrantSparseVectorRetriever, BM25Retriever
# from langchain_core.retrievers import RetrieverLike
# from .utils import load_docs_from_jsonl, global_logger as logger
# from typing import Any, List
# import pickle
# from ..analysis_files.database import *
# from .param import *
# from langchain_community.document_transformers import (
#     LongContextReorder,
# )
# # from .ensemble_cretriever import ConceptEnsembleRetriever
# # reordering = LongContextReorder()

# def configure_ensemble_retriever(search_type, retrievers, must_conditions, should_conditions, filters, score_threshold, topk):
#     """
#     Configures and returns an ensemble retriever.
    
#     Args:
#         search_type (str): The type of search to perform.
#         retrievers (list): List of retrievers to be used in the ensemble.
#         must_conditions (list): List of 'must' conditions for filtering.
#         should_conditions (list): List of 'should' conditions for filtering.
#         filters (dict): Filters to apply during search.
#         score_threshold (float): Threshold for score filtering.
#         topk (int): The number of top results to retrieve.

#     Returns:
#         EnsembleRetriever: Configured ensemble retriever.
#     """
#     dense_retriever = retrievers[0]
#     sparse_retriever = retrievers[1]
#     if search_type == 'sparse+bm25':
#         dense_retriever = update_search_param(dense_retriever, must_conditions, should_conditions, score_threshold=score_threshold, topk=topk)
#     elif search_type == 'dense+bm25':
#         dense_retriever = update_search_param(dense_retriever, must_conditions, should_conditions, score_threshold=score_threshold, dense=True, topk=topk)
#     elif search_type == 'dense+sparse':
#         dense_retriever = update_search_param(dense_retriever, must_conditions, should_conditions, score_threshold=score_threshold, dense=True, topk=topk)
#         sparse_retriever = update_search_param(sparse_retriever, must_conditions, should_conditions, score_threshold=score_threshold, topk=topk)
#     return initiate_ensemble_retriever([dense_retriever, sparse_retriever], weights=[0.5, 0.5], filter=filters)



# def get_collection_vectors(client, logger, collection_name):
#     try:
#         collection_status = client.get_collection(collection_name)
#         # logger.info(f"collection status={collection_status}")
#         vectors_count = collection_status.points_count
#         logger.info(f"{collection_name} has {vectors_count} vectors.")
#         return vectors_count
#     except Exception as e:
#         logger.info(f"Error fetching collection {collection_name} from Qdrant: {e}")
#         return 0


# def check_and_add_documents(
#         client: QdrantClient,
#         collection_name: str,
#         vocab: List[str],
#         documents_to_add
#     ):
#     """ Boolean function to check if data points exist in Qdrant and return True if no points are found for all vocabularies and new documents are added.
#     Make sure to pass only new vocabularies and documents to add.
#     Args:
#         client (QdrantClient): _description_
#         collection_name (str): _description_
#         vocab (List[str]): _description_
#         documents_to_add (_type_): _description_

#     Returns:
#         _type_: _description_
        
#     """
#     # Prepare the filter based on vocabularies
#     scroll_filter = rest.Filter(
#         must=[
#             rest.FieldCondition(key="metadata.vocab", match=rest.MatchAny(values=vocab))
#         ]
#     )
    
#     # Attempt to fetch data points from Qdrant
#     try:
#         response = client.scroll(
#             collection_name=collection_name,
#             scroll_request=rest.ScrollRequest(
#                 filter=scroll_filter,
#                 limit=10,  # Adjust limit as needed
#                 with_payload=True
#             )
#         )
#         logger.info(f"Scrolled {len(response.result.points)} points")

#         # If no points are found, add new documents
#         if not response.result.points:
#             logger.info("No data points found, adding new documents...")
#             return True
#         else:
#             logger.info("Data points found, skipping addition.")
#             return False

#     except Exception as e:
        
#         logger.info(f"Error fetching collections from Qdrant: {e}")
#         return False
# def configure_single_retriever(search_type, retrievers, must_conditions, should_conditions, filters, score_threshold, topk):
#     """
#     Configures and returns a single retriever.
    
#     Args:
#         search_type (str): The type of search to perform.
#         retrievers (list): List of retrievers to be used in the ensemble.
#         must_conditions (list): List of 'must' conditions for filtering.
#         should_conditions (list): List of 'should' conditions for filtering.
#         filters (dict): Filters to apply during search.
#         score_threshold (float): Threshold for score filtering.
#         topk (int): The number of top results to retrieve.

#     Returns:
#         Retriever: Configured single retriever.
#     """
#     if len(retrievers) == 1:
#         one_retriever = retrievers[0]
#         if search_type == 'only_dense':
#             one_retriever = update_search_param(one_retriever, must_conditions, should_conditions, score_threshold=score_threshold, dense=True, topk=topk)
#         return one_retriever
#     retriever_one = retrievers[0]
#     retriever_two = retrievers[1]
#     updated_retrievers = [retriever_one, retriever_two]
#     return initiate_ensemble_retriever(updated_retrievers, weights=[0.5, 0.5], filter=filters)


# #search_type =similarity_score_threshold , mrr
# def initate_dense_retriever(embedding_model, documents=None,url=VECTOR_PATH, collection_name=SYN_COLLECTION_NAME,
#                             vector_name="dense_vector", search_type="similarity_score_threshold", top_k=10, 
#                             min_score=0.5,flag:str="inference",logger=None, quant_type=QUANT_TYPE):
#         client =QdrantClient(url=url, port=QDRANT_PORT, https=True,timeout=500)
#         vectors_count = get_collection_vectors(client=client, logger=logger,collection_name=collection_name)
#         # collections = client.get_collections()
#         # logger.info(f"All Collections={collections}")
#         if flag == "inference" or flag == "update":
#             recreate = False 
#         else:
#             recreate = True
#         if recreate and documents is not None:
#             logger.info(f"Initializing Recreation of vectors of len={len(documents)}.")
            
#             if quant_type == 'scalar':
#                 quant_config = rest.ScalarQuantization(
#                     scalar=rest.ScalarQuantizationConfig(
#                         type=rest.ScalarType.INT8,
#                         quantile=0.99,
#                         always_ram=True,
#                     ),
#                     )
#             elif quant_type == 'binary':
#                 quant_config = rest.BinaryQuantization(
#                         binary=rest.BinaryQuantizationConfig(
#                             always_ram=True,
#                         )
#                         )
            
#             else:
#                 quant_config =rest.ProductQuantization(
#                 product=rest.ProductQuantizationConfig(
#                     compression=rest.CompressionRatio.X16,
#                     always_ram=True,
#                 ),)
#             qdrant = Qdrant.from_documents(
#                 url=url, 
#                 port=QDRANT_PORT,
#                 timeout=300,
#                 https=True,
#                 batch_size=64,  # Adjust batch size as needed
#                 collection_name=collection_name,
#                 documents = documents,
#                 embedding = embedding_model,
#                 optimizers_config=rest.OptimizersConfigDiff(
#                     indexing_threshold=20000,
#                     deleted_threshold=0.2,
#                     default_segment_number=15,    #based on the number of available CPUs
#                     memmap_threshold=20000,            #lower than indexing_thresold in case write load is greater than RAM
#                     max_segment_size = 20000
#                 ),
#                 vector_name = "dense_vector",
#                 hnsw_config=rest.HnswConfigDiff(
#                         m=64,
#                         ef_construct=128, 
#                         full_scan_threshold=20000,
#                         max_indexing_threads = 8,
#                         payload_m = 64
#                     ),
#                 quantization_config=quant_config,
#                 shard_number=2,
#                 # replication_factor=1,
#                 # write_consistency_factor=1,
#                 force_recreate = recreate,
#                 on_disk = True,
#                 on_disk_payload =True,
                
#         )
#             _create_payload_index(client, collection_name)
            
#         else:
        
#             #  _create_payload_index(client, collection_name)
#              logger.info(f"Dense Collection Exists with vectors = {vectors_count}")
            
#              qdrant = Qdrant(client=client, collection_name=collection_name,
#              embeddings= embedding_model,
#              vector_name = vector_name,
#              distance_strategy= 'COSINE')
#              if flag == "inference":
#                 # client.update_collection(
#                 #             collection_name=collection_name,
#                 #             vectors_config={
#                 #                 "dense_vector": models.VectorParamsDiff(
#                 #                     hnsw_config=models.HnswConfigDiff(
#                 #                         m=32,
#                 #                         ef_construct=64,
#                 #                         payload_m = 32
#                 #                     ),
#                 #                     quantization_config=rest.ScalarQuantization(
#                 #                             scalar=rest.ScalarQuantizationConfig(
#                 #                                 type=rest.ScalarType.INT8,
#                 #                                 quantile=0.99,
#                 #                                 always_ram=True,
#                 #                             ),
#                 #                             ),
#                 #                     on_disk=True,
#                 #                 ),
#                 #             },
#                 #             optimizer_config=rest.OptimizersConfigDiff
#                 #                                     (indexing_threshold=20000,
#                 #                                         deleted_threshold=0.2,
#                 #                                         default_segment_number=10,    #based on the number of available CPUs
#                 #                                         memmap_threshold=20000),
                
#                 #     )
                    
#                 if flag == "update":
#                     if documents and vectors_count < len(documents):
#                         documents = documents[vectors_count:]
#                         qdrant.add_documents(documents=documents)
#                         logger.info("collection updated")
#                         _create_payload_index(client, collection_name)
#         qdrant_retriever = qdrant.as_retriever(search_type=search_type,
#                     search_kwargs={
#                             # "fetch_k": 20,
#                             # "lambda_mult": 0.3,
#                             "k":top_k,
#                             "score_threshold":min_score,

#                         },
#                     return_source_documents=True,
#                     )
#         return qdrant_retriever

 
# def _create_payload_index(client, collection_name):
#     client.create_payload_index(
#                     collection_name=collection_name,
#                     field_schema= "keyword",
#                     field_name="metadata.label",
#                     wait = True
#                     )
#     client.create_payload_index(
#                     collection_name=collection_name,
#                     field_schema= "keyword",
#                     field_name="metadata.domain",
#                     wait=True
#                 )
#     client.create_payload_index(
#                     collection_name=collection_name,
#                     field_schema= "keyword",
#                     field_name="metadata.vocab",
#                     wait=True
#                 )
#     client.create_payload_index(
#                     collection_name=collection_name,
#                     field_schema= "keyword",
#                     field_name="metadata.is_standard",
#                     wait=True
#                 )
#     # client.create_payload_index(
#     #                 collection_name=collection_name,
#     #                 field_schema= "keyword",
#     #                 field_name="metadata.concept_class",
#     #                 wait=True
#     #             )
# def initate_sparse_retriever(sparse_encoder=Any, documents=None, url=VECTOR_PATH, collection_name=SYN_SPARSE_COLLECTION_NAME,
#                             vector_name="sparse_vector",  top_k=5, 
#                             flag:str="inference",logger=None):   
#     client = QdrantClientSingleton.get_instance(url=url)
#     vectors_count = get_collection_vectors(client=client, logger=logger,collection_name=collection_name)
#     if vectors_count == 0 and documents is not None:
#                 print(client.delete_collection(collection_name))
#                 client.create_collection(
#                     collection_name,
#                     vectors_config={},
#                     sparse_vectors_config={
#                         vector_name: rest.SparseVectorParams(
#                             index=rest.SparseIndexParams(
#                                 on_disk=False,
#                             )
#                         )
#                     },
#                     hnsw_config=rest.HnswConfigDiff(
#                         m=0,
#                         ef_construct=64,
#                         payload_m = 16,
#                         full_scan_threshold=20000
#                     ),
#                 )
#                 retriever = QdrantSparseVectorRetriever(
#                     client=client,
#                     collection_name=collection_name,
#                     sparse_vector_name=vector_name,
#                     sparse_encoder=sparse_encoder,
#                     k=round(top_k/2),
#                 )
                
#                 retriever.add_documents(documents)
#                 vectors_count = get_collection_vectors(client=client, logger=logger,collection_name=collection_name)
#                 _create_payload_index(client, collection_name)
#                 return retriever
        
#     else:
#         if documents is not None and vectors_count < len(documents):
#             _create_payload_index(client, collection_name)
#             documents = documents[vectors_count:]
#             logger.info(f"SPARSE Collection Exists with vectors = {vectors_count} and adding {len(documents)} new documents.")
#             retriever = QdrantSparseVectorRetriever(
#                 client=client,
#                 collection_name=collection_name,
#                 sparse_vector_name=vector_name,
#                 sparse_encoder=sparse_encoder,
#                 k=round(top_k/2),
#             )
#             retriever.add_documents(documents)
#             vectors_count = get_collection_vectors(client=client, logger=logger,collection_name=collection_name)
#         else:
#             # _create_payload_index(client, collection_name)
#             # logger.info(f"Collection Exists={vectors_count}")
#             retriever = QdrantSparseVectorRetriever(
#                 client=client,
#                 collection_name=collection_name,
#                 sparse_vector_name=vector_name,
#                 sparse_encoder=sparse_encoder,
#                 k=round(top_k/2),
#             )
#         return retriever




# def initiate_bm25(documents=None, data_dir=DATA_DIR,topk=5):
#     docs = None
#     main_docs_path = os.path.join(data_dir,'output/concepts_all.jsonl')
#     bm25_retriever_path = os.path.join(data_dir, 'output/bm25_retriever.pkl')
#     logger.info(f"BM25 Retriever Path: {bm25_retriever_path}") 

#     if  os.path.exists(bm25_retriever_path):
#         with open(bm25_retriever_path, 'rb') as f:
#             bm25_retriever = pickle.load(f)
#         logger.info(f"BM25 Retriever loaded from disk.")
#         return bm25_retriever

#     if documents:
#         docs = documents
#         # save_docs_to_jsonl(main_docs_path, docs)
#     elif os.path.exists(main_docs_path):
#         docs = load_docs_from_jsonl(main_docs_path)
#     if docs is not None:
#         for doc in docs:
#             doc.page_content = doc.metadata['label']
#         # logger.info(f"BM25 Retriever initialized with {len(docs)} documents.")
#         # bm25_params = {
#         #         'k1': 1.5,
#         #         'b': 0.3,
#         #     }
#         bm25_retriever = BM25Retriever.from_documents(docs)
#         bm25_retriever.k = 5
#         with open(bm25_retriever_path, 'wb') as f:
#             pickle.dump(bm25_retriever, f)
#         logger.info(f"BM25 Retriever initialized with {len(docs)} documents.")
#         return bm25_retriever
#     else:
#         raise ValueError("Documents cannot be None.")


# def initiate_ensemble_retriever(list_of_retrievers:List[RetrieverLike], weights:List,filter=None):
#     try:
#         ensemble_retriever = EnsembleRetriever(retrievers=list_of_retrievers, weights=weights,filters=filter)
#         return ensemble_retriever
#     except Exception as e:
#         logger.info(f"Failed to initialize EnsembleRetriever: {str(e)}")
#         raise

# # def pretty_print_docs(docs,field_key='semantic_type'):
# #     for i, doc in enumerate(docs):
# #         logger.info(f"Document {i + 1}:")
# #         logger.info(f"Page Content:\n{doc.page_content}")
# #         if field_key in doc.metadata:
# #             logger.info(f"concept class: {doc.metadata[field_key]}")
# #         logger.info("-" * 100)


# def retriever_docs(query,  retriever):
   
#     try:
#         # logger.info(f"Normalized Query: {normalized_query}")
#         """Helper function to process a single query."""
#         results = retriever.invoke(query)
#         if results:
#             for doc in results:
#                 if 'vector' in doc.metadata:
#                     del doc.metadata['vector']

#             return results
#         else:
#             return []      
#     except Exception as e:
#         logger.info(f"Error processing query: {e}")
#         return []



# # def update_ensemble_ret_search_filter(ensemble_retriever, domain='all'):
    
# #     api_retriever = update_api_search_filter(ensemble_retriever.retrievers[1], domain=domain)
# #     original_retriever = update_qdrant_search_filter(ensemble_retriever.retrievers[0], domain=domain)
# #     ensemble_retriever.retrievers = [original_retriever,api_retriever]
# #     return ensemble_retriever