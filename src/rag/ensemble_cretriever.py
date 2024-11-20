# import asyncio
# from collections import defaultdict
# from collections.abc import Hashable
# from itertools import chain
# from typing import (
#     Any,
#     Callable,
#     Dict,
#     Iterable,
#     Iterator,
#     List,
#     Optional,
#     TypeVar,
#     cast,
# )
# # from langchain.retrievers.document_compressors import EmbeddingsFilter
# # from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# from .param import *
# from .utils import global_logger as logger
# from sentence_transformers.cross_encoder import CrossEncoder
# from langchain_core.callbacks import (
#     AsyncCallbackManagerForRetrieverRun,
#     CallbackManagerForRetrieverRun,
# )
# from langchain_core.documents import Document
# from langchain_core.load.dump import dumpd
# from langchain_core.pydantic_v1 import root_validator
# from langchain_core.retrievers import BaseRetriever, RetrieverLike
# from langchain_core.runnables import RunnableConfig
# from langchain_core.runnables.config import ensure_config, patch_config
# from langchain_core.runnables.utils import (
#     ConfigurableFieldSpec,
#     get_unique_config_specs,
# )
# from .utils import global_logger as logger
# T = TypeVar("T")
# H = TypeVar("H", bound=Hashable)


# def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
#     """Yield unique elements of an iterable based on a key function.

#     Args:
#         iterable: The iterable to filter.
#         key: A function that returns a hashable key for each element.

#     Yields:
#         Unique elements of the iterable based on the key function.
#     """
#     seen = set()
#     for e in iterable:
#         if (k := key(e)) not in seen:
#             seen.add(k)
#             yield e

# import numpy as np
# from FlagEmbedding import FlagReranker
# import torch
# torch.cuda.empty_cache()
# class ConceptEnsembleRetriever(BaseRetriever):
#     """A concept ensemble retriever that combines  the top k documents from different retrievers.

#     This retriever only implements the sync method _get_relevant_documents.

#     If the retriever were to involve file access or network access, it could benefit
#     from a native async implementation of `_aget_relevant_documents`.

#     As usual, with Runnables, there's a default async implementation that's provided
#     that delegates to the sync implementation running on another thread.
#     """
#     retrievers: List[RetrieverLike]
#     weights: List[float]
#     c: int = 60
#     filters: Optional[Dict[str, List[Any]]] = None  # Define filters with type hint
#     # model = CrossEncoder("ncbi/MedCPT-Cross-Encoder", max_length=512)
#     model = CrossEncoder("jinaai/jina-reranker-v1-turbo-en", trust_remote_code=True)
#     # reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True,cache_dir='/workspace/mapping_tool/resources/models', device='cuda:2')

#     # embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5",
# #     model_kwargs={},
# #     encode_kwargs={"normalize_embeddings": True},
# # )
#     @property
#     def config_specs(self) -> List[ConfigurableFieldSpec]:
#         """List configurable fields for this runnable."""
#         return get_unique_config_specs(
#             spec for retriever in self.retrievers for spec in retriever.config_specs
#         )

#     @root_validator(pre=True)
#     def set_weights(cls, values: Dict[str, Any]) -> Dict[str, Any]:
#         if not values.get("weights"):
#             n_retrievers = len(values["retrievers"])
#             values["weights"] = [1 / n_retrievers] * n_retrievers
#         return values
#     @root_validator(pre=True)
#     def validate_filters(cls, values: Dict[str, Any]) -> Dict[str, Any]:
#         """Validate or set default filters if not provided."""
#         if 'filters' not in values or values['filters'] is None:
#             values['filters'] = {}  # Set default to an empty dict if not provided
#         return values
#     def filter_documents(self, docs: List[Document]) -> List[Document]:
#         """Filter documents based on the specified metadata criteria."""
#         if not self.filters:
#             return docs  # Return all docs if no filter is specified
#         filtered_docs = []
#         for doc in docs:
#             include_doc = True
#             for key, valid_values in self.filters.items():
#                 # logger.info(f"filter={key} valid_values={valid_values}")
#                 # Assuming the document metadata is a dictionary
#                 doc_value = doc.metadata.get(key)
#                 # logger.info(f"doc_value={doc_value}")
#                 if doc_value not in valid_values:
#                     include_doc = False
#                     break
#             if include_doc:
#                 filtered_docs.append(doc)
#         return filtered_docs

#     def invoke(
#         self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
#     ) -> List[Document]:
#         from langchain_core.callbacks import CallbackManager

#         config = ensure_config(config)
#         callback_manager = CallbackManager.configure(
#             config.get("callbacks"),
#             None,
#             verbose=kwargs.get("verbose", False),
#             inheritable_tags=config.get("tags", []),
#             local_tags=self.tags,
#             inheritable_metadata=config.get("metadata", {}),
#             local_metadata=self.metadata,
#         )
#         run_manager = callback_manager.on_retriever_start(
#             dumpd(self),
#             input,
#             name=config.get("run_name"),
#             **kwargs,
#         )
#         try:
#             result = self.rank_fusion(input, run_manager=run_manager, config=config)
#         except Exception as e:
#             run_manager.on_retriever_error(e)
#             raise e
#         else:
#             run_manager.on_retriever_end(
#                 result,
#                 **kwargs,
#             )
#             return result

#     async def ainvoke(
#         self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
#     ) -> List[Document]:
#         from langchain_core.callbacks import AsyncCallbackManager

#         config = ensure_config(config)
#         callback_manager = AsyncCallbackManager.configure(
#             config.get("callbacks"),
#             None,
#             verbose=kwargs.get("verbose", False),
#             inheritable_tags=config.get("tags", []),
#             local_tags=self.tags,
#             inheritable_metadata=config.get("metadata", {}),
#             local_metadata=self.metadata,
#         )
#         run_manager = await callback_manager.on_retriever_start(
#             dumpd(self),
#             input,
#             name=config.get("run_name"),
#             **kwargs,
#         )
#         try:
#             result = await self.arank_fusion(
#                 input, run_manager=run_manager, config=config
#             )
#         except Exception as e:
#             await run_manager.on_retriever_error(e)
#             raise e
#         else:
#             await run_manager.on_retriever_end(
#                 result,
#                 **kwargs,
#             )
#             return result

#     def _get_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: CallbackManagerForRetrieverRun,
#     ) -> List[Document]:
#         """
#         Get the relevant documents for a given query.

#         Args:
#             query: The query to search for.

#         Returns:
#             A list of reranked documents.
#         """

#         # Get fused result of the retrievers.
#         fused_documents = self.rank_fusion(query, run_manager)

#         return fused_documents

#     async def _aget_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: AsyncCallbackManagerForRetrieverRun,
#     ) -> List[Document]:
#         """
#         Asynchronously get the relevant documents for a given query.

#         Args:
#             query: The query to search for.

#         Returns:
#             A list of reranked documents.
#         """

#         # Get fused result of the retrievers.
#         fused_documents = await self.arank_fusion(query, run_manager)

#         return fused_documents
#     def rank_fusion(
#         self,
#         query: str,
#         run_manager: CallbackManagerForRetrieverRun,
#         *,
#         config: Optional[RunnableConfig] = None,
#     ) -> List[Document]:
#         """
#         Retrieve the results of the retrievers and use rank_fusion_func to get
#         the final result.

#         Args:
#             query: The query to search for.

#         Returns:
#             A list of reranked documents.
#         """

#         # Get the results of all retrievers.
#         retriever_docs = [
#             retriever.invoke(
#                 query,
#                 patch_config(
#                     config, callbacks=run_manager.get_child(tag=f"retriever_{i+1}")
#                 ),
#             )
#             for i, retriever in enumerate(self.retrievers)
#         ]
        
#         # Enforce that retrieved docs are Documents for each list in retriever_docs
#         for i in range(len(retriever_docs)):
#             retriever_docs[i] = [
#                 Document(page_content=cast(str, doc)) if isinstance(doc, str) else doc
#                 for doc in retriever_docs[i]
#             ]
#         retriever_docs = [self.filter_documents(docs) for docs in retriever_docs]
#         for list in retriever_docs:
#             for i,doc in enumerate(list):
#                 logger.info(f"{i}:{doc.metadata['label']}")
#         # word_count = len(query.split())
#         # if word_count > 3:
#         #     # More than 4 words in the query, apply cross-encoder ranking
#         #     # logger.info("Applying cross-encoder ranking due to query length.")
#         #     fused_documents = self._perform_embedding_based_rank(query, retriever_docs)
#         # else:
#         #     # 4 words or fewer, apply rank fusion
#         #     # logger.info("Applying rank fusion due to shorter query.")
#         fused_documents = self.weighted_reciprocal_rank(retriever_docs)

#         # Output the top documents after applying the chosen method
#         # logger.info(f"\n************Top documents after processing*************\n")
#         # for i,doc in enumerate(fused_documents):
#         #         logger.info(f"{i}:{doc.metadata['label']}")
#         return fused_documents
#         # # apply rank fusion
        
#         # # fused_documents = self.weighted_reciprocal_rank(retriever_docs)
#         # #apply cross encoder ranking
        
#         # fused_documents =  self.perform_cross_encoder_rank(query,retriever_docs)
#         # # logger.info(f"ensemble retriever topk = {fused_documents}")
#         # return fused_documents
#     def re_rank_search_results(self, mention, unique_docs):
        
#         # print(f"original ranking: {unique_docs}")
#         model = CrossEncoder("jinaai/jina-reranker-v1-turbo-en", trust_remote_code=True)
#         if len(unique_docs) == 0:
#             return []
#         documents = [f"{doc.metadata['label']},Domain:{doc.metadata['domain']}" for doc in unique_docs.values()]
#         # print(f"documents: {documents}")
#         results = model.rank(mention, documents, return_documents=True, top_k=5)
#         scores = [result['score'] for result in results]
#         print(f"cross re_ranked scores={scores}")
#         if len(scores) == 0:
#             max_score = 1  # Fallback if no scores
#         else:
#             max_score = max(scores)  # Corrected to use max()

#         if np.all(np.array(scores) >= 0) and np.all(np.array(scores) <= 1):
#             normalized_scores = scores
#         else:
#             if max_score == 0:
#                 logger.warning("Maximum score is zero, normalization will not be performed.")
#                 normalized_scores = scores
#             else:
#                 normalized_scores = [score / max_score for score in scores]

#         # Assign normalized scores to the corresponding documents
#         for result, score in zip(results, normalized_scores):
#             label = result['text'].split(",Domain:")[0]
#             if label in unique_docs:
#                 doc = unique_docs[label]
#                 doc.metadata['score'] = score
#         all_scored_docs = [doc for doc in unique_docs.values() if doc.metadata.get('score', 0) > 0.5]
#         sorted_docs = sorted(all_scored_docs, key=lambda doc: doc.metadata['score'], reverse=True)
#         logger.info(f"cross re_ranked sorted_docs={sorted_docs}")
#         return sorted_docs
            
#     def _perform_embedding_based_rank(self, query: str, doc_lists: List[List[Document]]) -> List[Document]:
#         # This will store all scored documents across all lists
#         if len(doc_lists) == 0:
#             return []

#         # Dictionary to track unique documents by their label to prevent duplicate processing
#         unique_docs: Dict[str, Document] = {}

#         # Gather all unique documents
#         for sublist in doc_lists:
#             for doc in sublist:
#                 unique_docs[doc.metadata['label']] = doc
#         return self.re_rank_search_results(query, unique_docs)

#     async def arank_fusion(
#         self,
#         query: str,
#         run_manager: AsyncCallbackManagerForRetrieverRun,
#         *,
#         config: Optional[RunnableConfig] = None,
#     ) -> List[Document]:
#         """
#         Asynchronously retrieve the results of the retrievers
#         and use rank_fusion_func to get the final result.

#         Args:
#             query: The query to search for.

#         Returns:
#             A list of reranked documents.
#         """

#         # Get the results of all retrievers.
#         retriever_docs = await asyncio.gather(
#             *[
#                 retriever.ainvoke(
#                     query,
#                     patch_config(
#                         config, callbacks=run_manager.get_child(tag=f"retriever_{i+1}")
#                     ),
#                 )
#                 for i, retriever in enumerate(self.retrievers)
#             ]
#         )
#         # Enforce that retrieved docs are Documents for each list in retriever_docs
#         for i in range(len(retriever_docs)):
#             retriever_docs[i] = [
#                 Document(page_content=doc) if not isinstance(doc, Document) else doc  # type: ignore[arg-type]
#                 for doc in retriever_docs[i]
#             ]
#         logger.info(f"retriever_docs={retriever_docs}")   
#         word_count = len(query.split())
#         if word_count > 2:
#             # More than 4 words in the query, apply cross-encoder ranking
#             # logger.info("Applying cross-encoder ranking due to query length.")
#             fused_documents = self._perform_embedding_based_rank(query, retriever_docs)
#         else:
#             # 4 words or fewer, apply rank fusion
#             # logger.info("Applying rank fusion due to shorter query.")
#             fused_documents = self.weighted_reciprocal_rank(retriever_docs)

#         # Output the top documents after applying the chosen method
#         # logger.info(f"Top documents after processing: {fused_documents}")
#         return fused_documents

#     def weighted_reciprocal_rank(
#         self, doc_lists: List[List[Document]]
#     ) -> List[Document]:
#         """
#         Perform weighted Reciprocal Rank Fusion on multiple rank lists.
#         You can find more details about RRF here:
#         https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

#         Args:
#             doc_lists: A list of rank lists, where each rank list contains unique items.

#         Returns:
#             list: The final aggregated list of items sorted by their weighted RRF
#                     scores in descending order.
#         """
#         if len(doc_lists) != len(self.weights):
#             raise ValueError(
#                 "Number of rank lists must be equal to the number of weights."
#             )

#         # Associate each doc's content with its RRF score for later sorting by it
#         # Duplicated contents across retrievers are collapsed & scored cumulatively
#         rrf_score: Dict[str, float] = defaultdict(float)
#         for doc_list, weight in zip(doc_lists, self.weights):
#             for rank, doc in enumerate(doc_list, start=1):
#                 rrf_score[doc.page_content] += weight / (rank + self.c)
        
#         # Docs are deduplicated by their contents then sorted by their scores
#         all_docs = chain.from_iterable(doc_lists)
#         unique_docs = list(unique_by_key(all_docs, lambda doc: doc.page_content))
#         # logger.info(f"RFF SCORE={rrf_score}")
#         for doc in unique_docs:
#             # logger.info(f"doc type={type(doc)}")
#             doc.metadata['score'] = rrf_score[doc.page_content]
#             # logger.info(f"docs={doc}")
#         sorted_docs = sorted(
#             unique_docs,
#             reverse=True,
#             key=lambda doc: doc.metadata['score'],
#         )
#         # logger.info(f"sorted docs = {sorted_docs}")
#         return sorted_docs
#         # sorted_docs = sorted(
#         #     unique_by_key(all_docs, lambda doc: doc.page_content),
#         #     reverse=True,
#         #     key=lambda doc: rrf_score[doc.page_content],
#         # )
#         # return sorted_docs
#         """
#                 # Associate each doc's content with its RRF score for later sorting by it
#         # Duplicated contents across retrievers are collapsed & scored cumulatively
#         rrf_score: Dict[str, float] = defaultdict(float)
#         for doc_list, weight in zip(doc_lists, self.weights):
#             for rank, doc in enumerate(doc_list, start=1):
#                 rrf_score[doc.page_content] += weight / (rank + self.c)

#         # Docs are deduplicated by their contents then sorted by their scores
#         all_docs = chain.from_iterable(doc_lists)
#         sorted_docs = sorted(
#             unique_by_key(all_docs, lambda doc: doc.page_content),
#             reverse=True,
#             key=lambda doc: rrf_score[doc.page_content],
#         )
#         return sorted_docs
#         """
    