from .retriever_ import *
from .param import *
from .utils import global_logger as logger

# def graceful_shutdown(signum, frame):
#     print("Graceful shutdown initiated.")
#     sys.exit(0)

# def load_retrievers(args):
#     if '+' in args.retriever_type:
#         vector_retriever, keyword_retriever = load_retriever(
#             model_id=args.model_id,
#             vector_path=args.vector_path,
#             data_dir=args.data_dir,
#             flag=args.flag,
#             topk=args.topk,
#             quant_type=QUANT_TYPE,
#             search_type=args.retriever_type,
#             collection_name=args.collection_name,
#             document_file_name=args.document_file_path,
#             emb_type=args.embedding_type
#         )
#         retriever_list = [vector_retriever, keyword_retriever]
#     else:
#         dense_retriever, _ = load_retriever(
#             model_id=args.model_id,
#             vector_path=args.vector_path,
#             data_dir=args.data_dir,
#             flag=args.flag,
#             topk=args.topk,
#             quant_type=QUANT_TYPE,
#             search_type=args.retriever_type,
#             collection_name=args.collection_name,
#             document_file_name=args.document_file_path,
#             emb_type=args.embedding_type
#         )
#         retriever_list = [dense_retriever]
#     return retriever_list

# def main(args):
#     # signal.signal(signal.SIGINT, graceful_shutdown)
#     retrievers = load_retrievers(args)
#     results = map_data(
#         args.input_data,
#         retrievers,
#         custom_data=args.custom_data,
#         output_file=args.output_file,
#         llm_name=args.llm_id,
#         search_type=args.retriever_type
#     )
#     return results
