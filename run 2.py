
import argparse
from rag.retriever_ import *
from rag.param import *
from rag.vector_index import *
from rag.utils import global_logger as logger
from rag.data_loader import load_data
# import asyncio
import signal
import sys

def graceful_shutdown(signum, frame):
    logger.info("Graceful shutdown initiated.")
    # Perform cleanup operations here: close database connections, save state, etc.
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    parser = argparse.ArgumentParser(description="Load and configure retrievers.")
    parser.add_argument('--model_id', type=str, default=EMB_MODEL_NAME, help='Model identifier for embeddings')
    parser.add_argument('--vector_path', type=str, default=VECTOR_PATH, help='Path to vector data')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Directory containing data')
    parser.add_argument('--retriever_type', type=str, default=RETRIEVER, help='Directory containing data')
    parser.add_argument('--collection_name', type=str, default=SYN_COLLECTION_NAME, help='vector collection name')
    parser.add_argument('--llm_id', type=str, default=LLM_ID, help='LLM model name')
    parser.add_argument('--topk', type=int, default=5, help='Top k documents to retrieve')
    parser.add_argument('--input_file', type=str, required=True, help='File containing queries (.csv or .txt)')
    parser.add_argument('--custom_data', action='store_true', help='Flag to use custom data for retriever')
    parser.add_argument('--eval', action='store_true', help='Flag to use custom data for retriever')
    parser.add_argument('--is_omop_data', action='store_true', help='Flag to use OMOP vocabulary')
    parser.add_argument('--output_file', type=str, required=False, help='File to save the retrieved documents')
    parser.add_argument('--flag', type=str, required=True,  default='inference', help='Flag to recreate , update or inference')
    parser.add_argument('--document_file_path', type=str, required=False, help='For recreate or update documents-add in data directory')
    parser.add_argument('--embedding_type', type=str, default= 'hier', required=False, help='For recreate or update documents-add in data directory')
    
    # parser.add_argument('--custom_data',action='store_true',help='Flag to use custom data for retriever')
    args = parser.parse_args()
    embeddings = SAPEmbeddings()
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
    output_file = None
    if args.output_file is None:
        output_file = args.output_file 
    data = load_data(args.input_file, load_custom=args.custom_data)
    hybrid_search = generate_vector_index(embeddings, sparse_embeddings, docs_file= args.document_file_path,mode=args.flag,
                                          collection_name=args.collection_name, topk=args.topk)
    # compressed_hybrid_retriever =  set_compression_retriever(hybrid_search)
    athena_api_retriever = initiate_api_retriever()
    merger_retriever= set_merger_retriever(retrievers=[hybrid_search,athena_api_retriever])
    merger_retriever = set_compression_retriever(merger_retriever)
    results=map_data(data,merger_retriever,custom_data=args.custom_data,output_file=args.output_file,
             llm_name=args.llm_id,topk=args.topk,do_eval=args.eval, is_omop_data=args.is_omop_data)
    append_results_to_csv(args.input_file,results) 
    logger.info(f"Retrieval complete, results saved to:{ args.output_file}")
    
