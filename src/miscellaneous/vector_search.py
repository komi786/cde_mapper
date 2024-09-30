import json
import numpy as np
import torch
import pdb
from tqdm import tqdm
import torch
import logging
LOGGER = logging.getLogger()
from sentence_transformers.cross_encoder import CrossEncoder
import faiss
from qdrant_client import QdrantClient
import qdrant_client.models as rest
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant

print(f"cuda available: {torch.cuda.is_available()}")
print(f"devices count: {torch.cuda.device_count()}")

class VectorSearch:
    def __init__(self, embedding, documents, topk, agg_mode="cls", collection_name="test"):
        self.embedding = embedding
        self.documents= documents
        self.topk = topk
        self.agg_mode = agg_mode
        self.collection_name = collection_name
        self.client= QdrantClient(url='komal.qdrant.137.120.31.148.nip.io', port=443, https=True, timeout=300)
    
    def create_qdrant_index(self):
        embedding_shape =  len(self.embedding.embed_documents(["test"])[0])
        print(f"size: {embedding_shape}")
        self.client.recreate_collection(self.collection_name,
                                vectors_config=VectorParams(
                                        size=embedding_shape,
                                        distance=Distance.COSINE,
                                        hnsw_config=rest.HnswConfigDiff(
                                                m=16,
                                                ef_construct=32, 
                                                full_scan_threshold=1000,
                                                max_indexing_threads = 8,
                                                payload_m = 16
                                            ),
                                        # datatype=rest.Datatype.FLOAT16,
                                ),
                                        #  optimizers_config=rest.OptimizersConfigDiff(
                                        #     indexing_threshold=0,
                                        #     deleted_threshold=0.1,
                                        #     default_segment_number=0,    #based on the number of available CPUs
                                        #     memmap_threshold=2000           #lower than indexing_thresold in case write load is greater than RAM
                                        # ),
                                        shard_number=1,
                                        on_disk_payload =True,
                        #                 quantization_config=rest.ScalarQuantization(
                        # scalar=rest.ScalarQuantizationConfig(
                        #     type=rest.ScalarType.INT8,
                        #     quantile=0.99,
                        #     always_ram=True,
                        # ),
                        # )
                                            
                                            
                                        
                                )
        
        batch_size = 64
        for i in tqdm(range(0, len(self.documents), batch_size)):
                # Extract batch
                texts = [doc.page_content for doc in self.documents[i:i+batch_size]]
                metadata = [doc.metadata for doc in self.documents[i:i+batch_size]]                
                # Generate embeddings
                embeddings =  self.embedding.embed_documents(texts)
                print(f"is it list: {isinstance(embeddings[0], list)}")
                # Create points for uploading
                points = []
                for j, emb in enumerate(embeddings):
                    point = rest.PointStruct(
                        id=i * batch_size + j,  # Unique ID for each point across batches
                        vector=emb,  # Convert numpy array to list if required
                        payload={
                            "page_content": texts[j],
                            "metadata": metadata[j]
                        }  
                    )
                    points.append(point)
                
                # Upload to Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # Assuming you want the operation to block until completed
                )
        print(self.client.get_collection(self.collection_name))
        # return self.client


    def search_qdrant(self,mention, score_threshold=0.5):

        embedding_vector = self.embedding.embed_documents([mention])[0]
        # print(f"type of embedding_vector: {type(embedding_vector)}")
        search_result = self.client.query_points(
        collection_name=self.collection_name,
        query=embedding_vector,
        with_payload=True,
        limit=self.topk,
        # score_threshold = score_threshold
        ).points
        search_result  = [Qdrant._document_from_scored_point(collection_name=self.collection_name, content_payload_key="page_content", metadata_payload_key="metadata", scored_point=res) for res in search_result]
        return search_result
    



