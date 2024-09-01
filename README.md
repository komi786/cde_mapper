![link_icon Background Removed](https://github.com/user-attachments/assets/45a29e7d-58d3-4532-a5ef-b90daa624bc7) # CT\_Mapper: Automated Terminology Mapping of Clinical Terms to OMOP Vocabularies using Retrieval Augmented Generation: 
CT Mapper is an automated concept linking tool to find appropriate standardized terms in OMOP Athena vocabularies for clinical terms in data dictionary. The mapping tool is built with advanced Retrieval Augment Generation method. We leverage the power of generative models and vector store to improve accuracy of linking for composite concepts. This tool allows the data custodians to transform ambiogious and semi-structured clinical data to harmonized schema.

# Task Description
To harmonize clinical data effectively, it is crucial to understand the varying levels of conceptual representation within it. Many existing frameworks have attempted to address the challenge of concept linking, focusing primarily on clinical terms with atomic representation. However, in the context of challenges encountered in mapping cohort studies for the ICARE4CVD project, we propose a solution that leverages retrieval-augmented generation and in-context learning. Rather than mapping individual terms, we focus on mapping data dictionaries presented in a horizontal table format, where each row is treated as a single query. Each query may include multiple components such as labels, descriptions, methods, formulas, units, and categorical values. To standardize and extract clinical terms from each query, we employed in-context learning with generative models like LLAMA and GPT-4. For retrieval, we utilized a hybrid vector search combined with metadata filtering on structured schema, which enhances precision. To further refine the results, we propose a multi-stage ranking method. This includes a large language model-based cross-ranking method to filter out irrelevant candidates, followed by a relevance-based scoring and relationship prediction. The cumulative score from these steps is used to identify the final candidate.
![rag_pipeline_major_revision_v4](https://github.com/user-attachments/assets/80f66675-cf69-4d9c-8781-e97d9aea4bd5)

# Packages
pandas
tqdm
torch
transformers
python-dotenv
qdrant-client
langchain
langchain_openai
ctransformers
pydantic>=1.10.8
typing-extensions>=4.8.0
torch>=2.2.2
openai>=1.19.0
qdrant-client>=1.8.2
langchain-community
togather
faiss-cpu
faiss-gpu
langchain-togather
simstring-fast
