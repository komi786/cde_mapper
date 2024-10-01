# CT_Mapper: Automated Terminology Mapping of Clinical Terms to OMOP Vocabularies using Retrieval Augmented Generation

![CT_Mapper Logo](https://github.com/user-attachments/assets/45a29e7d-58d3-4532-a5ef-b90daa624bc7)

CT_Mapper is an automated concept linking tool designed to find appropriate standardized terms in OMOP Athena vocabularies for clinical terms found in data dictionaries. Built with advanced Retrieval Augmented Generation (RAG) methods, CT_Mapper leverages the power of generative models and vector stores to enhance the accuracy of linking composite concepts. This tool enables data custodians to transform ambiguous and semi-structured clinical data into a harmonized schema effectively.

![CT_Mapper Diagram](https://github.com/user-attachments/assets/5fa77c82-58ad-4736-bc83-b3a57a33dab4)

## Table of Contents

- [Task Description](#task-description)
- [Installation Requirements](#installation-requirements)
- [Usage](#usage)
  - [Running Experiments on NCBI Dataset](#running-experiments-on-ncbi-dataset)
    - [Standard Inference](#standard-inference)
    - [LLAMA3.1 Inference](#llama31-inference)
- [Contributing](#contributing)
- [License](#license)

## Task Description

To harmonize clinical data effectively, it is crucial to understand the varying levels of conceptual representation within it. Many existing frameworks have attempted to address the challenge of concept linking, focusing primarily on clinical terms with atomic representation. However, in the context of challenges encountered in mapping cohort studies for the ICARE4CVD project, we propose a solution that leverages retrieval-augmented generation and in-context learning.

Instead of mapping individual terms, we focus on mapping data dictionaries presented in a horizontal table format, where each row is treated as a single query. Each query may include multiple components such as labels, descriptions, methods, formulas, units, and categorical values. To standardize and extract clinical terms from each query, we employ in-context learning with generative models like LLAMA and GPT-4. For retrieval, we utilize a hybrid vector search combined with metadata filtering on a structured schema to enhance precision. To further refine the results, we propose a multi-stage ranking method, including a large language model-based cross-ranking approach to filter out irrelevant candidates, followed by relevance-based scoring and relationship prediction. The cumulative score from these steps is used to identify the final candidate.



## Installation Requirements

To run **CT_Mapper**, you need to install the packages mentioned in requirements.in file:

You can install these dependencies using `pip`:

```bash
pip install pandas tqdm torch transformers python-dotenv qdrant-client langchain langchain_openai ctransformers pydantic>=1.10.8 typing-extensions>=4.8.0 torch>=2.2.2 openai>=1.19.0 qdrant-client>=1.8.2 langchain-community togather faiss-cpu faiss-gpu langchain-togather simstring-fast
```
## Usage

Running Experiments on NCBI Dataset
Below are examples of how to run experiments using the {NCBI} dataset or anyother dataset with CT_Mapper. These examples demonstrate both standard inference and using the LLAMA3.1 model for enhanced performance.

## Standard Inference: To perform standard inference on the NCBI dataset, use the following command:
```
PYTHONPATH=/workspace/mapping_tool python3 '/workspace/mapping_tool/rag/vector_index.py' \
  --mode recreate \
  --collection_name ncbi_custom_collection \
  --document_file_path /workspace/mapping_tool/data/eval_datasets/original_ncbi-disease/test_dictionary_docs.jsonl \
  --input_data /workspace/mapping_tool/data/eval_datasets/original_ncbi-disease/combined_test_queries.txt \
  --output_file /workspace/mapping_tool/data/eval_datasets/ncbi-disease_hybrid_not_compressed.txt
```

## Explanation of Parameters:

--mode inference: Sets the mode to recreate. Use 'inference' for existing collection
--collection_name ncbi_custom_collection: Specifies the collection name.
--document_file_path: Path to the JSONL file containing the list of langchain documents.
--input_data: Path to the text file containing combined test queries, format (id||mention)
--output_file: Path where the output results will be saved.

## LLAMA3.1 Inference

For enhanced inference using the LLAMA3.1 model, execute the following command:

```
PYTHONPATH=/workspace/mapping_tool python3 '/workspace/mapping_tool/rag/vector_index.py' \
  --mode inference \
  --collection_name ncbi_custom_collection \
  --document_file_path /workspace/mapping_tool/data/eval_datasets/original_ncbi-disease/test_dictionary_docs.jsonl \
  --input_data /workspace/mapping_tool/data/eval_datasets/original_ncbi-disease/combined_test_queries.txt \
  --output_file /workspace/mapping_tool/data/eval_datasets/ncbi-disease_hybrid_llama3.1_stage1_prompt2.txt \
  --use_llm \
  --llm_id llama3.1
```


## Contributing

Contributions are welcome! Please follow these steps to contribute:

Fork the Repository: Click the "Fork" button at the top-right corner of this page.
Clone Your Fork:
```git clone https://github.com/your-username/CT_Mapper.git ```
Create a New Branch:
```git checkout -b feature/YourFeature```
Make Your Changes: Implement your feature or bug fix.
Commit Your Changes:
```git commit -m "Add your descriptive commit message"```
Push to Your Fork:
```git push origin feature/YourFeature```
Create a Pull Request: Navigate to the original repository and click "New Pull Request".
Please ensure your contributions adhere to the existing code style and include appropriate tests where applicable.

```markdown
![License](https://img.shields.io/badge/license-MIT-blue.svg)


