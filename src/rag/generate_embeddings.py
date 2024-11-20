from rag.utils import load_docs_from_jsonl
from fastembed import TextEmbedding
from transformers import AutoTokenizer, AutoModel
import os
from tqdm.auto import tqdm
import json
import numpy as np
import pickle


def generate_sap_emb():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        clean_up_tokenization_spaces=True,
    )
    model = AutoModel.from_pretrained(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    ).cuda("cuda:2")

    # Load documents
    docs = load_docs_from_jsonl("/workspace/mapping_tool/data/output/concepts.jsonl")

    # Prepare the file to write updated documents incrementally
    with open("/workspace/mapping_tool/data/output/embedding_docs.jsonl", "w") as file:
        count = 0
        for doc in tqdm(docs, desc="Processing Documents"):
            if "synonyms" in doc.metadata:
                synonyms = doc.metadata["synonyms"].split(";;")
                names = [doc.metadata["label"]] + synonyms
                # Compute embeddings for each name
                embeddings = []
                for name in names:
                    toks = tokenizer.encode_plus(
                        name,
                        padding="max_length",
                        max_length=64,
                        truncation=True,
                        return_tensors="pt",
                    )
                    toks_cuda = {k: v.cuda("cuda:2") for k, v in toks.items()}
                    output = model(**toks_cuda)
                    cls_rep = (
                        output[0][:, 0, :].cpu().detach().numpy()
                    )  # Extract CLS token representation
                    embeddings.append(cls_rep)
                # Average the embeddings and add to the document
                avg_embedding = np.mean(embeddings, axis=0)
                doc.metadata["vector"] = avg_embedding.tolist()

                # Write updated document to file
                file.write(json.dumps(doc.to_json()) + "\n")
                count += 1
                print(f"{count} document processed and saved incrementally.")

    print("All documents processed and saved incrementally.")


def save_checkpoint(batch, checkpoint_number):
    """Save each checkpoint batch of docs to a pickle file."""
    file_name = f"/workspace/mapping_tool/data/output/Bge_embedding_checkpoint_{checkpoint_number}.pkl"
    with open(file_name, "wb") as file:
        pickle.dump(batch, file)
    print(f"Checkpoint {checkpoint_number} saved with {len(batch)} documents.")


def generate_document_strings(docs):
    """Generator to yield formatted document strings as needed."""
    for doc in docs:
        # Initialize an empty string for synonyms text
        synonyms_text = "."
        # Check if synonyms exist and format them
        if "synonyms" in doc.metadata and doc.metadata["synonyms"]:
            synonyms_text = (
                f" has synonyms [{', '.join(doc.metadata['synonyms'].split(';;'))}]."
            )

        # Determine the parent term or domain to use in the description
        parent_or_domain = (
            doc.metadata.get("parent_term")
            if doc.metadata.get("parent_term") not in [None, ""]
            else doc.metadata.get("domain", "unknown")
        )
        # Yield the formatted document string
        yield (
            f"{doc.metadata['label']}{synonyms_text} It's parent term in {doc.metadata.get('vocab')} vocabulary is {parent_or_domain}.",
            doc,
        )


def generate_bge_embeddings():
    # Initialize the embedding model
    embedding_model = TextEmbedding()
    print("The model BAAI/bge-small-en-v1.5 is ready to use.")
    docs_path = "/workspace/mapping_tool/data/output/concepts.jsonl"
    docs_generator = load_docs_from_jsonl(docs_path)

    text_batch = []  # This will hold the text for embedding
    doc_batch = []  # This will hold the original documents
    checkpoint_size = 10000
    checkpoint_number = 0
    all_docs_with_embeddings = []
    total_docs_processed = 0
    batch_size = 64
    for formatted_string, original_doc in tqdm(
        generate_document_strings(docs_generator), desc="Processing Documents"
    ):
        text_batch.append(formatted_string)
        doc_batch.append(original_doc)
        total_docs_processed += 1

        if len(text_batch) >= batch_size:
            embeddings = embedding_model.embed(text_batch, parallel=0)
            for doc, embedding in zip(doc_batch, embeddings):
                doc.metadata["vector"] = embedding  # Update the original document
                all_docs_with_embeddings.append(doc)

            text_batch = []  # Reset the text batch after processing
            doc_batch = []  # Reset the document batch after processing

            if len(all_docs_with_embeddings) >= checkpoint_size:
                save_checkpoint(all_docs_with_embeddings, checkpoint_number)
                all_docs_with_embeddings = []  # Reset the accumulated documents after saving
                checkpoint_number += 1

    # Save any remaining embeddings in the final batch
    if all_docs_with_embeddings:
        save_checkpoint(all_docs_with_embeddings, checkpoint_number)

    print(f"Total documents processed: {total_docs_processed}")


def combine_checkpoints(output_path):
    """Combine all checkpoint files into a single file or data structure."""
    checkpoint_files = sorted(
        [
            os.path.join("/workspace/mapping_tool/data/output", f)
            for f in os.listdir("/workspace/mapping_tool/data/output")
            if f.startswith("Bge_embedding_checkpoint_") and f.endswith(".pkl")
        ]
    )
    combined_data = []
    for file_path in checkpoint_files:
        with open(file_path, "rb") as file:
            combined_data.extend(pickle.load(file))

    # Optionally save the combined data back to a single file
    with open(output_path, "wb") as file:
        pickle.dump(combined_data, file)
    print(
        f"All checkpoints combined into {output_path}. Total documents combined: {len(combined_data)}."
    )


generate_bge_embeddings()
combine_checkpoints("/workspace/mapping_tool/data/output/Bge_OMOP_Embeddings.pkl")
