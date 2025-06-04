# Step 1: Import Libraries and Load Models
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertForMaskedLM
import torch
import torch.nn as nn


def combine_models(model_name_1, model_name_2):
    
    # Define the model names
    model_name_1 = 'bert-base-uncased'
    model_name_2 = 'roberta-base'

    # Load the pre-trained models
    model1 = AutoModelForMaskedLM.from_pretrained(model_name_1)
    model2 = AutoModelForMaskedLM.from_pretrained(model_name_2)

    # Load the corresponding tokenizers
    tokenizer1 = AutoTokenizer.from_pretrained(model_name_1)
    tokenizer2 = AutoTokenizer.from_pretrained(model_name_2)

    # Step 2: Extract Embeddings
    # Get the embeddings from each model
    embeddings1 = model1.base_model.embeddings.word_embeddings.weight
    embeddings2 = model2.base_model.embeddings.word_embeddings.weight

    # Step 3: Align Embedding Dimensions (if necessary)
    # Check if embedding dimensions match
    dim1 = embeddings1.shape[1]
    dim2 = embeddings2.shape[1]

    # Find the common embedding dimension
    common_dim = max(dim1, dim2)

    # If dimensions are different, project them to the common dimension
    if dim1 != common_dim:
        linear1 = nn.Linear(dim1, common_dim)
        embeddings1 = linear1(embeddings1)

    if dim2 != common_dim:
        linear2 = nn.Linear(dim2, common_dim)
        embeddings2 = linear2(embeddings2)

    # Step 4: Combine the Embeddings
    # Example: Average the embeddings
    combined_embeddings = (embeddings1 + embeddings2) / 2

    # Step 5: Create a New Model with Combined Embeddings
    # Use the configuration from one of the models
    config = BertConfig(
        vocab_size=combined_embeddings.shape[0],
        hidden_size=common_dim,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )

    # Initialize a new model with the combined configuration
    new_model = BertForMaskedLM(config)

    # Replace the embeddings with the combined embeddings
    new_model.base_model.embeddings.word_embeddings = nn.Embedding.from_pretrained(combined_embeddings)

    # Step 6: Save the New Model and Tokenizer
    # Choose one tokenizer to save (or create a new one if needed)
    tokenizer1.save_pretrained('combined_model')
    new_model.save_pretrained('combined_model')

    # Step 7: Load and Use the New Model
    # Load the new model and tokenizer
    loaded_model = BertForMaskedLM.from_pretrained('combined_model')
    loaded_tokenizer = AutoTokenizer.from_pretrained('combined_model')

    # Example usage
    input_text = "The quick brown fox jumps over the lazy [MASK]."
    inputs = loaded_tokenizer(input_text, return_tensors='pt')
    outputs = loaded_model(**inputs)
    logits = outputs.logits

    # Get the predicted token
    mask_token_index = (inputs.input_ids == loaded_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    predicted_token = loaded_tokenizer.decode(predicted_token_id)

    print(f"Predicted token: {predicted_token}")
