#!/usr/bin/env python3
from rag.llm_chain import extract_information
import json
import time
from rag.py_model import QueryDecomposedModel
from fuzzywuzzy import fuzz

# Define required relationships
REQUIRED_RELATIONSHIPS = [
    "domain",
    "base_entity",
    "categories",
    "unit",
    "additional_entities",
]


# Helper function to evaluate if relationships are predicted correctly
def evaluate_relationships(predicted, ground_truth):
    true_positive_rels = 0
    false_positive_rels = 0
    false_negative_rels = 0

    for rel in REQUIRED_RELATIONSHIPS:
        pred_value = predicted.get(rel, None)
        gt_value = ground_truth.get(rel, None)

        # True positive if both predicted and ground_truth have the same relationship
        if pred_value is not None and gt_value is not None:
            true_positive_rels += 1

        # False negative: The relationship exists in ground_truth but not in predicted (or predicted is None)
        elif pred_value is None and gt_value is not None:
            false_negative_rels += 1

        # False positive: The relationship exists in predicted but not in ground_truth (and is not None)
        elif pred_value is not None and gt_value is None:
            false_positive_rels += 1

    return true_positive_rels, false_positive_rels, false_negative_rels


# Adjusted function to compute fuzzy matching score for base entity and additional entities
def evaluate_value_fuzzy_match(predicted, ground_truth, threshold=80):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for rel in REQUIRED_RELATIONSHIPS:
        pred_value = predicted.get(rel)
        gt_value = ground_truth.get(rel)

        if (
            rel == "additional_entities" or rel == "base_entity"
        ):  # Special case for list comparison
            pred_value = pred_value if isinstance(pred_value, list) else [pred_value]
            gt_value = gt_value if isinstance(gt_value, list) else [gt_value]

            # Check if any base entity or additional entity is present in either set
            for pred_item in pred_value:
                matched = False
                for gt_item in gt_value:
                    if (
                        fuzz.ratio(str(pred_item).lower(), str(gt_item).lower())
                        >= threshold
                    ):
                        true_positives += 1
                        matched = True
                        break
                if not matched:
                    false_positives += 1
                    print(f"False positive: {pred_item}")
            for gt_item in gt_value:
                if all(
                    fuzz.ratio(str(gt_item).lower(), str(pred_item).lower()) < threshold
                    for pred_item in pred_value
                ):
                    false_negatives += 1
        else:  # Single string comparison for other relationships
            if pred_value and gt_value:
                if (
                    fuzz.ratio(str(pred_value).lower(), str(gt_value).lower())
                    >= threshold
                ):
                    true_positives += 1
                else:
                    false_positives += 1
            elif pred_value and not gt_value:
                false_positives += 1
            elif not pred_value and gt_value:
                false_negatives += 1

    return true_positives, false_positives, false_negatives


# Function to compute Precision, Recall, and F1 for both relationships and values
def compute_metrics(data_samples, threshold=80):
    # Initialize counters for relationship prediction
    total_true_positive_rels = 0
    total_false_positive_rels = 0
    total_false_negative_rels = 0

    # Initialize counters for fuzzy value matching
    total_true_positive_values = 0
    total_false_positive_values = 0
    total_false_negative_values = 0

    for sample in data_samples:
        ground_truth = sample["ground_truth"]
        predicted = sample["answer"]

        # Evaluate relationship prediction
        tp_rels, fp_rels, fn_rels = evaluate_relationships(predicted, ground_truth)
        total_true_positive_rels += tp_rels
        total_false_positive_rels += fp_rels
        total_false_negative_rels += fn_rels

        # Evaluate value matching using fuzzy logic
        tp_values, fp_values, fn_values = evaluate_value_fuzzy_match(
            predicted, ground_truth, threshold
        )
        total_true_positive_values += tp_values
        total_false_positive_values += fp_values
        total_false_negative_values += fn_values

    rel_precision = (
        total_true_positive_rels
        / (total_true_positive_rels + total_false_positive_rels)
        if (total_true_positive_rels + total_false_positive_rels) > 0
        else 0
    )
    rel_recall = (
        total_true_positive_rels
        / (total_true_positive_rels + total_false_negative_rels)
        if (total_true_positive_rels + total_false_negative_rels) > 0
        else 0
    )
    rel_f1 = (
        2 * (rel_precision * rel_recall) / (rel_precision + rel_recall)
        if (rel_precision + rel_recall) > 0
        else 0
    )
    print(
        f"Relationship Precision: {rel_precision:.2f}, Recall: {rel_recall:.2f}, F1: {rel_f1:.2f}"
    )

    # Compute metrics for fuzzy value matching
    value_precision = (
        total_true_positive_values
        / (total_true_positive_values + total_false_positive_values)
        if (total_true_positive_values + total_false_positive_values) > 0
        else 0
    )
    value_recall = (
        total_true_positive_values
        / (total_true_positive_values + total_false_negative_values)
        if (total_true_positive_values + total_false_negative_values) > 0
        else 0
    )
    value_f1 = (
        2 * (value_precision * value_recall) / (value_precision + value_recall)
        if (value_precision + value_recall) > 0
        else 0
    )
    print(
        f"Value Precision: {value_precision:.2f}, Recall: {value_recall:.2f}, F1: {value_f1:.2f}"
    )


start_time = time.time()


def perform_extraction(
    file_path="/workspace/mapping_tool/data/eval_datasets/custom_data/decompsiition_test.json",
    llm_model="llama3.1",
):
    with open(file_path) as json_file:
        query_decomposition = json.load(json_file)

    data_samples = []
    print(f"len(query_decomposition): {len(query_decomposition)}")
    for item in query_decomposition:
        input_query = item["input"]
        result: QueryDecomposedModel = extract_information(
            input_query, model_name=llm_model
        )

        result = result.dict() if isinstance(result, QueryDecomposedModel) else {}

        print(f"typeof(result): {type(result)}")
        # Parse ground truth from JSON string to dictionary
        try:
            ground_truth = json.loads(item["output"])
        except json.JSONDecodeError as e:
            print(f"Error parsing ground truth for query '{input_query}': {e}")
            ground_truth = {}

        # Add mention to result if it's a dictionary
        if isinstance(result, dict):
            result["mention"] = input_query
        else:
            # Handle cases where extraction failed or returned unexpected format
            result = {"mention": input_query}

        print(f"Mention: {input_query} | Result: {result}")

        # Append the sample to the list
        data_samples.append(
            {"question": input_query, "answer": result, "ground_truth": ground_truth}
        )
    return data_samples

    # End timing


LLM_ID = "llama3.1"
data_sample = perform_extraction(
    file_path="/workspace/mapping_tool/data/eval_datasets/custom_data/decompsiition_test.json",
    llm_model=LLM_ID,
)
end_time = time.time()
print(f"Time taken for {len(data_sample)} queries: {end_time - start_time:.2f} seconds")
print(f"len(data_samples): {len(data_sample)}")

# Save the results to a JSON file
with open(
    f"/workspace/mapping_tool/data/output/query_decomposition_{LLM_ID}.json", "w"
) as f:
    json.dump(data_sample, f, indent=4)

with open(
    f"/workspace/mapping_tool/data/output/query_decomposition_{LLM_ID}.json", "r"
) as f:
    data_sample = json.load(f)
compute_metrics(data_samples=data_sample, threshold=55)
