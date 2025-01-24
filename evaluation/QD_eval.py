import time

from llm_chain import extract_information
from py_model import QueryDecomposedModel

from fuzzywuzzy import fuzz
import json


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
        if rel in predicted and rel in ground_truth:
            true_positive_rels += 1
        elif rel in predicted and rel not in ground_truth:
            false_positive_rels += 1
        elif rel not in predicted and rel in ground_truth:
            false_negative_rels += 1

    return true_positive_rels, false_positive_rels, false_negative_rels


# Adjusted function to compute fuzzy matching score for base entity and additional entities
def evaluate_value_fuzzy_match(predicted, ground_truth, threshold=60):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for rel in REQUIRED_RELATIONSHIPS:
        pred_value = predicted.get(rel)
        gt_value = ground_truth.get(rel)
        print(f"pred_value: {pred_value}")
        print(f"gt_value: {gt_value}")
        if (
            rel == "additional_entities" or rel == "base_entity" or rel == "categories"
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
def compute_metrics(data_samples, threshold=60):
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


def filter_result(results):
    #     "domain",
    # "base_entity",
    # "categories",
    # "unit",
    # "additional_entities",
    # "rel",

    new_results = {}
    if "domain" in results and results["domain"] != "None":
        new_results["domain"] = results["domain"]
    if "base_entity" in results and results["base_entity"] != "None":
        new_results["base_entity"] = results["base_entity"]
    if "categories" in results and results["categories"] != "None":
        new_results["categories"] = results["categories"]
    if "unit" in results and results["unit"] != "None":
        new_results["unit"] = results["unit"]
    if "additional_entities" in results and results["additional_entities"] != "None":
        new_results["additional_entities"] = results["additional_entities"]
    # if "rel" in results and results["rel"] != "None":
    #     new_results["rel"] = results["rel"]
    return new_results


def compute_metrics_per_type(data_samples, threshold=60):
    # Initialize metrics for each element type
    metrics_per_type = {
        "ccde": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
        "acde": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
        "dcde": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
    }
    for sample in data_samples:
        ground_truth = sample["ground_truth"]
        predicted = sample["answer"]
        element_type = sample["element_type"]

        # Skip unknown element types
        if element_type not in metrics_per_type:
            continue

        # Evaluate value matching using fuzzy logic
        tp, fp, fn = evaluate_value_fuzzy_match(predicted, ground_truth, threshold)
        metrics_per_type[element_type]["true_positives"] += tp
        metrics_per_type[element_type]["false_positives"] += fp
        metrics_per_type[element_type]["false_negatives"] += fn

    # Compute precision, recall, and F1 for each type
    for element_type, metrics in metrics_per_type.items():
        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(
            f"Element Type: {element_type} | Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}"
        )


start_time = time.time()

with open(
    "/workspace/mapping_tool/data/eval_datasets/custom_data/decompsiition_test.json"
) as json_file:
    query_decomposition = json.load(json_file)


LLM_ID = "llama3.1"
data_samples = []
print(f"len(query_decomposition): {len(query_decomposition)}")
# Process each query decomposition item
for item in query_decomposition:
    input_query = item["input"]
    input_query = QueryDecomposedModel(
        name="input_query",
        full_query=input_query,
        original_label=input_query,
        base_entity=input_query,
    )
    # print(f"input_query: {input_query}")
    # Extract information using your extraction function
    result = extract_information(input_query, model_name=LLM_ID).dict()
    # result.pop("name")
    # result.pop("full_query")
    # result.pop("base_entity")
    # result.pop("formula")
    # result.pop("original_label")
    # result.pop("id")
    print(f"raw result: {result}")
    result = filter_result(result)
    print(f"result: {result}")
    try:
        ground_truth = json.loads(item["output"])
        element_type = item["type"]
    except json.JSONDecodeError as e:
        print(f"Error parsing ground truth for query '{input_query}': {e}")
        ground_truth = {}

    # Add mention to result if it's a dictionary
    if isinstance(result, dict):
        result["mention"] = input_query.full_query
    else:
        # Handle cases where extraction failed or returned unexpected format
        result = {"mention": input_query.full_query}

    # print(f"Mention: {input_query} | Result: {result}")

    # Append the sample to the list
    data_samples.append(
        {
            "question": input_query.full_query,
            "answer": result,
            "ground_truth": ground_truth,
            "element_type": element_type,
        }
    )

# End timing
end_time = time.time()
print(
    f"Time taken for {len(query_decomposition)} queries: {end_time - start_time:.2f} seconds"
)
print(f"len(data_samples): {len(data_samples)}")
threshold = 60
# Save the results to a JSON file
with open(
    f"/workspace/mapping_tool/data/output/query_decomposition_{LLM_ID}_{threshold}.json",
    "w",
) as f:
    json.dump(data_samples, f, indent=4)
# with open(
#     f"/workspace/mapping_tool/data/output/query_decomposition_{LLM_ID}.json", "r"
# ) as f:
#     data_samples = json.load(f)

compute_metrics(data_samples, threshold=threshold)
compute_metrics_per_type(data_samples, threshold=threshold)
