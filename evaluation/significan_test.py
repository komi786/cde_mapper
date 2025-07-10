import json
import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q


def load_model_predictions(model_file):
    print(f"Loading model predictions from {model_file}")
    """
    Load model predictions from a JSON file and extract correctness per query.
    """
    with open(model_file, "r") as f:
        data = json.load(f)

    correctness = []
    for query in data["queries"]:
        for mention in query["mentions"]:
            golden_cui = set(mention["golden_cui"])
            # Assume the model's top prediction is the first candidate
            top_candidate = (
                mention["candidates"][0] if len(mention["candidates"]) > 0 else None
            )
            if not top_candidate:
                continue
            if "id" in top_candidate:
                predicted_cui = set(top_candidate["id"].lower().split("|"))
            elif "labelcui" in top_candidate:
                predicted_cui = set(top_candidate["labelcui"].lower().split("|"))
            # Check if any of the predicted CUIs match the golden CUIs
            is_correct = int(len(golden_cui.intersection(predicted_cui)) > 0)
            correctness.append(is_correct)
    return correctness


def align_predictions(model_correctness):
    """
    Align predictions by retaining only the indices common across all models.
    """
    min_length = min(len(c) for c in model_correctness)
    aligned_correctness = [c[:min_length] for c in model_correctness]
    return aligned_correctness


def calculate_p_value(model_results):
    """
    Perform the Friedman test on the model results and return the p-value.
    """
    statistic, p_value = friedmanchisquare(*model_results)
    return statistic, p_value


def cochran_q_test(model_correctness_list):
    data = np.array(
        model_correctness_list
    ).T  # Transpose to shape (n_samples, n_models)
    result = cochrans_q(data)
    q_statistic = result.statistic
    p_value = result.pvalue
    return q_statistic, p_value


def pairwise_mcnemar_test(base_correctness, compare_correctness):
    both_correct = np.sum((base_correctness == 1) & (compare_correctness == 1))
    base_correct_compare_incorrect = np.sum(
        (base_correctness == 1) & (compare_correctness == 0)
    )
    base_incorrect_compare_correct = np.sum(
        (base_correctness == 0) & (compare_correctness == 1)
    )
    both_incorrect = np.sum((base_correctness == 0) & (compare_correctness == 0))

    contingency_table = [
        [both_correct, base_correct_compare_incorrect],
        [base_incorrect_compare_correct, both_incorrect],
    ]

    result = mcnemar(contingency_table, exact=True)
    return result.statistic, result.pvalue


def pairwise_significance_test(base_correctness, compare_correctness):
    """
    Perform Wilcoxon signed-rank test to compare two models.
    """
    try:
        stat, p_value = wilcoxon(base_correctness, compare_correctness)
        return stat, p_value
    except ValueError as e:
        print(f"Error during pairwise test: {e}")
        return None, None


def main():
    datasets = [
        "hf_studies",
        "miid",
        "ncbi",
        "bc5cdr-d",
    ]  # Replace with actual dataset names
    models = [
        "llama3.1",
        "gpt4omini",
        "gpt4",
        "sapbert",
        "biobert_snomed",
        "krissbert",
    ]  # Replace with actual model names

    # Directory structure: data/<dataset>/<model>.json
    base_dir = "data/eval_datasets/reported_results/reported_in_article/llm_ranking"  # Replace with the actual base directory containing the datasets

    results = []
    pairwise_results = []

    for dataset in datasets:
        model_correctness = []
        for model in models:
            model_file = os.path.join(base_dir, dataset, f"{model}.json")
            correctness = load_model_predictions(model_file)
            model_correctness.append(np.array(correctness))

        # Align predictions if models have different numbers of predictions
        lengths = [len(c) for c in model_correctness]
        if len(set(lengths)) != 1:
            print(
                f"Warning: Models have different number of predictions in {dataset}. Aligning predictions..."
            )
            model_correctness = align_predictions(model_correctness)

        # Perform the Friedman test
        q_statistic, p_value = cochran_q_test(model_correctness)
        print(
            f"Dataset: {dataset}, Friedman statistic: {q_statistic:.4f}, p-value: {p_value:.4f}"
        )
        results.append(
            {"dataset": dataset, "statistic": q_statistic, "p_value": p_value}
        )

        # Perform pairwise Wilcoxon signed-rank tests between llama3.1 and others
        base_correctness = model_correctness[0]  # `llama3.1`
        for i, compare_model in enumerate(models[1:], start=1):
            compare_correctness = model_correctness[i]
            stat, p_value = pairwise_mcnemar_test(base_correctness, compare_correctness)
            if stat is not None:
                pairwise_results.append(
                    {
                        "dataset": dataset,
                        "base_model": "llama3.1",
                        "compare_model": compare_model,
                        "statistic": stat,
                        "p_value": p_value,
                    }
                )

    # Save results to CSV files
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "/workspace/mapping_tool/evaluation/friedman_p_values_per_dataset.csv",
        index=False,
    )
    print("Friedman test results saved to friedman_p_values_per_dataset.csv")

    pairwise_results_df = pd.DataFrame(pairwise_results)
    pairwise_results_df.to_csv(
        "/workspace/mapping_tool/evaluation/pairwise_p_values_per_dataset.csv",
        index=False,
    )
    print("Pairwise comparison results saved to pairwise_p_values_per_dataset.csv")


if __name__ == "__main__":
    main()
