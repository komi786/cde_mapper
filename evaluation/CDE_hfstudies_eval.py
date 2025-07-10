# read json files which are model results each file has following format, its dict with key "queries" which is a list of queries where each item is dict with keys "mentions" and value of "mentions" is a dict with keys "golden_cui" and "candidates" and "mention"(mention text)


import json
import numpy as np

from collections import defaultdict


def classify_results_based_on_cde_type(result_json_file, input_text_file):
    data = json.load(open(result_json_file, "r"))
    cde_dict = {}
    with open(input_text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split("||")
            if len(line) == 4:
                cui, name, domain, cde_type = (
                    line[0],
                    line[1].strip().lower(),
                    line[2].strip().lower(),
                    line[3].strip().lower(),
                )
                if cde_type not in cde_dict:
                    cde_dict[cde_type] = []
                cde_dict[cde_type].append((cui, name, domain))
    # print(f"length of cde_dict: {len(cde_dict)}")
    # print(f"length of data: {len(data['queries'])}")
    for key in cde_dict:
        print(f"length of {key}: {len(cde_dict[key])}")

    print(f"json data length: {len(data['queries'])}")

    # divide the json data into cde types
    cde_type_data = {}
    for key in cde_dict:
        cde_type_data[key] = {"queries": []}
    for query in data["queries"]:
        for mention in query["mentions"]:
            for key in cde_dict:
                for cde in cde_dict[key]:
                    if mention["mention"].strip().lower() == cde[1].strip().lower():
                        mentions_list = []
                        mentions_list.append(mention)
                        cde_type_data[key]["queries"].append(
                            {"mentions": mentions_list}
                        )
                        # print(f"cdetype: {key} and mention: {mention}")
                        break
    for key in cde_type_data:
        print(f"length of {key}: {len(cde_type_data[key]['queries'])}")

    return cde_type_data


def check_k(queries):
    # return len(queries[0]["mentions"][0]["candidates"])
    return 10


def dcg(relevances, k):
    """Compute the Discounted Cumulative Gain (DCG) up to position k."""
    relevances = relevances[:k]
    return sum(
        rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)
    )  # idx + 2 because log2 starts at 2


def idcg(relevances, k):
    """Compute the Ideal DCG (IDCG) up to position k."""
    sorted_relevances = sorted(relevances, reverse=True)
    return dcg(sorted_relevances, k)


def calculate_ncgd(candidates, k):
    """Calculate NCGD at k."""
    relevances = [candidate["label"] for candidate in candidates]
    dcg_score = dcg(relevances, k)
    idcg_score = idcg(relevances, k)
    return dcg_score / idcg_score if idcg_score > 0 else 0


def evaluate_ncgd(data):
    top_k_values = [1, 3, 5, 10]
    queries = data["queries"]
    ncgd_at_k = {f"ncgd@{k}": [] for k in top_k_values}

    for query in queries:
        mentions = query["mentions"]
        # print(f"types of mentions: {type(mentions)}")
        for mention in mentions:
            candidates = mention["candidates"]
            # print(f"types of candidates: {type(candidates)}")
            for k in top_k_values:
                ncgd_score = calculate_ncgd(candidates, k)
                ncgd_at_k[f"ncgd@{k}"].append(ncgd_score)

    # Calculate average NCGD@k
    ncgd_at_k_avg = {k: np.mean(v) for k, v in ncgd_at_k.items()}

    # Print NCGD@k results

    # show in percentage and 1 decimal
    ncgd_at_k_avg = {k: v * 100 for k, v in ncgd_at_k_avg.items()}

    for k, ncgd in ncgd_at_k_avg.items():
        print(f"{k}: {ncgd:.1f}")
    # Update data with new metrics
    data.update(ncgd_at_k_avg)
    return data


# Add NCGD evaluation to the existing pipeline


def evaluate_recall_mrr(data, top_k=10):
    """
    Evaluate recall@k and Mean Reciprocal Rank (MRR) for top-k predictions.
    """
    queries = data["queries"]

    # Initialize accumulators for recall@k and MRR
    recall_at_k = {f"recall@{i+1}": [] for i in range(top_k)}
    mrr_scores = []

    for query in queries:
        mentions = query["mentions"]
        for mention in mentions:
            candidates = mention["candidates"]
            # Identify the indices of relevant items
            relevant_indices = [
                idx
                for idx, candidate in enumerate(candidates)
                if candidate["label"] == 1
            ]
            total_relevant = len(relevant_indices)
            retrieved_relevant = 0
            reciprocal_rank = 0

            # Initialize relevant_found_at_k only up to the number of candidates
            relevant_found_at_k = [0] * min(top_k, len(candidates))

            for idx in range(len(candidates)):
                candidate = candidates[idx]
                if candidate["label"] == 1:
                    retrieved_relevant += 1
                    # Update reciprocal rank if it's the first relevant item
                    if reciprocal_rank == 0:
                        reciprocal_rank = 1 / (idx + 1)
                # Update cumulative count of relevant items found up to each k
                if idx < top_k:
                    relevant_found_at_k[idx] = retrieved_relevant

            # Fill in the recall values only up to the number of retrieved candidates
            for k in range(len(relevant_found_at_k)):  # Limit to available candidates
                if total_relevant > 0:
                    recall = relevant_found_at_k[k] / total_relevant
                else:
                    recall = 0
                recall_at_k[f"recall@{k+1}"].append(recall)

            # Append reciprocal rank for MRR calculation
            mrr_scores.append(reciprocal_rank)

    # Calculate average recall@k and MRR
    recall_at_k_avg = {k: np.mean(v) for k, v in recall_at_k.items() if v}
    mrr = np.mean(mrr_scores)

    # show in percentage and 1 decimal
    for k, recall in recall_at_k_avg.items():
        print(f"{k}: {recall*100:.1f}")

    # show  MRR in percentage and 1 decimal
    print(f"MRR: {mrr*100:.1f}")
    # Update data with new metrics
    data.update(recall_at_k_avg)
    data["MRR"] = mrr
    return data


def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data["queries"]
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query["mentions"]
            mention_hit = 0
            for mention in mentions:
                candidates = mention["candidates"][: i + 1]  # to get acc@(i+1)
                mention_hit += np.any([candidate["label"] for candidate in candidates])

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit
            if mention_hit == len(mentions):
                hit += 1

        data["acc{}".format(i + 1)] = hit / len(queries)

    # show in percentage and 1 decimal
    for k in range(1, k + 1):
        print(f"acc@{k}: {data['acc{}'.format(k)]*100:.1f}")
    data = evaluate_precision_at_k(data)
    data = evaluate_recall_mrr(data)
    return data


def evaluate_precision_at_k(data):
    top_k = check_k(data["queries"])
    """
    Evaluate precision@k for top-k predictions.
    """
    queries = data["queries"]
    precision_at_k = defaultdict(list)

    for query in queries:
        mentions = query["mentions"]
        for mention in mentions:
            candidates = mention["candidates"]
            relevant_retrieved = 0

            # Calculate precision at each k
            for idx in range(min(top_k, len(candidates))):
                candidate = candidates[idx]
                if candidate["label"] == 1:
                    relevant_retrieved += 1

                # Precision@k is the proportion of relevant items in top-k results
                precision_at_k[f"precision@{idx+1}"].append(
                    relevant_retrieved / (idx + 1)
                )

            # Fill in remaining k values with 0 if fewer than top_k candidates
            for j in range(len(candidates), top_k):
                precision_at_k[f"precision@{j+1}"].append(0)

    # Calculate average precision@k
    precision_at_k_avg = {k: np.mean(v) for k, v in precision_at_k.items()}
    # show in percentage and 1 decimal
    precision_at_k_avg = {k: v * 100 for k, v in precision_at_k_avg.items()}
    # Print precision@k results
    for k, precision in precision_at_k_avg.items():
        print(f"{k}: {precision:.1f}")

    # Update data with new metrics
    data.update(precision_at_k_avg)
    return data


cde_data = classify_results_based_on_cde_type(
    "data/eval_datasets/reported_results/reported_in_article/llm_ranking/hf_studies/biobert_snomed.json",
    "data/eval_datasets/hf_studies/hf_studies_cdes.txt",
)
for key in cde_data:
    print(f"\nResults for {key.upper()}")
    cde_data[key]["ncgd"] = evaluate_ncgd(cde_data[key])
    cde_data[key]["acc"] = evaluate_topk_acc(cde_data[key])
    cde_data[key]["precision"] = evaluate_precision_at_k(cde_data[key])
    cde_data[key]["recall_mrr"] = evaluate_recall_mrr(cde_data[key])
    # print eval results for each cde type
    # print(f"NCDD results for {key}: {cde_data[key]['ncgd']}")
    # print(f"ACC results for {key}: {cde_data[key]['acc']}")
    # print(f"Precision results for {key}: {cde_data[key]['precision']}")
    # print(f"Recall MRR results for {key}: {cde_data[key]['recall_mrr']}")

    print("\n")
