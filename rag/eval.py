# from langchain_core.documents import Document
from .utils import global_logger as logger
from typing import List, Tuple
# def calculate_accuracy_at_k(ground_truth: List[Tuple[str, str]], retrieved_results: List[dict], k_values=[1, 3, 5, 10]):
#     """
#     Calculate accuracy at different values of k and log incorrect matches.
#     Parameters:
#     - ground_truth (List[Tuple[str, str]]): A list where each item is a tuple containing a query and a single relevant document ID.
#     - retrieved_results (List[dict]): A list of dictionaries, where each dictionary contains details of one query's results.
#     - k_values (List[int]): A list of k values for which to calculate the metrics.

#     Returns:
#     - Dict[int, Dict[str, float]]: A dictionary where keys are k values and values are dicts with accuracy and logs of incorrect matches.
#     """
#     results = {k: {'accuracy': 0, 'incorrect_matches': []} for k in k_values}
#     ground_truth_dict = {query.lower(): answers.replace(',', '|').split('|') if '|' in str(answers) else [answers] for answers, query in ground_truth}

#     for k in k_values:
#         correct_predictions = 0
#         total_queries = len(ground_truth)

#         for result in retrieved_results:
#             query = result['query']
#             if query.lower() not in ground_truth_dict:
#                 continue

#             true_answers = set(ground_truth_dict[query.lower()])
#             retrieved_labels = result['mapping'].split(',')
#             retrieved_codes = [code.split(":")[-1] for code in result['codes'].split(',')]
#             # Check if any retrieved result is in the true answers
#             if any(doc_id in true_answers for doc_id in retrieved_codes):
#                 correct_predictions += 1
#             elif any(doc_name.lower() == query for doc_name in retrieved_labels):
#                 correct_predictions += 1
#             else:
#                 results[k]['incorrect_matches'].append((query, retrieved_codes))

#         # Calculate accuracy for this k
#         if total_queries > 0:
#             accuracy = correct_predictions / total_queries
#         else:
#             accuracy = 0
#         results[k]['accuracy'] = accuracy

#         # Optional: Output the results for diagnostics
#     for k in k_values:
#         logger.info(f"Accuracy at k={k}: {results[k]['accuracy']:.2f}")
#         if results[k]['incorrect_matches']:
#             logger.info(f"Incorrect Matches at k={k}:")
#             for match in results[k]['incorrect_matches']:
#                 logger.info(f"  Query: {match[0]}, Retrieved: {match[1]}")
#     logger.info(f"R:P:F1={calculate_precision_recall_f1(ground_truth, retrieved_results, k_values=[1, 3, 5])}")
#     logger.info(f"MRR={calculate_mrr(ground_truth, retrieved_results)}")
#     logger.info(f"NDCG={calculate_ndcg(ground_truth, retrieved_results)}")

#     return results

# def calculate_precision_recall_f1(ground_truth: List[Tuple[str, str]], retrieved_results: List[dict], k_values=[1, 3, 5]):
#     metrics = {k: {'precision': 0, 'recall': 0, 'f1': 0} for k in k_values}

#     ground_truth_dict = {query.lower(): set(answers.replace(',', '|').split('|')) for  answers, query in ground_truth}

#     for k in k_values:
#         true_positives = 0
#         total_retrieved = 0
#         total_relevant = sum(len(v) for v in ground_truth_dict.values())

#         for result in retrieved_results:
#             query = result['query'].lower()
#             if query not in ground_truth_dict:
#                 continue

#             true_answers = ground_truth_dict[query]

#             retrieved_docs = [code.split(":")[-1] for code in result['codes'].split(',')]
#             retrieved_docs_names = result['mapping'].split(',')

#             tp = sum(1 for doc_id in retrieved_docs if doc_id in true_answers)
#             tp += sum(1 for doc_name in retrieved_docs_names if doc_name == query)  # Checking names
#             true_positives += tp
#             total_retrieved += len(retrieved_docs)

#         precision = true_positives / total_retrieved if total_retrieved > 0 else 0
#         recall = true_positives / total_relevant if total_relevant > 0 else 0
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#         metrics[k]['precision'] = precision
#         metrics[k]['recall'] = recall
#         metrics[k]['f1'] = f1

#     return metrics


# def calculate_mrr(ground_truth: List[Tuple[str, str]], retrieved_results: List[dict]):
#     ground_truth_dict = {query.lower(): set(answers.replace(',', '|').split('|')) for answers, query in ground_truth}
#     mrr = 0
#     queries_count = 0

#     for result in retrieved_results:
#         query = result['query'].lower()
#         if query not in ground_truth_dict:
#             continue

#         true_answers = ground_truth_dict[query]
#         retrieved_docs_labels = [code.split(":")[-1] for code in result['codes'].split(',')]

#         for rank, label in enumerate(retrieved_docs_labels, start=1):
#             if label.lower() in true_answers:
#                 mrr += 1 / rank
#                 break
#         queries_count += 1

#     return mrr / queries_count if queries_count > 0 else 0

# def calculate_ndcg(ground_truth: List[Tuple[str, str]], retrieved_results: List[dict], k=10):
#     ground_truth_dict = {query.lower(): set(answers.split('|')) for query, answers in ground_truth}
#     ndcg_values = []

#     for result in retrieved_results:
#         query = result['query'].lower()
#         if query not in ground_truth_dict:
#             continue

#         true_answers = ground_truth_dict[query]
#         retrieved_docs_labels = [code.split(":")[-1] for code in result['codes'].split(',')]

#         idcg = 1 + sum(1 / np.log2(i + 2) for i in range(min(len(true_answers), k)))
#         dcg = sum(1 / np.log2(i + 2) for i, label in enumerate(retrieved_docs_labels[:k], start=1) if label.lower() in true_answers)

#         ndcg = dcg / idcg if idcg > 0 else 0
#         ndcg_values.append(ndcg)

#     return np.mean(ndcg_values) if ndcg_values else 0


def evaluate_with_multiple_mappings(
    ground_truth: List[Tuple[str, str]],
    retrieved_results: List[dict],
    k_values=[1],
    model_name="llama",
):
    """
    ground truth is list of tuples with query and answers where each query can have multiple answers and each answer is the standard code from OMOP vocabulary
    retrieved_results is list of dictionaries with query_text, standard_concept_id, additional_context_concept_ids, categorical_concept_ids

    evaluate the accuracy to check if the retrieved results (combined from standard_concept_id, additional_context_concept_ids, categorical_concept_ids) are in the ground truth

    """
    results = {
        k: {"accuracy": 0, "precision": 0, "recall": 0, "incorrect_matches": []}
        for k in k_values
    }
    ground_truth_dict = {
        query: set(str(answers).lower().replace(",", "|").strip().split("|"))
        for item in ground_truth
        for answers, query in [item[:2]]  # Unpack only the first two elements
    }
    # logger.info(f"results=\n{retrieved_results}")
    #  "VARIABLE NAME": result_object.variable_name,
    #         "VARIABLE LABEL": result_object.original_query,
    #         "Domain": result_object.domain,
    #         "Variable Concept Label": main_term_labels,
    #         "Variable Concept Code": main_term_codes,
    #         "Variable OMOP ID": main_term_omop_id,
    #         "Additional Context Concept Label": additional_entities_labels,
    #         "Additional Context Concept Code": additional_entities_codes,
    #         "Additional Context OMOP ID": additional_entities_omop_ids,
    #         "Primary to Secondary Context Relationship": result_object.primary_to_secondary_rel,
    #         "Categorical Values Concept Label": categorical_values_labels,
    #         "Categorical Values Concept Code": categorical_values_codes,
    #         "Categorical Values OMOP ID": categorical_values_omop_ids,
    #         "Unit Concept Label": result_object.unit_matches[0].standard_label
    for k in k_values:
        correct_predictions = 0
        total_queries = 0
        total_relevant_retrieved = 0
        total_relevant = 0
        total_retrieved = 0

        for result in retrieved_results:
            query = result.get("VARIABLE LABEL", "").strip().lower()
            # print(f"query={query} have result={result}")
            if query not in ground_truth_dict:
                logger.info(f"Query not found: {query}")
                continue
            total_queries += 1
            true_answers = ground_truth_dict[query]
            total_relevant += len(true_answers)
            # removed code.split(":")[-1]  because we use omop ids
            # dynamic_k = len(true_answers) + k if len(true_answers) > k else k
            # status_codes = []
            values_codes = []
            retrieved_codes = []

            # Handle categorical_concept_ids

            # print(f"status_codes={status_codes}")

            # Handle additional_context_concept_ids

            # print(f"values_codes={values_codes}")

            # Handle standard_concept_id
            standard_concept_id = result.get("Variable OMOP ID", "")
            if "|" in standard_concept_id:
                retrieved_codes = standard_concept_id.split("|")
            else:
                if standard_concept_id:  # Add only if it's not empty
                    retrieved_codes.append(standard_concept_id)
            additional_context_concept_ids = result.get(
                "Additional Context OMOP ID", ""
            )
            if "|" in additional_context_concept_ids:
                values_codes = additional_context_concept_ids.split(";;")
            else:
                if additional_context_concept_ids:  # Add only if it's not empty
                    values_codes.append(additional_context_concept_ids)
            retrieved_codes.extend(values_codes)
            retrieved_answers = set(retrieved_codes)

            total_retrieved += len(retrieved_answers)
            relevant_retrieved = retrieved_answers & true_answers
            # total_relevant_retrieved += len(relevant_retrieved)
            relevant_retrieved = any(code in retrieved_answers for code in true_answers)
            total_relevant_retrieved += sum(
                code in retrieved_answers for code in true_answers
            )
            if (
                relevant_retrieved
            ):  # Check if there is an intersection between the two sets
                correct_predictions += 1
            else:
                # logger.info(f"In Correct: Query: {query}, Retrieved: {retrieved_answers}, Expected: {true_answers}")
                results[k]["incorrect_matches"].append((query, list(retrieved_answers)))

        accuracy = correct_predictions / total_queries if total_queries > 0 else 0
        # precision = (
        #     total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
        # )
        # recall = total_relevant_retrieved / total_relevant if total_relevant > 0 else 0
        results[k]["accuracy"] = accuracy
        # results[k]['precision'] = precision
        # results[k]['recall'] = recall

    for k in k_values:  # Moved this loop out of the main k loop
        logger.info(f"Metrics at k={k}:")
        logger.info(f"  Accuracy: {results[k]['accuracy']:.2f}")
        # logger.info(f"  Precision: {results[k]['precision']:.2f}")
        # logger.info(f"  Recall: {results[k]['recall']:.2f}")  # does not apply as we select top1 but ground truth may have multiple answers

    return results


# # Example usage
# ground_truth = [('answer1|answer2', 'query1'), ('answer3', 'query2')]
# retrieved_results = [
#     {'query_text': 'query1', 'standard_code': 'code1:answer1|code2:wrong', 'additional_context_codes': 'context1:answer2;;context2:irrelevant'},
#     {'query_text': 'query2', 'standard_code': 'code3:answer3|
