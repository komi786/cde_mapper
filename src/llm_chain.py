from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from .utils import *
from langchain.callbacks.tracers import ConsoleCallbackHandler
from .utils import global_logger as logger
from pydantic.v1 import ValidationError
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from typing import List, Dict

from langchain_core.prompts.chat import MessagesPlaceholder
import numpy as np
from scipy.stats import skew
from langchain.globals import set_llm_cache
from collections import defaultdict
from langchain_community.cache import InMemoryCache
from .param import MAPPING_FILE
from .manager_llm import *
set_llm_cache(InMemoryCache())
parsing_llm = LLMManager.get_instance('llama3.1')
parser = JsonOutputParser ()
fixing_parser = OutputFixingParser.from_llm(parser = parser, llm = parsing_llm, max_retries=3)
# import langchain
# langchain.verbose = True
# # example_selector: Optional[SemanticSimilarityExampleSelector] = None
# class CustomHandler(BaseCallbackHandler):
#     def on_llm_start(
#         self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
#     ) -> Any:
#         formatted_prompts = "\n".join(prompts)
#         _log.info(f"Prompt:\n{formatted_prompts}")

# def is_history_relevant(history_message, query):
#     content = history_message.content
#     if match := re.match(r'query:(.*?), output:(.*)', content):
#          input_ = match.group(1).strip()
#          score = reranker.compute_score([input_,query],normalize=True)
#          logger.info(f"History Relevance score={score}")
#          if score > 0.9:
#              return True
#     return False
def get_relevant_examples(query: str, content_key:str, examples: List[Dict[str, str]],topk=3,min_score=0.5) -> List[Dict]:
    
    try:
        # Obtain the singleton example selector
        if examples is None:
            logger.error("No examples found")
            return []
        selector =  ExampleSelectorManager.get_example_selector(content_key,examples,k=topk,score_threshold=min_score)
        
        selected_examples = selector.select_examples({"input": f"{query}"})

        return selected_examples
    except Exception as e:
        logger.info(f"Error in get_relevant_examples: {e} for query:{query} and content_key:{content_key}")
        return []


chat_history = []
# chat_history = load_chat_history(CHAT_HISTORY_FILE)
from langchain_core.messages import HumanMessage
def extract_information(query, model_name=LLM_ID, prompt=None):
    if query:
        global chat_history
        try:
            active_model = LLMManager.get_instance(model=model_name)
            mapping_for_domain, _,_ = load_mapping(MAPPING_FILE, 'all')
            if mapping_for_domain is None:
                logger.error("Failed to load mapping for domain")
                return None
            examples = mapping_for_domain['examples']
            select_examples = get_relevant_examples(query,'extract_information', examples, topk=2, min_score=0.6)
            if select_examples is None:
                logger.error("No relevant examples found")
                select_examples = []
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                examples= select_examples,
                example_prompt=ChatPromptTemplate.from_messages(
                [("human", "{input}"), ("ai", "{output}")]
                ),
                input_variables=["input"]
            )
            if prompt:
                base_prompt = prompt
            else:
                base_prompt="""Role: You are a helpful assistant with expertise in data science and the biomedical domain.
                ***Task Description:
                    - Extract information from the provided medical query to link it to OHDSI OMOP controlled vocabularies.
                ** Perform the following actions in order to identify relevant information:
                    -Identify if there are any acronyms and abbreviations in given medical query and expand them.
                    -Domain: Determine the most appropriate OHDSI OMOP standards from list of following domains: [Condition, Anatomic Site, Body Structure, Measurement, Procedure, Drug, Device, Unit,  Visit,  Death,  Demographics, Family History, Life Style, History of Events].
                    -Base Entity: The primary concept or entity mentioned in the medical query.
                    - Associated Entities: Extract associated entities related to the base entity.
                    - Unit: Unit of measurement associated if mentioned.
                    - Status: If mentioned, list all provided categorical values which represents the status of the base entity.
                **Considers:
                    -Translate all visits with time indicators as follow-up month
                    -Don't consider status values as context. Assume they are categorical values.
                    -Don't add additional unit of measurement if not mentioned in the query.
                **Check Examples: If examples are provided, use them to guide your extraction. If no examples or relevant examples are provided, generate new examples to aid the extraction process.
                **Desired format: Provide the output in JSON format with the following fields: 'domain', 'base_entity', 'additional_entities', 'status' and 'unit'.
                Don't add any preamble or explanations. Use examples if given as a guide to understand how and what information to extract.
                    medical query: {input}
                    Output: 
                        """
            final_prompt = (
                    SystemMessagePromptTemplate.from_template(
                        base_prompt
                    )
                    +few_shot_prompt
                    +HumanMessagePromptTemplate.from_template("{input}")
                )
            chain = final_prompt | active_model
            result =  chain.invoke({"input": query})
            # print(f"initial extract.llm result={result}")
            if not isinstance(result, dict):
                try:
                    result = fixing_parser.parse(result.content)
                    if result is None:
                        return None
                    chat_history.extend([HumanMessage(content=f"query:{query}, output:{result}")])
                  
                    return result
                except ValidationError as e:
                    logger.info(f"Validation Error: {e}")
                    result = None
            else:
                chat_history.extend([HumanMessage(content=f"query:{query}, output:{result}")])
            generate_information_triples(result, model_name=model_name)
            return result
            
        except Exception as e:

            logger.info(f"Error in prompt:{e}")
            return None
    else:
        return None
    
def generate_information_triples(query, model_name='llama3.1'):
    try:
            # random.shuffle(chat_history)
            active_model = LLMManager.get_instance(model=model_name)
            human_template="""***Task Description:
                - Given the medical query, its categorical values and the unit of measurement, generate information triples.
                - Each triple should consist of the following components: 'subject', 'predicate', and 'object'. The 'subject' should be the main entity, the 'predicate' should be the relationship, and the 'object' should be the associated entity.
            ** Perform the following actions in order to generate information triples:
                - Determine the domain of the medical query based on OHDSI OMOP standards e.g.  condition, anatomic site, body structure, measurement, procedure, drug, device, unit, visit, death, demographics, family history, life style, or history of events.
                - Determine the base entity from the medical query.
                - Find additional context that aids in understanding the base entity and infer relationships between them.
                - If unit of measurement is provided, include it in the triple with the appropriate relationship to the base entity.
                - If status values are provided, include them in the triple with the appropriate relationship to the base entity.
            ** Desired Format: Provide the output in List of dictionaries format with the following fields: 'subject', 'predicate', and 'object'.
            Input: {input}
            """
            system = "You are a helpful assistant with expertise in data science and the biomedical domain."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human_template)], template_format='mustache')
            chain = prompt | active_model
            chain_results =  chain.invoke({"input": query})
            save_triples_to_txt(query, chain_results, "/workspace/mapping_tool/data/gissi_llama_triples.txt")
    except Exception as e:
        logger.info(f"Error loading LLM: {e}")
            
def save_triples_to_txt(query, triples, output_file):
    with open(output_file, 'a') as f:
        for triple in triples:
            f.write(f"{query}\t{triple['subject']}\t{triple['predicate']}\t{triple['object']}\n")
            
# def find_domain(query):
#     # Human-readable template for the assistant to categorize the input into OHDSI OMOP domains
#     # print(f"original query={query}")
#     human_template = f"""
#     **Categorize the input into one of the following domains based on OHDSI OMOP standards:
#         1. [Unit (e.g., 'Milligram per deciliter, kg/m2, pmol/L'),  
#         2. Condition (e.g., 'Diabetes mellitus, Radial styloid tenosynovitis'), 
#         3. Anatomic Site (e.g., 'Liver, Lead site V3'), 
#         4. Measurement (e.g., 'Lab tests like Cholesterol in Serum or Plasma, Staging/Scaling like Massachusetts Stroke Scale (MSS), Biomarkers like Cholestanol, Measurement values like maximum blood pressure or cholesterol levels'), 
#         5. Procedure (e.g., 'Appendectomy, Echocardiography'), 
#         6. Drug (e.g., 'Aspirin, Insulin, Diuretics'), 
#         7. Device (e.g., 'Pacemaker, Glucose monitor'), 
#         8. Visit (e.g., patient hospital id, 'Outpatient visit, Emergency room visit, follow-up visit, post-operative discharge follow-up'), 
#         9. Death (e.g., 'Death certificate, Cause of death'), 
#         10. Demographics (e.g., 'Age, Gender, Marital status, Race'), 
#         11. Family History (e.g., 'Family history of cancer (mother), Family history of heart disease'), 
#         12. Life Style (e.g., 'Smoking status, Alcohol consumption'), 
#         13. History of Events (e.g., 'Previous stroke, prior surgical interventions')].
#         14. Observation (e.g., 'patient enrolled in trial, patient has a fever, medication adherence')
        
#     **Instructions:
#         1. Interpret the input in a clinical or medical context.
#         2. Only if needed, rewrite the input query ensuring  or shorthand are expanded to their full forms (e.g., 'pmol/L' to 'picomole per liter', 'pfhb2' to 'Progressive Familial Heart Block, Type II' etc.).
#         4. Categorize the rewritten input into one of the above domains. If unclear, categorize as 'Unknown'.
#     ** Desired Format: Respond with a only JSON object containing the following structure:
#     {{
#         "domain": "<category>",
#         "query": "<standardized_query>"
#     }}
#     Don't add any preamble or explanations.
#     Input: {query}
#     """

#     # System message to ensure the assistant understands its role
#     system = "You are a helpful assistant expert in medical/clinical domain and designed to output JSON responses with the domain and standardized query."

#     # Combine system and human template into a prompt
#     template = ChatPromptTemplate.from_messages(
#         [
#             ("system", system), 
#             ("human", human_template)
#         ], 
#         template_format='mustache'
#     )

#     # Execute the template using the appropriate language model manager (LLMManager)
#     chain = template | LLMManager.get_instance('llama') | JsonOutputParser()
    
#     # Run the query through the chain and get the result
#     result = chain.invoke({"input": query})
#     print(f"find_domain result={result}")
    
#     # Check if the result is a dictionary and return domain and query in JSON format
#     if isinstance(result, dict):
#         return {
#             "domain": result.get("domain", "Unknown").lower(), 
#             "base_entity": result.get("query", query)
#         }
#     else:
#         result =  fix_json_quotes(result.content)
#         if result:
#             return {
#                 "domain": result.get("domain", "Unknown").lower(), 
#                 "base_entity": query
#             }
#         else:
#             return {
#                 "domain": "Unknown", 
#                 "base_entity": query
#             }


# def pass_mapping_guideline_to_llm(query, mapping_guideline,llm=None,llm_name='llama'):
#     try:
#         prompt_cot = f"""{mapping_guideline['prompt']}. 
#         Respond only with information explicitly mentioned in the text, and do not infer or add details not present in the query.
#         Use these examples as a guide to understand how and what information to extract.\n
#         input: {input}\n,
#         output: """
#         examples = mapping_guideline['examples']
#         example_prompt = ChatPromptTemplate.from_messages(
#             [
#                 MessagesPlaceholder(variable_name="chat_history")
#                 ("human", "{input}"),
#                 ("ai", "{output}"),
#             ]
#         )
#         few_shot_prompt = FewShotChatMessagePromptTemplate(
#             example_prompt=example_prompt,
#             examples=examples,
#             # input_variable = ["input","target_language"]
#             # partial_variables={"format_instructions":format_instructions},
#         )
    
#         final_prompt = (
#             SystemMessagePromptTemplate.from_template(
#                 prompt_cot
#             )
#             + few_shot_prompt
#             + HumanMessagePromptTemplate.from_template("{input}")
#         )
#         # structured_llm = llm.with_structured_output(MappedTerms,method="json_mode")
#         # logger.info(f"structured_llm={structured_llm}")

#         # parser = JsonOutputParser()
#         chain = final_prompt | llm 
#         response =  chain.invoke({"input": query})
#         if not isinstance(response, dict):
#             try:
#                 # if llm_name == 'llama' or 'gpt' in llm_name:
#                 #     response= fixing_parser.parse(response.content)
#                 # else:
#                 #     response = get_json_output(response.content)
#                 response = fixing_parser.parse(response.content)
#                 return response
#             except ValidationError as e:
#                 logger.info(f"Validation Error: {e}")
#                 response = None
#     except Exception as e:
#         logger.info(f"Error in pass_mapping_guideline_to_llm: {e}")
#         response = None
#         return response
# """            
# When assigning scores, consider the relevance of each candidate and how they compare with each other in terms of relevance to the query.
#             Also, take into account the semantic type of each answer, as certain types may be more appropriate or suitable for the query context. Also, take into account the semantic type of each answer, 
#             as certain types may be more appropriate or suitable for the query context. For measurements, prioritize lab test:measurement. For medical conditions, prioritize disorder:condition. For medications, prioritize drug, 
#             For demographics, prioritize observable entity:observation etc.
# """

def generate_link_prediction_prompt(query,documents, domain=None, in_context=True):
    if in_context:
        _,_,link_prediction_examples = load_mapping(MAPPING_FILE, 'all')
        # logger.info(f"{len(link_prediction_examples)}:link prediction examples loaded")
        examples = get_relevant_examples(query,'link_prediction',link_prediction_examples, topk=2, min_score=0.6)
        # logger.info(f"selected_examples for Link Prediction={examples}")
        
        # human_template = f"""
        # Task: What is the relationship between medical query : {query} and each candidate terms from Standard Medical Terminologies/vocabularies:{documents}. Categorize the relationship, between medical query and candidate term based on their closeness in meaning as one of the following: ['synonym','highly relevant', 'partially relevant', 'not relevant'].
        # A candidate term should be categorized as an 'exact match' only if it completely and accurately represents the medical query in meaning. For each candidate term, provide a brief justification of your chosen relationship category, focusing on their relevance, closeness in meaning and specificity in the context of the query.
        # Examples: Follow the examples provided to understand how to categorize the relationship between the medical query and candidate terms.
        # Response Format: Please format your response as a list of dictionaries, each containing the keys "answer", "relationship", and "explanation". 
        # Ensure your response adheres to a valid JSON schema. Begin your response with the word '[' and include no extra comments or information.
        # """
        human_template= """
        Task: Determine the relationship between a given medical query and candidate terms from standard medical terminologies aka. vocabularies (SNOMED, LOINC, MeSH, UCUM, ATC, RxNorm, OMOP Extension etc). You must determine relationship of each candidate term with given medical query in clinical/medical context.
        **Instructions:
            Medical Query: {query}
            Candidate Terms: {documents}
        ** Categorization Criteria:
            Exact Match: The term is identical in meaning and context to the query.
            Synonym: The term has the same meaning as the query but may be phrased differently.
            Highly Relevant: The term is very closely related to the query but not synonymous.
            Partially Relevant: The term is broadly related to the query but there are significant differences.
            Not Relevant: The term has no significant relation to the query.
        
        **Task Requirements: Answer following questions to determine the relationship between the medical query and candidate terms:
                -Does the term accurately represent the query in meaning?
                -Is there any term that is an exact match to the query?
                -If all terms are specific than the query, which one is the closest match?
                -If all terms are broad or generic, which one is the most relevant to determine exact match?
        Provide a brief justification for your categorization, focusing on relevance, closeness in meaning, and specificity in the context of the query.Do not assign higher scores just because there is not a perfect or accurate match.
        Check Examples: Determine if examples are provided. If examples are provided and aligned with the current medical query, use them to guide your categorization. If they are provided but not aligned, create new relevant examples using the same format. If no examples are provided, generate new examples to illustrate how to categorize the relationships.
        **Desired format: Your response should strictly adhere to a valid JSON schema as a list of dictionaries, each containing the keys "answer", "relationship", and "explanation". Don't add any preamble or additional comments.
        """
        system = "You are a helpful assistant with expertise in clinical/medical domain and designed to respond in JSON"
        example_prompt = ChatPromptTemplate.from_messages(
            [
            ("system", system), ("human", human_template)
            ], template_format='mustache'
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples
        )
        final_prompt = (
            SystemMessagePromptTemplate.from_template(
                system
            )
            + few_shot_prompt
            + HumanMessagePromptTemplate.from_template(human_template)
        )
        # logger.info(f"final_prompt={final_prompt}")   
        return final_prompt
    else:
        human_template = f"""
        What is the relationship between medical query : {query} and each candidate term from Standard Medical Terminologies/vocabulariess:{documents}. Categorize the relationship, between medical query and candidate term based on their closeness in meaning as one of the following: [synonym','highly relevant', 'partially relevant', 'not relevant'].
        A candidate term should be categorized as an 'synonym' only if it completely and accurately represents the medical query in meaning. For each candidate term, provide a brief justification of your chosen relationship category, focusing on the broder or specific  and relevance of the answer. 
        Please format your response as a list of dictionaries, each containing the keys "answer", "relationship", and "explanation". 
        Ensure your response adheres to a valid JSON schema. Begin your response with the word '[' and include no extra comments or information.
            """
        system = "You are a helpful assistant expert in medical domain and designed to output JSON"
        return ChatPromptTemplate.from_messages([("system", system), ("human", human_template)], template_format='mustache')
            
def generate_ranking_prompt(query,documents,domain=None,in_context=True):
    if in_context:
        _, ranking_examples,_ = load_mapping(MAPPING_FILE, domain=domain)
        print(f"{len(ranking_examples)}:ranking examples loaded")
        examples = get_relevant_examples(query,'ranking',ranking_examples,topk=1, min_score=0.6)
        # logger.info(f"selected_examples for Ranking Prediction={examples}")
        human_template = """Objective: Rank candidate terms from the Standard Medical Terminologies/vocabularies(SNOMED, LOINC, MeSH, ATC, UCUM, RxNorm, OMOP Extension) based on their relevance and closeness in meaning to a given medical query.
            **Instructions: For each given candidate term, please evaluate its relevance and closeness in meaning in medical/clinical context to the given query on a scale from 0 to 10 where,
                -10: The candidate term is an accurate and an exact match/synonym to the input.
                -0: The candidate term is completely irrelevant to the query.
            **Reasoning: Ask yourself the following questions before assigning a score:
                -Is there any term that is an exact match to the query? Does the term fully capture the intended concept expressed in the query?
                -If all terms are specific than the query, which one is the closest match?
                -If all terms are broad or generic, which one is the most relevant to determine exact match?
            
            **Examples: if provided Follow the examples to understand how to rank candidate terms based on their relevance to the query.
            **Desired format: Your response should strictly adhere to a valid JSON schema as a list of dictionaries, each containing the keys "answer", "score", and "explanation". Don't add any preamble or additional comments.
            Input: {query}
            Candidate Terms: {documents}
            Ranked answers:
            """
        system = "You are a helpful assistant expert in medical domain and designed to output JSON"
        example_prompt = ChatPromptTemplate.from_messages(
        [
           ("system", system), ("human", human_template)
        ], template_format='mustache'
    )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
        # partial_variables={"format_instructions":format_instructions},
    )
        final_prompt = (
        SystemMessagePromptTemplate.from_template(
            system
        )
        + few_shot_prompt
        + HumanMessagePromptTemplate.from_template(human_template)
    )
        
        # logger.info(f"final_prompt={final_prompt}")   
        return final_prompt
    else:
         human_template = """Objective: Rank candidate terms from the Standard Medical Terminologies/vocabularies based on their relevance  and closeness in meaning to a given medical query.
            Instructions: For each given candidate term, please evaluate its relevance and closeness in contextual meaning to the given query on a scale from 0 to 10 where,
                -10 indicates that system answer is an accurate and an exact match(synonym) to the input.
                -0: The term is completely irrelevant to the query.
            Provide a brief justification for each score, explaining why the assigned score was chosen.Focus on the following aspects:
                -Specificity: How closely does the term align with the specific details of the query?
                -Conceptual Match: Does the term capture the intended concept expressed in the query, even if it's not a direct match?
                -Ambiguity: Does the term have multiple meanings that could lead to misinterpretation in the context of the query?
            Your response should strictly adhere to a valid JSON schema as a list of dictionaries, each containing the keys "answer", "score", and "explanation". Don't add any preamble or additional comments.
            Input: {query}
            Candidate Terms: {documents}
            Ranked answers:
            """
         system = "You are a helpful assistant"

         template_ = ChatPromptTemplate.from_messages([("system", system), ("human", human_template)], template_format='mustache')
        #  print(f"template={template_.format_messages()}")
         return template_


def adjust_percentile(scores, base_percentile=75):
    score_skewness = skew(scores)
    if score_skewness > 1:  # highly skewed distribution
        adjusted_percentile = base_percentile - 3  # lower the percentile slightly
    elif score_skewness < -1:  # highly negatively skewed
        adjusted_percentile = base_percentile + 3  # increase the percentile slightly
    else:
        adjusted_percentile = base_percentile
    return np.percentile(scores, adjusted_percentile)

def calculate_dynamic_threshold(scores, base_threshold, exact_match_found):
    if not scores:
        return 0.0

    max_score = max(scores) if scores else 1
    if max_score == 0:
        # Handle the case where all scores are zero
        # Possibly return a zero threshold or a decision that no candidates are valid
        # logger.info("All scores are zero, returning zero threshold")
        return 0.0
    normalized_scores = [score / max_score for score in scores]
    # Use a higher base threshold if an exact match is found
    if exact_match_found:
        base_threshold = max(base_threshold, 8)  # Example value, adjust as needed

    # Calculate percentile as the belief threshold
    belief_threshold = adjust_percentile(normalized_scores) 
    
    # logger.info(f"Base Threshold={base_threshold}, Belief Threshold={belief_threshold}")
    return max(belief_threshold, base_threshold / max_score)  # Adjust base_threshold similarly

def calculate_belief_scores(ranking_scores, base_threshold, exact_match_found):
    belief_scores = defaultdict(list)
    logger.info(f"Ranking Scores")
    scores = [int(res.get('score', 0)) for res in ranking_scores]
    # logger.info(f"Ranking Score={ranking_scores}")
    if not scores:
        return None
    max_score = max(scores)
    if max_score == 0:
        print(f"all zeros")
        return None  # All scores are zero, indicating no suitable matches, return None

    # dynamic_threshold = calculate_dynamic_threshold(scores, base_threshold, exact_match_found)
    # logger.info(f"dynamic threshold={dynamic_threshold}")
    # Aggregate scores for documents that appear more than once
    for res in ranking_scores:
        score = int(res.get('score', 0))
        answer = res['answer']
        belief_scores[answer].append(score)

    # Calculate average score for each document and determine belief score
    final_belief_scores = {}
    for answer, score_list in belief_scores.items():
        avg_score = sum(score_list) / len(score_list)
        normalized_score = avg_score / max(scores)  # Normalize the score
        final_belief_scores[answer] = normalized_score if normalized_score >= base_threshold else 0

    # logger.info(f"Belief Scores={final_belief_scores}")
    return final_belief_scores

import time

def get_llm_results(prompt, query, documents, max_retries=2, llm=None, llm_name='llama'):
    # print(f"get_llm_results for Query={query}")
    #divide documents into 2 chunks
    if len(documents) > 5:
        midpoint = len(documents) // 2
    else: 
        midpoint = len(documents)
    first_half = documents[:midpoint]
    second_half = documents[midpoint:]
    
    def process_half(doc_half, half_name):
        attempt = 0
        while attempt <= max_retries:
            logger.info(f"Attempt {attempt} to invoke {llm_name} ")
            try:
                chain = prompt | llm
                # start_times = time.time()n
                
                # config={'callbacks': [ConsoleCallbackHandler()]}) for verbose
                results =  chain.invoke({"query": query, "documents": documents})
                results = results.content
                # print(f"Time taken for llm chain: {time.time() - start_times}")
                if isinstance(results, list) and all(isinstance(item, dict) for item in results):
                    # logger.info(f"Initial Results={results}")
                    return results

                # Attempt to parse results as JSON if it's a string
                if isinstance(results, str):
                    
                    fixed_results = fix_json_quotes(results)
                    if isinstance(fixed_results, list) and all(isinstance(item, dict) for item in fixed_results):
                        return fixed_results
                    # else:
                    #     logger.info(f"Invalid JSON response: {fixed_results}")
                    #     attempt += 1
                    #     continue
                
                # Use fixing_parser to parse results if not a list of dictionaries
                if not (isinstance(results, list) and all(isinstance(item, dict) for item in results)):
                    try:
                        results = fixing_parser.parse(results)
                        # Verify the results after fixing_parser parsing
                        if isinstance(results, list) and all(isinstance(item, dict) for item in results):
                            # logger.info(f"Fixed Results with fixing_parser: {results}")
                            return results
                    except Exception as e:  # Broad exception handling for any error from fixing_parser
                        logger.info(f"fixing_parser parsing error: {e}")
                        time.sleep(0.00005)
                        attempt += 1
                        continue  # Retry if fixing_parser parsing fails

                # logger.info(f"Results \n{results} are not in the expected format after attempts to parse, retrying...")
                attempt += 1

            except ValidationError as e:
                logger.info(f"Validation Error: {e}")
                attempt += 1
                continue  # Retry on validation errors
            
            except Exception as e:
                logger.info(f"LLM Unexpected Error: {e}")
                attempt += 1
                if attempt > max_retries:
                    logger.info("Max retries reached without a valid response, returning None")
                    return None
    results_first_half = process_half(first_half, "first_half")
    results_second_half = process_half(second_half, "second_half") if len(second_half) > 0 else None
    if results_first_half is None and results_second_half is None:
        logger.error("Failed to obtain valid results from both halves.")
        return None

    # Initialize combined results
    combined_results = []

    if results_first_half:
        combined_results.extend(results_first_half)
    if results_second_half:
        combined_results.extend(results_second_half)

    # logger.info(f"Combined Results: {combined_results}")
    return combined_results
                
def pass_to_chat_llm_chain(query, top_candidates, n_prompts =1, threshold=0.8, llm_name='llama', domain=None, prompt_stage:int=2):
    relationship_scores = {
        'synonym': 10,
        'exact match': 10,
        'highly relevant': 8,
        'partially relevant': 6,
        'not relevant': 0
    }
    # def calculate_final_score(doc,ranking_scores):
    #     try:
    #         normalized_label = create_document_string(doc)
    #         scores = [int(result['score']) for result in ranking_scores if normalize(result['answer']) == normalized_label]
    #         final_score = np.mean(scores) if scores else 0
    #         # logger.info(f"Final Score for {normalized_label}: {final_score}")
    #         return final_score
    #     except Exception as e:
    #         logger.info(f"Error in calculate_final_score: {e}")
    #         return 0
    try:
        try:
            model = LLMManager.get_instance(llm_name)
        except Exception as e:
            logger.info(f"Error loading LLM: {e}")
        # _, ranking_examples = load_mapping(MAPPING_FILE, None)
        seen = set()
        documents = []
        for doc in top_candidates:
            doc_str = create_document_string(doc)
            if doc_str not in seen:
                seen.add(doc_str)
                documents.append(doc_str)
        # print(f"documents={documents}")
        ranking_scores = []        
        link_predictions_results = []
        exact_match_found = False

        for _ in range(n_prompts):  # Assume n_prompts is 3
            # Ranking phase
            # score_prompt_start_time = time.time()
            ranking_prompt = generate_ranking_prompt(query=query, documents=documents,domain=domain,in_context=True)
            ranking_results =  get_llm_results(prompt=ranking_prompt, query=query, documents=documents, llm=model,llm_name=llm_name)
            
            # print(f"Time taken for ranking prompt: {time.time() - score_prompt_start_time}")
            if ranking_results:
                ranking_scores.extend(ranking_results)
                for result in ranking_results:
                    if isinstance(result, dict) and int(result.get('score', 0)) == 10:
                        exact_match_found =  True if result['answer'] in documents else False
                        # print(f"{ranking_results}")
                        logger.info(f"Exact match found in Ranking: {result['answer']} = {exact_match_found}. Does it exist in original documents={result['answer'] in documents}")
            score_prompt_start_time = time.time()
            link_predictions_results = []
            if prompt_stage == 2:
                link_prediction_prompt = generate_link_prediction_prompt(query, documents,domain=domain,in_context=True)
                lp_results =  get_llm_results(prompt=link_prediction_prompt, query=query, documents=documents, llm=model,llm_name=llm_name)
                # print(f"Time taken for link prediction prompt: {time.time() - score_prompt_start_time}")
                if lp_results:
                    # print(f"link prediction results={lp_results}")
                    for res in lp_results:
                        if isinstance(res, dict):
                            res['score'] = relationship_scores.get(res.get('relationship','').strip().lower(), 0)
                    link_predictions_results.extend(lp_results)
                    for res in lp_results:
                        if isinstance(res, dict) and res['relationship'] == 'exact match':
                            exact_match_found = True
                            # if res['answer'] not in documents:
                            logger.info(f"Exact match found in Link Prediction: {res['answer']} = {exact_match_found}. Does it exist in original documents={res['answer'] in documents}")
                    # print(f"{lp_results}")
        combined_scores = ranking_scores + link_predictions_results     
        if isinstance(combined_scores, str): print(f"combined_scores={combined_scores}") 
        avg_belief_scores = calculate_belief_scores(combined_scores, threshold, exact_match_found=exact_match_found)
        if avg_belief_scores is None:
            return [], False
        sorted_belief_scores = sorted(avg_belief_scores.items(), key=lambda item: item[1], reverse=True)
        sorted_belief_scores = dict(sorted_belief_scores)
        logger.info(f"belief_threshold={threshold}")
        for doc in top_candidates:
            doc_string = create_document_string(doc)
            doc.metadata['belief_score'] = sorted_belief_scores.get(doc_string, 0)
        filtered_candidates = [doc for doc in top_candidates if sorted_belief_scores.get(create_document_string(doc), 0) >= threshold]
        #sort the documents based on there score from high to low
        logger.info(f"filtered candidates")
        sorted_filtered_candidates = sorted(filtered_candidates, key=lambda doc: doc.metadata['belief_score'], reverse=True)
        print(f"filtered_candidates={[doc.metadata['label'] for doc in sorted_filtered_candidates]}")
        # print(f"Time taken for pass_to_chat_llm_chain: {end_time - start_time}")
        return sorted_filtered_candidates, exact_match_found

    except Exception as e:
        logger.info(f"Error in pass_to_chat_llm_chain: {e}")
        return [], False
    

def get_json_output(input_text:str):
        llm=LLMManager.get_instance("gpt3.5")
        prompt = PromptTemplate(
                template=f"""
                    Convert the given input into a valid JSON format. The input provided is:
                    {input_text}
                    You should return a list of dictionaries where each dictionary includes 'answer' and 'score' keys.
                    Json Output:
                    """,
                input_variables=["input_text"]
            )        
        chain = prompt | llm  | JsonOutputParser()
        results = chain.invoke({"input_text":input_text})
        # logger.info(f"json results={results}")
        return results
    
    
    
# return filtered_documents      

# belief_scores = {}
# for result in ranking_scores:
#     # logger.info(f"type of result={type(result)}")
#     if (isinstance(result, dict)):
#         answer = normlize_content(result['answer'])
#         score = int(result['score'])
        
#         binary_score = 1 if score >= threshold else 0
#         belief_scores.setdefault(answer, []).append(binary_score)
#     else:
#         logger.info(f"Invalid result: {result}")

# avg_belief_scores = {answer: np.mean(scores) if scores else 0 for answer, scores in belief_scores.items()}
# logger.info(f"Average Belief Scores: {avg_belief_scores}")
# non_zero_scores = [score for score in avg_belief_scores.values() if score > 0]
# belief_threshold = np.percentile(non_zero_scores, 80) if non_zero_scores else 0
# # belief_threshold = np.percentile([score for score in avg_belief_scores.values() if score], 80) if avg_belief_scores else 0
# logger.info(f"Belief Threshold: {belief_threshold}")
# filtered_documents = [doc for doc in top_candidates if avg_belief_scores.get(normlize_content(doc.metadata['label']), 0) >= belief_threshold]
# # if belief_threshold >= 0.6:
# # logger.info(f"All scores={all_scores}")
# match_found = (exact_match_found and (belief_threshold >= 0.9)) or belief_threshold >= 1.0
# sorted_documents = sorted(filtered_documents, key=calculate_final_score, reverse=True)
# return sorted_documents, match_found

# class AnswerScorePair(BaseModel):
#     answer: str  = ""
#     score: int = 0
#     explanation :str = ""
#     @validator("score")
#     def score_must_be_positive(cls, field):
#         if field < 0:
#             raise ValueError("Score must be a positive number")
#         return field

# class ListOfAnswerScorePairs(BaseModel):
#     pairs: List[AnswerScorePair]
# class TermDomain(BaseModel):
#     domain: str = Field(description="The domain of the medical term")
#     revised_term: str = Field(description="The revised term")
# class MappedTerms(BaseModel):
#     domain: str = None
#     revised_term: str = None
#     values : List[str] =  None
#     dosage_unit : str = None
#     frequency: List[str] = None
#     context:str = None
#     meas_unit :str = None