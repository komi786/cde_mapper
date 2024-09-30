
# # def handle_query(query_text, llm_name, initial_domain, domain_retrievers, all_retrievers,db, search_type:str):
# #     print(f"initial_domain={initial_domain}")
# #     retriever = domain_retrievers.get(initial_domain, None)
# #     if retriever is None:
# #         print(f"define ensemble retriever for initial_domain={initial_domain}")
# #         domain_retrievers[initial_domain] = define_ensemble_retriever(get_vocabulary(initial_domain, query_text), all_retrievers, domain=initial_domain, search_type=search_type)
# #         retriever = domain_retrievers[initial_domain]
# #     retrieved_docs_1 = []
# #     match_found = False
# #     if not any(word in query_text for word in ['mother', 'father', 'parent']):
# #         retrieved_docs_1 = perform_retrieval(query_text, retriever)
# #         if len(retrieved_docs_1) > 0:
# #             processed_docs_ , match_found = process_retrieved_docs(query=query_text, docs=retrieved_docs_1, llm_name=llm_name,domain=initial_domain)
# #     if not match_found:
# #         print(f"no exact concepts found for query={query_text}")
# #         query_decomposed = pre_retieval_chain(query_text, llm_name)
# #         new_domain = query_decomposed.get('domain','unknown')
# #         if new_domain != initial_domain and new_domain not in domain_retrievers:
# #             print(f"new_domain={new_domain}")
# #             if new_domain not in domain_retrievers:
# #                 print(f"define ensemble retriever for new domain={new_domain}")
# #                 domain_retrievers[new_domain] = define_ensemble_retriever(get_vocabulary(new_domain, query_text), all_retrievers, domain=new_domain, search_type=search_type)
# #                 retriever = domain_retrievers[new_domain]
# #         revised_term = query_decomposed.get('revised_term', query_text)
# #         retrieved_docs_2 = perform_retrieval(revised_term, retriever)        
# #         processed_docs_2,_ = process_retrieved_docs(revised_term, retrieved_docs_2, llm_name,domain=new_domain)
# #         context = query_decomposed.get('context',None) 
# #         status = query_decomposed.get('status', None)
# #         unit = query_decomposed['unit'] if 'unit' in query_decomposed else None
# #         if context:
# #             if 'unknown' not in domain_retrievers:
# #                 print(f"define ensemble retriever for context domain")
# #                 domain_retrievers['context'] = define_ensemble_retriever(get_vocabulary('unknown',query=context), all_retrievers, domain='context', search_type=search_type)
# #         context_docs = process_context(context, domain_retrievers['unknown'], llm_name,domain=new_domain) if context else []
# #         values_docs = process_values(status, domain_retrievers['unknown'], llm_name,domain=new_domain) if status else []
# #         if unit:
# #             if 'unit' not in domain_retrievers:
# #                 print(f"define ensemble retriever for unit domain")
# #                 domain_retrievers['unit'] = define_ensemble_retriever(get_vocabulary('unit',query=unit), all_retrievers, domain='unit', search_type=search_type)            
# #         unit_docs = process_unit(unit, domain_retrievers['unit'],llm=llm_name,domain='unit') if unit else []
# #         values_labels = ', '.join(doc['standard_label'] for doc in values_docs[:len(status)]) if values_docs else ''
# #         value_codes = ','.join(doc['standard_code'] for doc in values_docs[:len(status)]) if values_docs else ''
# #         labels_parts = [
# #         ','.join(doc['standard_label'] for doc in processed_docs_2[:2]) if processed_docs_2 else '',
# #         context_docs[0]['standard_label'] if context_docs else ''
# #         ]
# #         mapping_codes = [
# #             ','.join(doc['standard_code'] for doc in processed_docs_2[:2]) if processed_docs_2 else '',
# #             context_docs[0]['standard_code']if context_docs else ''
            
# #         ]
# #         # Filter out empty strings and join the rest with commas
# #         combine_docs_label = ','.join(filter(None, labels_parts))
# #         combine_mapping_codes = ','.join(filter(None, mapping_codes))
# #         processed_result = {
# #             'query': query_text,
# #             'revised_query': f"{revised_term},{context},{unit},{status}",
# #             'domain': new_domain,
# #             'standard_label': combine_docs_label,
# #             'standard_code': combine_mapping_codes,
# #             'values': values_labels,
# #             'value_codes': value_codes,
# #             'unit': unit_docs[0]['standard_label'] if unit_docs else '',
# #             'unit_code': unit_docs[0]['standard_code'] if unit_docs else ''
            
# #         }
# #         db.save_domain_to_csv(query_text, processed_result['domain'], processed_result['standard_code'],
# #                                 processed_result['standard_label'],processed_result['values'],processed_result['value_codes'],
# #                                 processed_result['unit'],processed_result['unit_code'])
# #         return processed_result

# #     else:
# #         combine_docs_label = ','.join(doc['standard_label'] for doc in processed_docs_[:3]) if processed_docs_ else ''
# #         combine_doc_codes = ','.join(doc['standard_code'] for doc in processed_docs_[:3]) if processed_docs_ else ''
# #         unit_code, unit_label = '', ''
# #         if initial_domain == 'unit':
# #             unit_code = processed_docs_[0]['standard_code'] if processed_docs_ else ''
# #             unit_label = processed_docs_[0]['standard_label'] if processed_docs_ else ''
# #         processed_result = {
# #             'query': query_text,
# #             'revised_query': query_text,
# #             'domain': initial_domain,
# #             'standard_label': combine_docs_label,
# #             "standard_code":combine_doc_codes,
# #             'values': '',
# #             'value_codes': '',
# #             'unit': unit_label,
# #             'unit_code': unit_code
# #         }
# #     db.save_domain_to_csv(query_text, processed_result['domain'], processed_result['standard_code'],processed_result['standard_label'],
# #                             processed_result['values'],processed_result['value_codes'],processed_result['unit'],
# #                             processed_result['unit_code'])
# #     return processed_result

# #---------Handle Query V1----------------
# # def handle_query_v1(query_text, llm_name, initial_domain, all_retrievers,db, search_type:str):
# #     print(f"initial_domain={initial_domain}")
# #     retriever = define_ensemble_retriever(select_vocabulary(query_text,domain=initial_domain), all_retrievers, domain=initial_domain, search_type=search_type)
# #     retrieved_docs_1 = []
# #     match_found = False
# #     if not any(word in query_text for word in ['mother', 'father', 'parent']):
# #         retrieved_docs_1 = perform_retrieval(query_text, retriever)
# #         if len(retrieved_docs_1) > 0:
# #             processed_docs_ , match_found = process_retrieved_docs(query=query_text, docs=retrieved_docs_1, llm_name=llm_name,domain=initial_domain)
# #     if not match_found:
# #         print(f"no exact concepts found for query={query_text}")
# #         query_decomposed = pre_retieval_chain(query_text, llm_name)
# #         new_domain = query_decomposed.get('domain','unknown')
# #         if new_domain != initial_domain:
# #             print(f"new_domain={new_domain}")
# #             retriever = define_ensemble_retriever(select_vocabulary(query_text,domain=new_domain), all_retrievers, domain=new_domain, search_type=search_type)
# #         revised_term = query_decomposed.get('revised_term', query_text)
# #         retrieved_docs_2 = perform_retrieval(revised_term, retriever)        
# #         processed_docs_2,_ = process_retrieved_docs(revised_term, retrieved_docs_2, llm_name,domain=new_domain)
# #         context = query_decomposed.get('additional_information',None) 
# #         status = query_decomposed.get('status', None)
# #         unit = query_decomposed['unit'] if 'unit' in query_decomposed else None
# #         values_docs = {}
# #         status_docs = []
# #         if context:
# #             values_docs = {}
# #             if isinstance(context, list):
# #                 retriever = define_ensemble_retriever(select_vocabulary(context[0],domain='unknown'), all_retrievers, domain='unknown', search_type=search_type)
# #                 values_docs = process_values(context, retriever, llm_name,domain='unknown') if context else []
# #             else:
# #                 retriever = define_ensemble_retriever(select_vocabulary(context,domain='unknown'), all_retrievers, domain='unknown', search_type=search_type)
# #                 values_docs[context] = process_context(context, retriever, llm_name,domain='unknown') if context else []
# #         unit_docs = None
# #         if unit:
# #             retriever = define_ensemble_retriever(select_vocabulary(unit,domain='unit'), all_retrievers, domain='unit', search_type=search_type)            
# #         unit_docs = process_unit(unit, retriever,llm=llm_name,domain='unit') if unit else []
# #         if status:
# #             retriever = define_ensemble_retriever(select_vocabulary(status[0],domain='unknown'), all_retrievers, domain='unknown', search_type=search_type)
# #             status_docs= process_values(status, retriever, llm_name,domain='unknown') if status else []
# #         if isinstance(context, list):
# #             values_labels = ','.join(','.join(doc['standard_label'] for doc in values_docs[ctxt]) for ctxt in context if ctxt in values_docs and values_docs[ctxt]) if values_docs else ''
# #             value_codes = ','.join(','.join(doc['standard_code'] for doc in values_docs[ctxt]) for ctxt in context if ctxt in values_docs and values_docs[ctxt]) if values_docs else ''
# #         else:
# #             values_labels = ','.join(doc['standard_label'] for doc in values_docs[context]) if context in values_docs and values_docs[context] else ''
# #             value_codes = ','.join(doc['standard_code'] for doc in values_docs[context]) if context in values_docs and values_docs[context] else ''
                    
# #         labels_parts = [
# #         ','.join(doc['standard_label'] for doc in processed_docs_2[:2]) if processed_docs_2 else '',
# #           values_labels if values_labels else ''
# #         ]
# #         mapping_codes = [
# #             ','.join(doc['standard_code'] for doc in processed_docs_2[:2]) if processed_docs_2 else '',
# #             value_codes if value_codes else ''
# #         ]
# #         status_label = values_labels = ','.join(','.join(doc['standard_label'] for doc in status_docs[value]) for value in context if value in status_docs and status_docs[value]) if status_docs else ''
# #         status_codes = ','.join(','.join(doc['standard_code'] for doc in status_docs[value]) for value in context if value in status_docs and status_docs[value]) if status_docs else ''
# #         # Filter out empty strings and join the rest with commas
# #         combine_docs_label = ','.join(filter(None, labels_parts))
# #         combine_mapping_codes = ','.join(filter(None, mapping_codes))
# #         processed_result = {
# #             'query': query_text,
# #             'revised_query': f"{revised_term},{context}",
# #             'domain': new_domain,
# #             'standard_label': combine_docs_label,
# #             'standard_code': combine_mapping_codes,
# #             'additional_context': values_labels,
# #             'additional_context_codes': value_codes,
# #             'categorical_values': status_label,
# #             'categorical_codes': status_codes,
# #             'unit': unit_docs[0]['standard_label'] if unit_docs else '',
# #             'unit_code': unit_docs[0]['standard_code'] if unit_docs else ''

            
# #         }
# #         # # db.save_domain_to_csv(query_text, processed_result['domain'], processed_result['standard_code'],
# #         #                         processed_result['standard_label'],processed_result['values'],processed_result['value_codes'],
# #         #                         processed_result['unit'],processed_result['unit_code'])
# #         # return processed_result

# #     else:
# #         combine_docs_label = ','.join(doc['standard_label'] for doc in processed_docs_[:3]) if processed_docs_ else ''
# #         combine_doc_codes = ','.join(doc['standard_code'] for doc in processed_docs_[:3]) if processed_docs_ else ''
# #         unit_code, unit_label = '', ''
# #         if initial_domain == 'unit':
# #             unit_code = processed_docs_[0]['standard_code'] if processed_docs_ else ''
# #             unit_label = processed_docs_[0]['standard_label'] if processed_docs_ else ''
# #         processed_result = {
# #             'query': query_text,
# #             'revised_query': query_text,
# #             'domain': initial_domain,
# #             'standard_label': combine_docs_label,
# #             "standard_code":combine_doc_codes,
# #             'additional_context': '',
# #             'additional_context_codes': '',
# #             'categorical_values': '',
# #             'categorical_codes': '',
# #             'unit': unit_label,
# #             'unit_code': unit_code
# #         }
# #     # # db.save_domain_to_csv(query_text, processed_result['domain'], processed_result['standard_code'],processed_result['standard_label'],
# #     #                         processed_result['values'],processed_result['value_codes'],processed_result['unit'],
# #     #                         processed_result['unit_code'])
# #     return processed_result
# #-------------------Handle Query V2-----------
    
# def find_domain(query,model_name='llama', llm= None):
#     try:
#         if llm is None:
#             active_model = LLMManager.get_instance(model=model_name)
#         else:
#             active_model = llm
#         prompt2 = PromptTemplate(
#         template = """
#                 Role: You are a helpful assistant with expertise in data science and the biomedical domain.\n
#                 Task: Classify the provided medical source term into the most appropriate category only if you are 100% confident based on the detailed descriptions provided. If there is any ambiguity or potential for multiple interpretations, use 'Unknown'.\n
#                 Category Descriptions and their respective Examples:\n
#                 - Visit: Refers to a visit or encounter with a healthcare provider. Examples: 'follow-up month 1', 'date of baseline visit', 'visit to hospital', 'hospitalization due to heart attack', 'Hypertension-related hospitalization'.
#                 - Family History: Related to familial health histories. Examples: 'family history of diabetes', 'history of heart failure (mother)', 'family history of hypertension pertaining to the parents'.
#                 - Observation: Includes observational clinical data and qualifier values not classified under any other category. Examples: 'history of surgical procedures', 'Needs help with cooking', 'medication prescribed', 'History of event within 5 years', 'No', 'Possible'.
#                 - Measurement: Includes measured values, laboratory tests, vital signs, assessment instruments, key components of lipid panel, complete metabolic panel, complete blood counts, microbiologic cultures, liver function tests, urinalysis, and viral panels. Examples: 'Ejection fraction measured at centre of inclusion', 'c-reactive protein (crp)', 'hospital anxiety and depression scale (hads)', 'Aldosterone [Mass/volume] in Blood', 'blood pressure', 'current smoker'.
#                 - Drug: Captures records about drug substances and medication-related details, including dosage units, frequency of administration, and the process of dispensing. Examples: 'Sulfonylurea', 'aspirin 100 mg', 'metformin twice a day', 'alpha-2 adrenergic antagonist', 'dapagliflozin'.
#                 - Procedure: Captures data related to medical procedures or surgeries performed. Examples: 'angioplasty', 'CT scan', 'MRI scan', 'appendectomy'.
#                 - Device: Instruments, implants, reagents, or similar items used to diagnose, prevent, or treat disease. Examples: 'pacemaker', 'stethoscope', 'defibrillator'.
#                 - Condition: Indications of diseases, symptoms, or disorders. Examples: 'secondary fibromyalgia', 'Hiccoughs', 'headache', 'Right atrial area abnormality', 'anxiety'.
#                 - Death: Details pertaining to mortality, including causes and circumstances surrounding a patient's death. Examples: 'death due to heart failure', 'non-cardiovascular death', 'mortality rate'.
#                 - Specimen: Refers to specimens or samples used for testing. Examples: 'blood sample', 'biopsy tissue', 'urine sample'.
#                 - Demographics: Related to patient demographics, including race, gender, and age. Examples: 'Age at baseline', 'ethnicity', 'Asian race', 'female', 'Black or African American', 'South Asian'.
#                 - Unit: Includes units of measurement or drug dosage units. Examples: 'millisecond', 'trillion per liter', 'mm[Hg]', 'mg/kg', 'day per year'.
#                 - Unknown: Use if there is any doubt about the fit with other categories or multiple categories might apply.\n
#             Do not respond with more than one word.

#             Source Term: "{source_term}"

#             Domain:
#                 """,
#                 input_variables=["source_term"],
#                 validate_template=True
#                 )
        
#         refine_term =  query
#         chain = prompt2 | active_model
#         result = chain.invoke({"source_term": refine_term})
#         domain = normalize(result.content)
#         if len(domain.split()) > 1 and 'family history' not in domain:
#             domain = 'unknown'    
#         # Check if domain is None or more than one word
#         return domain
#     except Exception as e:
#         print(f"Error in find_domain: {e}")
#         domain = 'unknown'
#         return domain
        

# def pre_retieval_chain(query,llm_name = 'llama'):
#         try: 
#             active_model = LLMManager.get_instance(model=llm_name)
#             print(f"Pre-Retrievel Query={query}")
#             final_results = {}
#             domain = normalize(find_domain(query,llm=active_model,model_name=llm_name))
#             final_results['domain'] = domain
#             final_results['revised_term'] = query
#             mapping_for_domain, _,_ = load_mapping(MAPPING_FILE, domain)
#             mapping_result=pass_mapping_guideline_to_llm(query, mapping_for_domain,llm=active_model,llm_name=llm_name)
#             try:
#                 if mapping_result:
#                     # mapping_keys = mapping_for_domain.keys()
#                     if 'domain' in mapping_result and final_results['domain'] == 'unknown':
#                         final_results['domain'] = mapping_result['domain']
#                     if 'status' in mapping_result:
#                         final_results['status']  = mapping_result['status']
#                     if 'additional_entities' in mapping_result:
#                         final_results['additional_entities'] = mapping_result['additional_entities']
#                     if 'main_entity' in mapping_result:
#                         final_results['base_entity']  = mapping_result['main_entity']
#                     if 'unit' in mapping_result:
#                         final_results['unit']  = mapping_result['unit']
#                 print(f"decomposed Query={final_results}")
#                 return  final_results
#             except KeyError as e:
#                 print(f"KeyError:Incorrect key {e} in mapping_for_domain")
#         except Exception as e:
#             print(f"Error in pre-retrieval chain: {e}")
#             return {}

# # def handle_query_v2(query_text, llm_name, original_retrievers, initial_domain, retrievers_list,db,search_type, topk=5):
# #     print(f"initial_domain={initial_domain}")
# #     query_decomposed = pre_retieval_chain(query_text, llm_name)
# #     query_retriever = retrievers_list[0]
# #     new_domain = query_decomposed.get('domain','unknown')
# #     if new_domain != initial_domain:
# #         initial_domain = new_domain
# #         query_retriever = define_ensemble_retriever(select_vocabulary(query_text,domain=initial_domain), original_retrievers, domain=initial_domain, search_type=search_type,topk=topk)
# #     revised_term = query_decomposed.get('revised_term', query_text)
# #     retrieved_docs_2 = perform_retrieval(revised_term, query_retriever)        
# #     processed_docs_2,_ = process_retrieved_docs(revised_term, retrieved_docs_2, llm_name,domain=initial_domain)
# #     context = query_decomposed.get('additional_information',None) 
# #     status = query_decomposed.get('status', None)
# #     unit = query_decomposed['unit'] if 'unit' in query_decomposed else None
# #     values_docs = {}
# #     status_docs = []
# #     if context:
# #         values_docs = {}
# #         if isinstance(context, list):
# #             # retriever = define_ensemble_retriever(select_vocabulary(context[0],domain='unknown'), all_retrievers, domain='unknown', search_type=search_type)
# #             values_docs = process_values(context, query_retriever, llm_name,domain=initial_domain) if context else []
# #         else:
# #             # retriever = define_ensemble_retriever(select_vocabulary(context,domain='unknown'), all_retrievers, domain='unknown', search_type=search_type)
# #             values_docs[context] = process_context(context, query_retriever, llm_name,domain=initial_domain) if context else []
# #     unit_docs = None
# #     if unit:
# #         unit_retriever = retrievers_list[1]          
# #         unit_docs = process_unit(unit, unit_retriever,llm=llm_name,domain='unit') if unit else []
# #     if status:
# #         status_retriever = retrievers_list[2]   
# #         status_docs= process_values(status, status_retriever, llm_name,domain='unknown') if status else []
# #     if isinstance(context, list):
# #         values_labels = ','.join(','.join(doc['standard_label'] for doc in values_docs[ctxt]) for ctxt in context if ctxt in values_docs and values_docs[ctxt]) if values_docs else ''
# #         value_codes = ','.join(','.join(doc['standard_code'] for doc in values_docs[ctxt]) for ctxt in context if ctxt in values_docs and values_docs[ctxt]) if values_docs else ''
# #     else:
# #         values_labels = ','.join(doc['standard_label'] for doc in values_docs[context]) if context in values_docs and values_docs[context] else ''
# #         value_codes = ','.join(doc['standard_code'] for doc in values_docs[context]) if context in values_docs and values_docs[context] else ''
                
# #     labels_parts = [
# #     ','.join(doc['standard_label'] for doc in processed_docs_2[:2]) if processed_docs_2 else '',
# #         values_labels if values_labels else ''
# #     ]
# #     mapping_codes = [
# #         ','.join(doc['standard_code'] for doc in processed_docs_2[:2]) if processed_docs_2 else '',
# #         value_codes if value_codes else ''
# #     ]
# #     status_label = values_labels = ','.join(','.join(doc['standard_label'] for doc in status_docs[value]) for value in context if value in status_docs and status_docs[value]) if status_docs else ''
# #     status_codes = ','.join(','.join(doc['standard_code'] for doc in status_docs[value]) for value in context if value in status_docs and status_docs[value]) if status_docs else ''
# #     # Filter out empty strings and join the rest with commas
# #     combine_docs_label = ','.join(filter(None, labels_parts))
# #     combine_mapping_codes = ','.join(filter(None, mapping_codes))
# #     processed_result = {
# #         'query': query_text,
# #         'revised_query': f"{revised_term},{context}",
# #         'domain': new_domain,
# #         'standard_label': combine_docs_label,
# #         'standard_code': combine_mapping_codes,
# #         'additional_context': values_labels,
# #         'additional_context_codes': value_codes,
# #         'categorical_values': status_label,
# #         'categorical_codes': status_codes,
# #         'unit': unit_docs[0]['standard_label'] if unit_docs else '',
# #         'unit_code': unit_docs[0]['standard_code'] if unit_docs else ''

        
# #     }
# #     # # db.save_domain_to_csv(query_text, processed_result['domain'], processed_result['standard_code'],
# #     #                         processed_result['standard_label'],processed_result['values'],processed_result['value_codes'],
# #     #                         processed_result['unit'],processed_result['unit_code'])
# #     return processed_result
       