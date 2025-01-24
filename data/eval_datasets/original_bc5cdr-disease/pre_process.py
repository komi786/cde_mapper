import json
from rag.utils import load_docs_from_jsonl, save_docs_to_jsonl
from rag.data_loader import load_data
from langchain.schema import Document
from multiprocessing import Pool, cpu_count
import glob
from tqdm import tqdm
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
def combine_concept_files(data_dir, output_file):
    """       
    Parameters
    ----------
    data_dir : str
        a path of data

    Returns
    -------
    data : np.array 
        cui and sentence pairs
    """
    data = []
    file_types = ("*.concept", "*.txt")
    concept_files = []
    
    # Collect all concept and text files
    for ft in file_types:
        concept_files.extend(glob.glob(os.path.join(data_dir, ft)))

    for concept_file in tqdm(concept_files):
        with open(concept_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in tqdm(lines):
                line = line.split('||')
                
                # Skip lines that do not have exactly 5 elements
                if len(line) != 5:
                    print(f"Skipping line due to incorrect format: {line}")
                    continue
                
                cui = line[4].strip()
                sentence = line[3].strip()
                
                # Only add if sentence and CUI are valid
                if sentence and cui:
                    data.append((cui, sentence))
    
    # Remove duplicates
    data = list(dict.fromkeys(data))
    print("Query size:", len(data))
    print("Sample data:", data[:1])
    
    # Save the combined data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for cui, sentence in data:
            f.write(f"{cui}||{sentence}\n")
    
    print(f"Saved to {output_file}")
#### These values are incorrect or missing in dictionary with respect to queries (combined.txt) file. after creating docs, we manually added them by checking ATHENA website

#  ['loss of consciousness', 'pain', 'heart failure', 'bleeding', 'priapism', 'toxoplasmosis', 'downbeat nystagmus', 'chronic heart failure', 'irregular heart beat', 'anxiety', 'anemia', 'leukemia', 'patent ductus arteriosus', 'vomitus', 'glomerulosclerosis', 'atrial fibrillation', 'neuroleptic malignant syndrome', 'peripheral sensory neuropathy', 'hypokalemia', 'malignant hyperthermia', 'hyperventilation', 'abnormal ocular motility', 'sweating', 'multiple myeloma', 'lymphadenopathy', 'hemolytic anemia', 'visual field defect', 'leprosy', 'depressed', 'cerebral ischemia', 'middle cerebral artery occlusion', 'breast cancer', 'precordial pain', 'right bundle branch block', 'hemorrhagic bronchopneumonia', 'hepatitis b', 'inattention', 'atrioventricular block', 'chill', 'malaria', 'chronic liver disease', 'aphasia', 'essential hypertension', 'leukocytosis', 'lung mass', 'delirium', 'antineutrophil cytoplasmic antibody positive vasculitis', 'cerebrovascular disease', 'acute kidney injury', 'hypothermia', 'hypertrophied', 'cardiogenic shock', 'macroprolactinemia', 'axonal neuropathy', 'emergency department', 'trauma', 'depressive disorder', 'antisocial personality disorder', 'thrombotic microangiopathy', 'al', 'postoperative delirium', 'optic disc edema', 'toxic optic neuropathy', 'fibrillation', 'status migrainosus', 'hodgkin lymphoma', 'amnesia', 'agranulocytosis', 'substance abuse disorder', 'delusions', 'hemiplegic migraine', 'maculopapular eruption', 'diastolic dysfunction', 'thrombolysis', 'pulmonary embolism', 'growth retardation', 'raised intraocular pressure', 'lateral epicondylitis', 'blepharoconjunctivitis', 'dry eyes', 'microphthalmos', 'optic nerve hypoplasia', 'mi', 'ataxia', 'papillary necrosis', 'hypoglycemia', 'atrial thrombosis', 'myocardial degeneration', 'acute cholecystitis', 'hydroureteronephrosis', 'sensory neuropathy', 'cirrhotic', 'alkalosis', 'ventricular tachyarrhythmia', 'dehydrated', 'depressive episode']
# #manullay added following :{"id": null, "metadata": {"label": "zygomycosis", "sid": "432830", "synonyms": "", "domain": "condition", "parent_term": "mycosis", "concept_class": "disorder", "vocab": "snomed", "scode": "59277005", "is_standard": "S"}, "page_content": "zygomycosis", "type": "Document"}
# {"id": null, "metadata": {"label": "downbeat nystagmus", "sid": "4146640", "synonyms": "Downbeat central nystagmus;;Downbeat nystagmus (disorder)", "domain": "condition", "parent_term": "mycosis", "concept_class": "disorder", "vocab": "snomed", "scode": "307676006", "is_standard": "S"}, "page_content": "downbeat nystagmus", "type": "Document"}
# {"id": null, "metadata": {"label": "Chronic heart failure", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Cardiac failure chronic", "scode": "48447003", "sid": "444031", "synonyms": "Chronic heart failure (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "chronic heart failure", "type": "Document"}
# {"id": null, "metadata": {"label": "irregular heart beat", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Abnormal heart beat", "scode": "361137007", "sid": "4263255", "synonyms": "irregular heart rate (finding);;heartbeats irregular;;irregular heart;;irregular heart rate;;heartbeat irregular;;heart irregular;;beat heart irregular;;irregular heartbeats;;heart beat irregular;;irregular heart beats", "has_answers": "", "is_standard": "S"}, "page_content": "irregular heart beat", "type": "Document"}
# {"id": null, "metadata": {"label": "Vomitus", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Gastric contents", "scode": "1985008", "sid": "4057954", "synonyms": "Vomit;;Vomitus (substance)", "has_answers": "", "is_standard": "S"}, "page_content": "Vomitus", "type": "Document"}
# {"id": null, "metadata": {"label": "peripheral sensory neuropathy", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Peripheral nerve disease", "scode": "789588003", "sid": "37312197", "synonyms": "Peripheral sensory neuropathy (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "peripheral sensory neuropathy", "type": "Document"}
# {"id": null, "metadata": {"label": "Abnormal ocular motility", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Eye movement - finding", "scode": "103252009", "sid": "4011323", "synonyms": "Abnormal eye movement;;Abnormal ocular motility (finding)", "has_answers": "", "is_standard": "S"}, "page_content": "Abnormal ocular motility", "type": "Document"}
# {"id": null, "metadata": {"label": "Sweating", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Finding of sweating", "scode": "415690000", "sid": "4188566", "synonyms": "Sweating (finding);;Sweats", "has_answers": "", "is_standard": "S"}, "page_content": "sweating", "type": "Document"}
# {"id": null, "metadata": {"label": "visual field defect", "domain": "condition", "concept_class": "clinical finding", "vocab": "snomed", "parent_term": "Finding of sweating", "scode": "12184005", "sid": "377286", "synonyms": "VFD - Visual field defect;;Visual field defect (finding)", "has_answers": "", "is_standard": "S"}, "page_content": "visual field defect", "type": "Document"}
# {"id": null, "metadata": {"label": "precordial pain", "domain": "condition", "concept_class": "clinical finding", "vocab": "snomed", "parent_term": "Anterior chest wall pain;;Precordial pain", "scode": "71884009", "sid": "134159", "synonyms": "Precordial pain (finding)", "has_answers": "", "is_standard": "S"}, "page_content": "precordial pain", "type": "Document"}
# {"id": null, "metadata": {"label": "Hemorrhagic bronchopneumonia", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Bronchial hemorrhage;;Bronchopneumonia;;Hemorrhagic pneumonia", "scode": "181007", "sid": "4111119", "synonyms": "Haemorrhagic bronchopneumonia;;Hemorrhagic bronchopneumonia (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "Hemorrhagic bronchopneumonia", "type": "Document"}
# {"id": null, "metadata": {"label": "inattention", "domain": "observation", "concept_class": "clinical finding", "vocab": "snomed", "parent_term": "General inattentiveness;;Bronchopneumonia;;Inattention (finding)", "scode": "22058002", "sid": "4318665", "synonyms": "General inattentiveness;;Inattention (finding)", "has_answers": "", "is_standard": "S"}, "page_content": "Inattention", "type": "Document"}
# {"id": null, "metadata": {"label": "chill", "domain": "observation", "concept_class": "clinical finding", "vocab": "snomed", "parent_term": "chills;;Shivering or rigors;;Shivering", "scode": "43724002", "sid": "434490", "synonyms": "Chill (finding);;Shivering", "has_answers": "", "is_standard": "S"}, "page_content": "chill", "type": "Document"}
# {"id": null, "metadata": {"label": "chronic liver disease", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Chronic digestive system disorder;;Chronic liver disease", "scode": "328383001", "sid": "4212540", "synonyms": "Chronic liver disease (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "chronic liver disease", "type": "Document"}
# {"id": null, "metadata": {"label": "essential hypertension", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Hypertensive disorder;;Primary hypertension;;Unspecified essential hypertension", "scode": "59621000", "sid": "320128", "synonyms": "Essential hypertension (disorder);;Idiopathic hypertension;;Primary hypertension;;Systemic primary arterial hypertension", "has_answers": "", "is_standard": "S"}, "page_content": "essential hypertension", "type": "Document"}
# {"id": null, "metadata": {"label": "lung mass", "domain": "condition", "concept_class": "clinical finding", "vocab": "snomed", "parent_term": "Lung finding;;Mass of respiratory structure", "scode": "309529002", "sid": "4203096", "synonyms": "Lung mass (finding)", "has_answers": "", "is_standard": "S"}, "page_content": "lung mass", "type": "Document"}
# {"id": null, "metadata": {"label": "Antineutrophil cytoplasmic antibody positive vasculitis", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Anti-neutrophil cytoplasmic antibody positive vasculitis", "scode": "722191003", "sid": "42535714", "synonyms": "Antineutrophil cytoplasmic antibody (ANCA) positive vasculitis;;HAntineutrophil cytoplasmic antibody positive vasculitis (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "Antineutrophil cytoplasmic antibody positive vasculitis", "type": "Document"}
# {"id": null, "metadata": {"label": "Cerebrovascular disease", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Cerebral vascular lesion;;Cerebrovascular disorder;;Disorder of cardiovascular system", "scode": "62914000", "sid": "381591", "synonyms": "Cerebrovascular disease (disorder);;CVD - Cerebrovascular disease", "has_answers": "", "is_standard": "S"}, "page_content": "Cerebrovascular disease", "type": "Document"}
# {"id": null, "metadata": {"label": "Hypertrophied", "sid": "36310407", "synonyms": "", "domain": "meas value", "parent_term": "", "concept_class": "answer", "vocab": "loinc", "scode": "LA19617-2", "is_standard": "S"}, "page_content": "Hypertrophied", "type": "Document"}
# {"id": null, "metadata": {"label": "macroprolactinemia", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Idiopathic hyperprolactinemia;;Macroprolactinaemia", "scode": "722943000", "sid": "36716768", "synonyms": "Macroprolactinaemia;;Macroprolactinemia (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "macroprolactinemia", "type": "Document"}
# {"id": null, "metadata": {"label": "axonal neuropathy", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Disorder of the central nervous system;;Axonal neuropathy", "scode": "60703000", "sid": "4246080", "synonyms": "Axonal neuropathy (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "axonal neuropathy", "type": "Document"}
# {"id": null, "metadata": {"label": "Emergency Department", "sid": "706470", "synonyms": "", "domain": "meas value", "parent_term": "", "concept_class": "Doc Setting", "vocab": "loinc", "scode": "LP173043-3", "is_standard": "S"}, "page_content": "Emergency Department", "type": "Document"}
# {"id": null, "metadata": {"label": "trauma", "sid": "706467", "synonyms": "", "domain": "meas value", "parent_term": "", "concept_class": "Doc Subject Matter", "vocab": "loinc", "scode": "LP183499-5", "is_standard": "S"}, "page_content": "trauma", "type": "Document"}
# {"id": null, "metadata": {"label": "al", "sid": "46237491", "synonyms": "", "domain": "meas value", "parent_term": "", "concept_class": "answer", "vocab": "loinc", "scode": "LA21991-7", "is_standard": "S"}, "page_content": "al", "type": "Document"}
# {"id": null, "metadata": {"label": "postoperative delirium", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Delirium;;Postoperative confusion", "scode": "771418002", "sid": "36674302", "synonyms": "Delirium following surgical procedure;;Delirium following surgical procedure (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "postoperative delirium", "type": "Document"}
# {"id": null, "metadata": {"label": "optic disc edema", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Optic disc disorder", "scode": "423341008", "sid": "4308632", "synonyms": "Edema of optic disc;;Edema of optic disc (disorder);;Optic disc oedema", "has_answers": "", "is_standard": "S"}, "page_content": "optic disc edema", "type": "Document"}
# {"id": null, "metadata": {"label": "fibrillation", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Cardiac arrhythmia", "scode": "40593004", "sid": "4226399", "synonyms": "Cardiac fibrillation;;Fibrillation (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "fibrillation", "type": "Document"}
# {"id": null, "metadata": {"label": "substance abuse disorder", "sid": "36310283", "synonyms": "", "domain": "meas value", "parent_term": "", "concept_class": "Answer", "vocab": "loinc", "scode": "LA28195-8", "is_standard": "S"}, "page_content": "substance abuse disorder", "type": "Document"}
# {"id": null, "metadata": {"label": "delusions", "domain": "condition", "concept_class": "clinical finding", "vocab": "snomed", "parent_term": "Delusion", "scode": "2073000", "sid": "444401", "synonyms": "Delusional ideas;;Delusional thoughts", "has_answers": "", "is_standard": "S"}, "page_content": "delusions", "type": "Document"}
# {"id": null, "metadata": {"label": "hemiplegic migraine", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Finding of head region", "scode": "59292006", "sid": "433763", "synonyms": "Hemiplegic migraine (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "hemiplegic migraine", "type": "Document"}
# {"id": null, "metadata": {"label": "maculopapular eruption", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Generalised maculopapular rash;;Eruption", "scode": "247471006", "sid": "4083795", "synonyms": "Maculopapular eruption (disorder);;Maculopapular exanthema;;Maculopapular rash", "has_answers": "", "is_standard": "S"}, "page_content": "maculopapular eruption", "type": "Document"}
# {"id": null, "metadata": {"label": "diastolic dysfunction", "domain": "condition", "concept_class": "clinical finding", "vocab": "snomed", "parent_term": "Abnormal cardiovascular function", "scode": "3545003", "sid": "141038", "synonyms": "Diastolic dysfunction (finding)", "has_answers": "", "is_standard": "S"}, "page_content": "diastolic dysfunction", "type": "Document"}
# {"id": null, "metadata": {"label": "thrombolysis", "sid": "4196955", "synonyms": "Thrombolysis, function;;Thrombolysis, function (observable entity)", "domain": "observation", "parent_term": "Hematologic function", "concept_class": "observable entity", "vocab": "snomed", "scode": "51308000", "is_standard": "S"}, "page_content": "thrombolysis", "type": "Document"}
# {"id": null, "metadata": {"label": "growth retardation", "sid": "4245750", "synonyms": "Decreased growth function;;Growth retardation (morphologic abnormality);;Growth suppression", "domain": "observation", "parent_term": "Growth alteration", "concept_class": "morph abnormality", "vocab": "snomed", "scode": "59576002", "is_standard": "S"}, "page_content": "growth retardation", "type": "Document"}
# {"id": null, "metadata": {"label": "raised intraocular pressure", "domain": "condition", "concept_class": "clinical finding", "vocab": "snomed", "parent_term": "Abnormal intraocular pressure", "scode": "112222000", "sid": "4011560", "synonyms": "Elevated intraocular pressure;;Raised intraocular pressure (finding)", "has_answers": "", "is_standard": "S"}, "page_content": "raised intraocular pressure", "type": "Document"}
# {"id": null, "metadata": {"label": "lateral epicondylitis", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Arthritis of elbow", "scode": "202855006", "sid": "81379", "synonyms": "Lateral epicondylitis (disorder);;Lateral epicondylitis of elbow;;Tennis elbow", "has_answers": "", "is_standard": "S"}, "page_content": "lateral epicondylitis", "type": "Document"}
# {"id": null, "metadata": {"label": "Blepharoconjunctivitis", "sid": "374347", "synonyms": "Blepharoconjunctivitis (disorder)", "domain": "condition", "parent_term": "Conjunctivitis", "concept_class": "disorder", "vocab": "snomed", "scode": "68659002", "is_standard": "S"}, "page_content": "Blepharoconjunctivitis", "type": "Document"}
# {"id": null, "metadata": {"label": "dry eyes", "domain": "condition", "concept_class": "clinical finding", "vocab": "snomed", "parent_term": "Dry eye;;Eye dryness;;Finding of moistness of eye", "scode": "162290004", "sid": "4036620", "synonyms": "Dry eyes (finding)", "has_answers": "", "is_standard": "S"}, "page_content": "dry eyes", "type": "Document"}
# {"id": null, "metadata": {"label": "optic nerve hypoplasia", "sid": "45881718", "synonyms": "", "domain": "meas value", "parent_term": "", "concept_class": "answer", "vocab": "loinc", "scode": "LA16338-8", "is_standard": "S"}, "page_content": "optic nerve hypoplasia", "type": "Document"}
# {"id": null, "metadata": {"label": "mi", "sid": "46237512", "synonyms": "", "domain": "meas value", "parent_term": "", "concept_class": "answer", "vocab": "loinc", "scode": "LA22011-3", "is_standard": "S"}, "page_content": "mi", "type": "Document"}
# {"id": null, "metadata": {"label": "papillary necrosis", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Disorder of renal parenchyma", "scode": "90241004", "sid": "444409", "synonyms": "Necrotising papillitis;;Necrotising renal papillitis;;Necrotizing papillitis;;Necrotizing renal papillitis;;Papillary necrosis (disorder);;Renal papillary necrosis", "has_answers": "", "is_standard": "S"}, "page_content": "papillary necrosis", "type": "Document"}
# {"id": null, "metadata": {"label": "atrial thrombosis", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Atrial cardiopathy;;Atrial thrombosis", "scode": "195147006", "sid": "4108352", "synonyms": "Thrombus of atrium;;Thrombus of atrium (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "atrial thrombosis", "type": "Document"}
# {"id": null, "metadata": {"label": "myocardial degeneration", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Cardiomyopathy;;Degenerative disorder", "scode": "64077000", "sid": "321320", "synonyms": "Mural degeneration of heart;;Degeneration of heart;;Muscular degeneration of heart;;Myocardial degeneration (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "myocardial degeneration", "type": "Document"}
# {"id": null, "metadata": {"label": "Hydroureteronephrosis", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Obstructive hydronephrosis;;Obstructive hydroureter;;Obstructive nephropathy", "scode": "40068008", "sid": "4220484", "synonyms": "Hydroureteronephrosis (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "Hydroureteronephrosis", "type": "Document"}
# {"id": null, "metadata": {"label": "Sensory neuropathy", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Neuropathy", "scode": "95662005", "sid": "4318869", "synonyms": "Sensory neuropathy (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "Sensory neuropathy", "type": "Document"}
# {"id": null, "metadata": {"label": "Sensory neuropathy", "domain": "meas value", "concept_class": "qualifier value", "vocab": "snomed", "parent_term": "Causal appearances", "scode": "255417007", "sid": "4114538", "synonyms": "cirrhotic (qualifier value)", "has_answers": "", "is_standard": "S"}, "page_content": "cirrhotic", "type": "Document"}
# {"id": null, "metadata": {"label": "ventricular tachyarrhythmia", "domain": "condition", "concept_class": "disorder", "vocab": "snomed", "parent_term": "Tachyarrhythmia", "scode": "6624005", "sid": "40622721", "synonyms": "Ventricular tachyarrhythmia (disorder)", "has_answers": "", "is_standard": "S"}, "page_content": "ventricular tachyarrhythmia", "type": "Document"}
# {"id": null, "metadata": {"label": "Dehydrated", "sid": "36308918", "synonyms": "", "domain": "meas value", "parent_term": "", "concept_class": "Answer", "vocab": "loinc", "scode": "LA27897-0", "is_standard": "S"}, "page_content": "Dehydrated", "type": "Document"}
# {"id": null, "metadata": {"label": "Depressive episode", "domain": "observation", "concept_class": "clinical finding", "vocab": "snomed", "parent_term": "Depressed mood", "scode": "871840004", "sid": "3656234", "synonyms": "Episode of depression;;Episode of depression (finding)", "has_answers": "", "is_standard": "S"}, "page_content": "Depressive episode", "type": "Document"}
def remove_redundant_rows(input_file, output_file):
    """
    Removes redundant rows from the input_file based on unique IDs (ID1|ID2).
    Writes the first occurrence of each unique ID to the output_file.
    
    Parameters:
    - input_file: Path to the input .txt file.
    - output_file: Path where the processed file will be saved.
    """
    seen_ids = set()  # To store unique ID combinations
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            
            if not line:
                # Skip empty lines
                continue
            
            # Split the line at '||' to separate IDs from the query
            try:
                ids_part, query = line.split('||', 1)
            except ValueError:
                print(f"Warning: Line {line_number} is malformed: '{line}'")
                continue  # Skip malformed lines
            
            # Normalize the IDs by stripping whitespace
            ids_part = ids_part.strip()
            
            if ids_part in seen_ids:
                # Duplicate ID found; skip this line
                continue
            else:
                # Unique ID; write to output and mark as seen
                outfile.write(line + '\n')
                seen_ids.add(ids_part)

    print(f"Processing complete. Unique entries saved to '{output_file}'.")

def process_doc(args):
    doc, label_dict = args
    label = doc.metadata.get('label', '').lower()
    if label in label_dict:
        source_doc = label_dict[label]
        # Update metadata
        doc.metadata['synonyms'] = source_doc.metadata.get('synonyms', [])
        doc.metadata['domain'] = source_doc.metadata.get('domain', '')
        doc.metadata['parent_term'] = source_doc.metadata.get('parent_term', '')
        doc.metadata['concept_class'] = source_doc.metadata.get('concept_class', '')
        doc.metadata['vocab'] = source_doc.metadata.get('vocab', '')
        doc.metadata['stid'] =  doc.metadata.get('sid', '')
        doc.metadata['scode'] = source_doc.metadata.get('scode', '')
        doc.metadata['sid'] = source_doc.metadata.get('sid', '')
        doc.metadata['has_answers'] = source_doc.metadata.get('has_answers', '')
        doc.metadata['is_standard'] = source_doc.metadata.get('is_standard', False)
        print(f"Added synonyms to {doc.metadata['sid']}")
    return doc

from collections import defaultdict

def open_dict_file(file):
    """
    Parses the input file and groups labels by their unique IDs.

    Parameters:
    - file: Path to the input .txt file.

    Returns:
    - A dictionary with IDs as keys and sets of labels as values.
    """
    data = defaultdict(set)  # Using set to avoid duplicate labels
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            parts = line.split('||')
            if len(parts) != 2:
                print(f"Warning: Line {line_number} is malformed: '{line}'")
                continue  # Skip malformed lines

            ids_part, query = parts
            if 'cui-less' in ids_part:
                continue  # Skip lines with 'cui-less' in IDs

            # Split CUIs and labels
            ids = ids_part.split('|') if '|' in ids_part else [ids_part]
            labels = query.split('|') if '|' in query else [query]
            
            # Normalize CUIs and labels
            normalized_ids = [cui.strip().lower() for cui in ids if cui.strip()]
            normalized_labels = [label.strip().lower() for label in labels if label.strip()]
            
            # Determine if multiple CUIs and only one label
            if len(normalized_ids) > 1 and len(normalized_labels) == 1:
                # Combine CUIs into a single identifier
                combined_cui = '|'.join(normalized_ids)
                data[combined_cui].add(normalized_labels[0])
            else:
                # Proceed with the original behavior
                for cui in normalized_ids:
                    for label in normalized_labels:
                        data[cui].add(label)
            

    return data

def save_docs_to_jsonl(docs, output_file):
    """
    Saves a list of Document objects to a JSON Lines (.jsonl) file.

    Parameters:
    - docs: List of Document objects.
    - output_file: Path to the output .jsonl file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in docs:
            json_record = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            f.write(json.dumps(json_record) + '\n')
    print(f"Documents saved to '{output_file}'.")
    
def add_information_to_doc(doc, matching_doc_metadata):
    doc.metadata.update({
        'synonyms': matching_doc_metadata.get('synonyms', ''),
        'domain': matching_doc_metadata.get('domain', ''),
        'parent_term': matching_doc_metadata.get('parent_term', ''),
        'concept_class': matching_doc_metadata.get('concept_class', ''),
        'vocab': matching_doc_metadata.get('vocab', ''),
        'scode': matching_doc_metadata.get('scode', ''),
        'is_standard': matching_doc_metadata.get('is_standard', '')
    })
    print(f"Added information to {doc.metadata['label']}")
    return doc

def preprocess_docs(docs):
    # Create a dictionary with label and synonyms as keys for fast lookup
    doc_dict = {}
    
    for doc_ in docs:
        label = doc_.metadata['label']
        label = label.strip().lower().replace('-', ' ').replace('_', ' ').replace(',', ' ')
        doc_dict[label] = doc_.metadata
        
        # synonyms = doc_.metadata['synonyms'].lower().split(";;")
        # for synonym in synonyms:
        #     doc_dict[synonym] = doc_.metadata
    
    return doc_dict

def add_synonyms_to_docs(docs, custom_docs, json_output_file):
    # Preprocess docs for faster lookup
    doc_dict = preprocess_docs(docs)
    
    with open(json_output_file, 'w') as jsonl_file:
        for custom_doc in tqdm(custom_docs):
            label = custom_doc.metadata['label']
            label = label.strip().lower().replace('-', ' ').replace('_', ' ').replace(',', ' ')
            # Lookup by label or synonyms in doc_dict
            if label in doc_dict:
                add_information_to_doc(custom_doc, doc_dict[label])
                print(f"Added synonyms to {custom_doc.metadata['label']}")
                # Write custom_doc to file
            jsonl_file.write(custom_doc.model_dump_json() + '\n')

             
def convert_dict_to_docs(file):
    """
    Converts grouped ID-label data into Document objects with synonyms.

    Parameters:
    - file: Path to the input .txt file.
    """
    grouped_data = open_dict_file(file)
    docs = []
    for idx, labels in grouped_data.items():
        if not labels:
            continue  # Skip if no labels are associated with the ID

        sorted_labels = sorted(labels)  # Sort for consistency
        labels_list = list(sorted_labels)
        for i, primary_label in enumerate(labels_list):
            # Synonyms are all other labels in the group except the current one
            # synonyms = [label for j, label in enumerate(labels_list) if j != i]
            # synonyms_str = ';;'.join(synonyms) if synonyms else ''

            doc = Document(
                page_content=primary_label,
                metadata={
                    'label': primary_label,
                    'sid': idx,
                    'synonyms': "",
                    'domain': 'condition',
                    'parent_term': '',
                    'concept_class': 'disorder',
                    'vocab': 'mesh',
                    'scode': '',
                    'is_standard': 'S'
                }
            )
            docs.append(doc)

    output_file = file.replace('.txt', '_docs.jsonl')
    save_docs_to_jsonl(docs, output_file)

def only_label_docs(docs):
    new_docs = [] 
    for doc in docs:
        new_docs.append(Document(doc.page_content, metadata={
            'label': doc.metadata['label'],
            'sid': doc.metadata['sid']
        }))
    return new_docs


def check_missing_ids(data_queries, custom_docs):
    found_ids = {}
    missing_ids = []
    for query in data_queries:
        cuis = str(query[0]).lower().split('|')
        name = query[1].strip().lower()
        found = False
        
        for doc in custom_docs:
            if 'sid' in doc.metadata:
                cui_gt = str(doc.metadata['sid']).lower().split('|')
                if any(cui in cuis for cui in cui_gt) or any(cui_gt in cuis for cui_gt in cuis):
                    # print(f"Found {name} in custom docs")
                    found_ids[name] = doc.metadata['sid']
                    found = True
                    break  # Break once you find the matching doc
            else:
                print(f"Missing sid in {doc.metadata['label']}")
        if not found:
            print(f"Missing {name} and cuis: {cuis}")
            missing_ids.append(name)  # Add to missing list if not found
    with open(f"{base_dir}/data/eval_datasets/original_bc5cdr-disease/test_queries.txt", 'w') as f:
        for key, value in found_ids.items():
            f.write(f"{value}||{key}\n")
    return found_ids, missing_ids

# combine_concept_files(f"{base_dir}/data/eval_datasets/original_bc5cdr-disease/processed_test", f"{base_dir}/data/eval_datasets/original_bc5cdr-disease/combined_test_queries.txt")
def add_missing_from_concept_kg(concept_docs, missing_ids):
    custom_docs = load_docs_from_jsonl(f"{base_dir}/data/eval_datasets/original_bc5cdr-disease/test_dictionary_docs.jsonl")
    for missing_id in missing_ids:
        for doc in concept_docs:
            if missing_id in doc.metadata['label'].strip().lower():
                custom_docs.append(doc)
                print(f"Added {missing_id} to custom docs")
                break
    save_docs_to_jsonl(custom_docs, f"{base_dir}/data/eval_datasets/original_bc5cdr-disease/test_dictionary_docs.jsonl")

omop_concepts = load_docs_from_jsonl(f"{base_dir}/data/output/concepts.jsonl")

convert_dict_to_docs(f"{base_dir}/data/eval_datasets/original_bc5cdr-disease/test_dictionary.txt")
custom_docs = load_docs_from_jsonl(f"{base_dir}/data/eval_datasets/original_bc5cdr-disease/test_dictionary_docs.jsonl")
add_synonyms_to_docs(omop_concepts, custom_docs, f"{base_dir}/data/eval_datasets/original_bc5cdr-disease/test_dictionary_docs.jsonl")
custom_docs = load_docs_from_jsonl(f"{base_dir}/data/eval_datasets/original_bc5cdr-disease/test_dictionary_docs.jsonl")

docs=only_label_docs(custom_docs)
save_docs_to_jsonl(docs, f"{base_dir}/data/eval_datasets/original_bc5cdr-disease/test_dictionary_docs_wo_syn.jsonl")


quries  = load_data('~/data/eval_datasets/original_bc5cdr-disease/combined_test_queries.txt')
print(f"quries: {quries[:2]}")

found_ids, missing_ids = check_missing_ids(quries, custom_docs)
print(f"missing_ids: {missing_ids}")  

