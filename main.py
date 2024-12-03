# Import necessary functions
import os
import random
import json
from ner_processing import sanity_check, identify_named_entities
#from entity_masking import mask_entities, save_mapping
#from mt_training import prepare_data_for_training, fine_tune_mt_model
#from entity_reintegration import reintegrate_entities
#from evaluation import compute_bleu_score, compute_entity_precision_recall

def main():
    #filepaths to preprocessed data --> see data_processing.py for more info
    es_datapath = os.path.abspath('data/spanish_w_labels.json')
    it_datapath = os.path.abspath('data/italian_w_labels.json')

    #split data into train and test sets
    es_train, it_train = split(es_datapath, "spanish", 0.1)[0], split(it_datapath, "italian", 0.1)[0]

'''
    # Step 2: Perform Named Entity Recognition (NER)
    print("Loading NER model...")
    ner_model = load_ner_model()
    
    print("Identifying named entities...")
    entities = [identify_named_entities(text) for text in source_texts]

    # Step 3: Mask Named Entities
    print("Masking entities...")
    masked_texts = []
    mappings = []
    for text, entity in zip(source_texts, entities):
        masked_text, mapping = mask_entities(text, entity)
        masked_texts.append(masked_text)
        mappings.append(mapping)

    # Save the mapping for reintegration later
    save_mapping(mappings, "entity_mapping.json")

    # Step 4: Fine-Tune the Machine Translation Model
    print("Fine-tuning MT model...")
    prepare_data_for_training()
    fine_tune_mt_model(masked_texts)

    # Step 5: Translate text and reintegrate entities
    print("Translating and reintegrating entities...")
    translated_texts = []  # Dummy: Replace with actual translations from the model
    final_translations = []
    for translation, mapping in zip(translated_texts, mappings):
        final_translation = reintegrate_entities(translation, mapping)
        final_translations.append(final_translation)

    # Step 6: Evaluate results
    print("Evaluating results...")
    references = ["La Tour Eiffel est à Paris."]  # Example references
    bleu = compute_bleu_score(final_translations, references)
    print(f"BLEU Score: {bleu}")

    entity_precision, entity_recall = compute_entity_precision_recall(final_translations, references)
    print(f"Entity Precision: {entity_precision}, Entity Recall: {entity_recall}")
'''

def split(filepath, language, testratio):
    '''
    Given a filepath, randomly splits into train and test sets. Saving in separate json files in data folder.

    Args:
        filepath (str): path to preprocessed json file
        testratio (float): percent of observations to use in test set
        langauge (str): corresponding to language, for file labeling purposes

    Returns:
        [train, test] [list[dicts], list[dicts]]: converted json to list of dictionaries
    '''
    
    with open(filepath, 'r') as jf:
        data = [json.loads(line) for line in jf]
      
    random.shuffle(data)
    
    # Split into train and test
    split_index = int(len(data) * (1 - testratio))
    train, test = data[:split_index], data[split_index:]

    # Ensure the "data" directory exists
    os.makedirs('data', exist_ok=True)

    # Save to separate JSON files
    train_path = os.path.join('data', '{}_train.json'.format(language))
    test_path = os.path.join('data', '{}_test.json'.format(language))
    
    with open(train_path, 'w') as train_file:
        json.dump(train, train_file, indent=4)

    with open(test_path, 'w') as test_file:
        json.dump(test, test_file, indent=4)

    return train, test

if __name__ == "__main__":
    main()
