# Import necessary functions
import os
import json
import random
from entity_masking import load_entities_mapping, mask_entities_with_spanmarker, ensure_pad_token
from mt_translation import translate_with_placeholders_m2m
from entity_reintegration import reintegrate_entities
from ner_processing import sanity_check, identify_named_entities


def main():
    es_train = os.path.abspath('data/spanish_train.json')
    it_train = os.path.abspath('data/italian_train.json')

    print("Loading data... \n")
  
    nerRes = {"es": None,"it": None}

    # Step 2: Perform Named Entity Recognition (NER)
    for language in ['es', 'it']:
        print("Identifying named entities {}... \n".format(language))
        file = es_train if language == 'es' else it_train

        nerRes[language] = identify_named_entities(file, language, "train")
        input = {"entities_labeled": nerRes['target_entities_labeled'], "entities_predicted": nerRes['target_entities_predicted']}

        precision, recall, F1 = sanity_check(input)
        print("Accuracy of SpanMarker for Multilingual Named Entity Recognition {}\n".format(language))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(F1))
'''
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

def split(filepath, language):
    '''
    Given a filepath, randomly splits into train, test, and dev sets. Saves in separate JSON files in the `data` folder.

    Args:
        filepath (str): Path to preprocessed JSON file.
        language (str): Corresponding to language, for file labeling purposes.

    Returns:
        [train, test, dev] [list[dict], list[dict], list[dict]]: Lists of dictionaries for train, test, and dev sets.
    '''
    train, test, dev = 0.8, 0.1, 0.1


    with open(filepath, 'r') as jf:
        data = [json.loads(line) for line in jf]
      
    random.shuffle(data)
    
    # Split into train (80%), test (10%), and dev (10%)
    n = len(data)
    train_end = int(train * n)
    test_end = train_end + int(test * n)

    train, test, dev = data[:train_end], data[train_end:test_end], data[test_end:]

    # Ensure the "data" directory exists
    os.makedirs('data', exist_ok=True)

    # Save to separate JSON files
    train_path = os.path.join('data', f'{language}_train.json')
    test_path = os.path.join('data', f'{language}_test.json')
    dev_path = os.path.join('data', f'{language}_dev.json')
    
    with open(train_path, 'w') as train_file:
        json.dump(train, train_file, indent=4)

    with open(test_path, 'w') as test_file:
        json.dump(test, test_file, indent=4)

    with open(dev_path, 'w') as dev_file:
        json.dump(dev, dev_file, indent=4)

    return train, test, dev

if __name__ == "__main__":
    main()
