# Import necessary functions
from ner_processing import load_ner_model, identify_named_entities
from entity_masking import mask_entities, save_mapping
from mt_training import prepare_data_for_training, fine_tune_mt_model
from entity_reintegration import reintegrate_entities
from evaluation import compute_bleu_score, compute_entity_precision_recall

def main():
    # Step 1: Load dataset
    print("Loading data...")
    source_texts = ["The Eiffel Tower is in Paris."]  # Example dataset

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

if __name__ == "__main__":
    main()
