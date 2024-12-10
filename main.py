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

    # Load SpanMarker model for NER
    print("Loading SpanMarker model for NER...")
    from span_marker import SpanMarkerModel
    ner_model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd")
    ensure_pad_token(ner_model)

    # Load M2M100 model and tokenizer for translation
    print("Loading M2M100 model for translation...")
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    for language, train_file in [('es', es_train), ('it', it_train)]:
        print(f"\nProcessing language: {language.upper()}...")

        # Step 1: Load and preprocess data
        print(f"Loading data from {train_file}...")
        data = load_entities_mapping(train_file)

        # Step 2: Process each entry
        translated_entries = []
        for idx, entry in enumerate(data): 

            # Extract source text
            source_text = entry['source']
            print(f"Original Text: {source_text}")

            # Mask entities
            masked_text, entity_mapping = mask_entities_with_spanmarker(source_text, ner_model)
            print(f"Masked Text: {masked_text}")
            print(f"Entity Mapping: {json.dumps(entity_mapping, indent=4)}")

            # Translate masked text
            translated_text = translate_with_placeholders_m2m(
                masked_text, entity_mapping, 
                src_lang="en", tgt_lang="es_Latn" if language == 'es' else "it_Latn",
                model=translation_model, tokenizer=translation_tokenizer
            )
            print(f"Translated Text with Placeholders: {translated_text}")

            # Reintegration of entities
            final_translation = reintegrate_entities(
                translated_text, entity_mapping,
                src_lang="en", tgt_lang="es" if language == 'es' else "it"
            )
            print(f"Final Translation: {final_translation}")

            # Store the processed entry
            translated_entries.append({
                "id": entry["id"],
                "original_text": source_text,
                "masked_text": masked_text,
                "translated_text": translated_text,
                "final_translation": final_translation,
                "entity_mapping": entity_mapping
            })

        # Save the results
        output_path = f"data/translated_{language}_first10.json"
        with open(output_path, 'w') as outfile:
            json.dump(translated_entries, outfile, indent=4)
        print(f"Translation results saved to {output_path}")

if __name__ == "__main__":
    main()
