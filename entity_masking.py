import json
from span_marker import SpanMarkerModel

def load_entities_mapping(file_path):
    """
    Loads entities and their translations from the given JSON file, 
    Args:
        file_path (str): Path to the JSON file containing entity mappings.
    Returns:
        list: A list of dictionaries representing the JSON data.
    """

    with open(file_path, 'r') as jf:
        try:
            return json.load(jf)  # Load the entire JSON array
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")
            return []


def mask_entities_with_spanmarker(text, model):
    """
    Masks named entities in text using predictions from SpanMarkerModel.

    Args:
        text (Union[str, List[str]]): Input text(s). Can be a single string or a list of strings.
        model (SpanMarkerModel): Pretrained SpanMarker model for entity prediction.

    Returns:
        Union[Tuple[str, dict], List[Tuple[str, dict]]]:
            - For a single string:
                - masked_text (str): Text with entities replaced by placeholders.
                - mapping (dict): Mapping of placeholders to original entities.
            - For a list of strings:
                - List of (masked_text, mapping) tuples.
    """

    # Check if input is a single string or a list of strings
    is_single_string = isinstance(text, str)
    if is_single_string:
        text = [text]  # Convert single string to list 


    # Perform predictions
    predictions = model.predict(text)  

    results = []
    for i, sentence in enumerate(text):
        masked_text = sentence
        mapping = {}
        type_counters = {}
        offset = 0

        # If no predictions for this sentence
        if not predictions[i]:  # Empty list
            print(f"No entities found for text: {sentence}")
            results.append((masked_text, mapping))
            continue

        # Process each entity
        for entity in predictions[i]:
            entity_text = entity['span']
            entity_type = entity['label'].upper()
            start = entity['char_start_index']
            end = entity['char_end_index']

            # Counter for entity type
            if entity_type not in type_counters:
                type_counters[entity_type] = 1
            else:
                type_counters[entity_type] += 1

            # Generate a placeholder
            placeholder = f"[ENTITY_{entity_type}_{type_counters[entity_type]}]"

            # Replace entity with placeholder in the text
            start += offset
            end += offset
            masked_text = masked_text[:start] + placeholder + masked_text[end:]
            offset += len(placeholder) - len(entity_text)

            # Add to mapping
            mapping[placeholder] = {
                "original_text": entity_text,
                "type": entity_type,
                "confidence": entity['score'],
                "start": start,
                "end": end
            }

        results.append((masked_text, mapping))

    #if text input is a single string
    return results[0] if is_single_string else results

def ensure_pad_token(model):
    if not hasattr(model.tokenizer, "pad_token") or model.tokenizer.pad_token is None:
        print("Setting pad_token manually...")
        model.tokenizer.pad_token = "[PAD]"
        model.tokenizer.pad_token_id = model.tokenizer.convert_tokens_to_ids("[PAD]")


def save_mapping(mapping, filename):
    """
    Saves the entity-to-placeholder mapping to a file.
    
    Args:
        mapping (dict): The entity-to-placeholder mapping.
        filename (str): File path to save the mapping.
    """
    with open(filename, 'w') as f:
        json.dump(mapping, f, indent=4)

