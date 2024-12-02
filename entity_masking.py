import json

def load_entities_mapping(file_path):
    """
    Loads entities and their translations from the given JSON file.
    Args:
        file_path (str): Path to the JSON file containing entity mappings.
    Returns:
        dict: A dictionary mapping entity IDs to their translations.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def mask_entities(text, entities, entity_mapping):
    """
    Replaces named entities in the text with unique placeholders based on their category/type.
    Args:
        text (str): The original text.
        entities (dict): Entities with translations and IDs.
        entity_mapping (dict): Mapping of Q-IDs to their details.
    Returns:
        masked_text (str): Text with entities replaced by placeholders.
        mapping (dict): Mapping of placeholders to original entities.
    """
    masked_text = text
    mapping = {}
    type_counters = {}  # To track counts for each entity type
    offset = 0  # Track character shift due to replacements

    for entity_id, entity_data in entities.items():
        entity_text = entity_data['en']  # Assume English entity text is provided
        entity_details = entity_mapping.get(entity_id, {})
        entity_type = entity_details.get('category', 'UNKNOWN').upper()  # Default type to 'UNKNOWN'

        # Initialize the counter for this type if not already done
        if entity_type not in type_counters:
            type_counters[entity_type] = 1
        else:
            type_counters[entity_type] += 1

        # Generate placeholder
        placeholder = f"[ENTITY_{entity_type}_{type_counters[entity_type]}]"

        # Find the entity text in the original text
        start = text.find(entity_text)
        if start == -1:
            print(f"Entity '{entity_text}' not found in text.")
            continue  # Skip if the entity text isn't found

        end = start + len(entity_text)

        # Replace entity with placeholder
        start += offset
        end += offset
        masked_text = masked_text[:start] + placeholder + masked_text[end:]

        # Update offset and store mapping
        offset += len(placeholder) - len(entity_text)
        mapping[placeholder] = {
            "original_text": entity_text,
            "label": entity_details.get('label', {}).get('en', 'UNKNOWN'),
            "id": entity_id,
            "type": entity_type
        }

    return masked_text, mapping

def save_mapping(mapping, filename):
    """
    Saves the entity-to-placeholder mapping to a file.
    Args:
        mapping (dict): The entity-to-placeholder mapping.
        filename (str): File path to save the mapping.
    """
    with open(filename, 'w') as f:
        json.dump(mapping, f, indent=4)