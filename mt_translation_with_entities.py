from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load the M2M100 model and tokenizer
model_name = "facebook/m2m100_418M"
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

def translate_with_placeholders_m2m(masked_text, mapping, src_lang, tgt_lang):
    """
    Translates the non-entity parts of a text and re-inserts placeholders using M2M100.

    Args:
        masked_text (str): Text with placeholders for entities.
        mapping (dict): Mapping of placeholders to entity details.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.

    Returns:
        str: Translated text with placeholders intact.
    """
    # Split the text into parts
    parts = masked_text.split(" ")
    non_entity_parts = []
    placeholders = []

    # Separate placeholders and non-entity text
    for word in parts:
        if word.startswith("[ENTITY_") and word.endswith("]"):
            placeholders.append(word)
        else:
            non_entity_parts.append(word)

    # Translate non-entity parts
    non_entity_text = " ".join(non_entity_parts)
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(non_entity_text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    translated_non_entity_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # Reconstruct the sentence
    reconstructed_text = []
    non_entity_words = iter(translated_non_entity_text.split(" "))
    for word in parts:
        if word in mapping:  # Keep placeholders as is
            reconstructed_text.append(word)
        else:  # Replace non-entity words with the translated ones
            reconstructed_text.append(next(non_entity_words, ""))

    return " ".join(reconstructed_text)

