from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def translate_entity(entity, src_lang, tgt_lang):
    """
    Translates an entity if its type requires translation, otherwise returns the original entity.

    Args:
        entity (str): The placeholder or entity to process.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.

    Returns:
        str: The translated or original entity.
    """
    # types that should not be translated
    non_translatable_types = {"PER", "INST", "MEDIA", "PLANT", "VEHI"}

    # Check if the entity contains one of the non-translatable types
    if any(ntype in entity for ntype in non_translatable_types):
        return entity 

    # Translate the entity using M2M100
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer.src_lang = src_lang
    encoded_entity = tokenizer(entity, return_tensors="pt")
    generated_tokens = model.generate(**encoded_entity, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    translated_entity = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return translated_entity


def reintegrate_entities(translation, mapping, src_lang, tgt_lang):
    """
    Replaces placeholders in the translated text with original or translated entities.

    Args:
        translation (str): Translated text with placeholders.
        mapping (dict): Mapping of placeholders to original entities.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.

    Returns:
        str: Translation with entities reintegrated.
    """
    for placeholder, entity_info in mapping.items():
        # Get the original entity
        original_entity = entity_info["original_text"]

        # Translate the entity if needed
        translated_entity = translate_entity(original_entity, src_lang, tgt_lang)

        # Replace placeholder with the translated entity
        translation = translation.replace(placeholder, translated_entity)

    return translation
