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
    tokenizer.src_lang = src_lang

    # Encode and translate the entire sentence
    encoded_text = tokenizer(masked_text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        max_length=128,  # Ensure sufficient length for translation
        early_stopping=True
    )

    # Decode the translated sentence
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text


