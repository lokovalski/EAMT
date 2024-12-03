from span_marker import SpanMarkerModel
from tqdm import tqdm
import time
import torch
import json

def sanity_check(filepath):
    '''
    Compares identified named entities from semeval data to entities identified by hugging face model.
    Intended to showcase model's strong performance, using precision rather than recall because 
    model includes entities that semeval omits for an unknown reason.

    Args: 
        filepath (str): Input file to process
    Returns:
        precision (int): 
    '''
    entities = identify_named_entities(filepath)


def identify_named_entities(filepath):
    """
    Identifies named entities in the given text using pretrained hugging face model
    Args:
        filepath (str): Input file to process.
    Returns:
        entities (list of dict): List of identified entities with their categories.
    """
    model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd", clean_up_tokenization_spaces = True)
    
