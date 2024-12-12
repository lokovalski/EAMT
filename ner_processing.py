from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import torch
import os
import json
from span_marker import SpanMarkerModel

def sanity_check(inputs):
    '''
    Compares identified named entities from semeval data to entities identified by hugging face model.
    Intended to showcase model's strong performance, using precision rather than recall because 
    model includes entities that semeval omits for an unknown reason.

    Args: 
        inputs dict(Str: Tensor): entities_labeled, entities_predicted
    Returns:
        precision (int)
        recall (int)
        f1 (int)
    '''
    entities_labeled = inputs["entities_labeled"]
    entites_predicted = inputs["entities_predicted"]
    TP, FP, FN = 0, 0, 0

    # Loop through each observation
    for true, pred in zip(entities_labeled, entites_predicted):
        true_set = set(true)
        pred_set = set(pred)
        
        TP += len(true_set & pred_set)  # Intersection: correctly predicted
        FP += len(pred_set - true_set)  # Predicted but not true
        FN += len(true_set - pred_set)  # True but not predicted

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def identify_named_entities(inputs, language, split):
    """
    Identifies named entities in the given text using pretrained hugging face model
    Args:
        inputs (list[dict]): Input to process
    Returns:
        dict(str: Tensors): source, source_entities_labeled, source_entities_predicted,
                        target, target_entities_labeled, target_entities_predicted
    """
    source = np.array([line['source'] for line in inputs], dtype=object)
    target = np.array([line['target'] for line in inputs], dtype=object)
    lstents = [obs['entities'] for obs in inputs]

    source_entities_labeled = np.array([item[key]['en'] for item in lstents for key in item], dtype=object)
    target_entities_labeled = np.array([item[key][language] for item in lstents for key in item], dtype=object)

    model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd", clean_up_tokenization_spaces=True)

    source_pred = np.array([model.predict(sentence) for sentence in tqdm(source, desc="Processing source sentences")], dtype=object)
    source_entities_predicted = np.array([[d['span'] for d in row] for row in source_pred], dtype=object)
    target_pred = np.array([model.predict(sentence) for sentence in tqdm(target, desc="Processing target sentences")], dtype=object)
    target_entities_predicted = np.array([[d['span'] for d in row] for row in target_pred], dtype=object)

    results = {"source": source, "source_entities_labeled": source_entities_labeled, "source_entities_predicted": source_entities_predicted,
               "source_pred": source_pred, "target": target, "target_entities_labeled": target_entities_labeled,
               "target_entities_predicted": target_entities_predicted, "target_pred": target_pred}
 
    # Directory to save model output to reduce repeated computation
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'NERpredictions_{}{}.json'.format(split,language)), 'w') as resultfile:
        json.dump({"source": source.tolist(), "source_entities_labeled": source_entities_labeled.tolist(),
                   "source_entities_predicted": source_entities_predicted.tolist(),
                   "source_pred": source_pred.tolist(), "target": target.tolist(),
                   "target_entities_labeled": target_entities_labeled.tolist(),
                   "target_entities_predicted": target_entities_predicted.tolist(),
                   "target_pred": target_pred.tolist()}, resultfile, indent=4)

    return results
