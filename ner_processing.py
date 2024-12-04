from span_marker import SpanMarkerModel
import numpy as np
from tqdm import tqdm
import time

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
    entities_labeled = inputs["source_entities_labeled"]
    entites_predicted = inputs["target_entities_labeled"]
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
    


def identify_named_entities(inputs, language):
    """
    Identifies named entities in the given text using pretrained hugging face model
    Args:
        inputs (list[dict]): Input to process
    Returns:
        dict(str: Tensors): source, source_entities_labeled, source_entities_predicted,
                        target, target_entities_labeled, target_entities_predicted
    """
    source = np.array([line['source'] for line in inputs], dtype = object)
    target = np.array([line['target'] for line in inputs], dtype = object)
    lstents = [obs['entities'] for obs in inputs]
    keys = [ent for key in lstents for ent in key]

    source_entities_labeled = np.array([key[enkey]['en'] for enkey in keys for key in lstents], dtype = object)
    target_entities_labeled = np.array([key[enkey][language] for enkey in keys for key in lstents], dtype = object)

    model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd", clean_up_tokenization_spaces = True)

    # TODO FINISH PREDICTIONS CORRECTLY
    source_pred = np.array([model.predict(sentence) for sentence in source], dtype = object)
    source_entities_predicted = np.array(source_pred["span"], dtype = object)
    target_pred = np.array([model.predict(sentence) for sentence in target], dtype = object)   
    target_entities_predicted = np.array(target_pred["span"], dtype = object)

    return {"source": source, "source_entities_labeled": source_entities_labeled, "source_entities_predicted": source_entities_predicted,
            "source_pred": source_pred, "target": target, "target_entities_labeled": target_entities_labeled, "target_entities_predicted":  target_entities_predicted,
            "target_pred": target_pred}
