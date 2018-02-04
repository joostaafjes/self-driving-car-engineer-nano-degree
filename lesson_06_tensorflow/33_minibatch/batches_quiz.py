import math
import numpy as np

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    assert len(features) >= batch_size

    batch = []
    size = len(features)
    for start_index in range(0, size, batch_size):
        end_index = min(start_index + batch_size, size)
        batch.append([features[start_index:end_index], labels[start_index:end_index]])

    return batch



