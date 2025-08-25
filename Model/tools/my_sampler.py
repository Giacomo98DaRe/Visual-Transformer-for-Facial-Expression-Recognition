# IMPORT

import torch
from torch.utils.data import WeightedRandomSampler

########################################################################################################################

def sampler_creation(train_AffectNet):

    class_weights = train_AffectNet.labels_cardinality()
    dataset_length = train_AffectNet.__len__()

    for emotion, cardinality in class_weights.items():
        cardinality = cardinality+1
        class_weights[emotion] = [dataset_length / cardinality]

    sample_weights = [0] * dataset_length

    for idx in range(0, len(sample_weights)):
        _, label = train_AffectNet.__getitem__(idx)
        sample_weights[idx] = (class_weights[str(label)])

    sample_weights_tensor = torch.tensor(sample_weights)
    sample_weights_flat = torch.flatten(sample_weights_tensor)
    sample_weights = sample_weights_flat.tolist()

    sampler = WeightedRandomSampler(sample_weights, train_AffectNet.__len__(), replacement=True)

    return sampler, sample_weights



class LabelsIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for _, _, label in self.dataset:
            yield label

def two_streams_sampler_creation(train_AffectNet):
    class_weights = train_AffectNet.labels_cardinality()
    dataset_length = len(train_AffectNet)

    for emotion, cardinality in class_weights.items():
        cardinality = cardinality+1
        class_weights[emotion] = dataset_length / cardinality

    sample_weights = [class_weights[str(label)] for label in LabelsIterableDataset(train_AffectNet)]

    sample_weights_tensor = torch.tensor(sample_weights)
    sample_weights_flat = torch.flatten(sample_weights_tensor)
    sample_weights = sample_weights_flat.tolist()

    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(train_AffectNet), replacement=True)

    return sampler, sample_weights