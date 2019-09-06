import numpy as np
import torch

from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.samplers.bucket_batch_sampler import BucketBatchSampler
from torchnlp.utils import (collate_tensors, lengths_to_mask,
                            sampler_to_iterator)


def set_seed(seed: int, cuda: bool=True):
    """
    Sets a numpy and torch seeds.
    :param seed: the seed value.
    :param cuda: if True sets the torch seed directly in cuda.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def prepare_sample(
        sample: dict, 
        text_encoder: WhitespaceEncoder,
        label_encoder: LabelEncoder,
        max_length: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Function that receives a sample from the Dataset iterator and prepares t
    he input to feed the transformer model.
    :param sample: dictionary containing the inputs to build the batch 
        (e.g: [{'source': 'This flight was amazing!', 'target': 'pos'}, 
               {'source': 'I hate Iberia', 'target': 'neg'}])
    :param text_encoder: Torch NLP text encoder for tokenization and vectorization.
    :param label_encoder: Torch NLP label encoder for vectorization of labels.
    :param max_length: Max length of the input sequences.
         If a sequence passes that value it is truncated.
    """
    sample = collate_tensors(sample)
    input_seqs, input_lengths = text_encoder.batch_encode(sample['source'])
    target_seqs = label_encoder.batch_encode(sample['target'])
    # Truncate Inputs
    if input_seqs.size(1) > max_length:
        input_seqs = input_seqs[:, :max_length]
    input_mask = lengths_to_mask(input_lengths).unsqueeze(1)
    return input_seqs, input_mask, target_seqs

def get_iterators(configs, train, test):
    """
    Function that receives the training and testing Datasets and build an iterator over them.
    :param configs: dictionary containing the configs from the default.yaml file.
    :param train: Dataset obj for training.
    :param test: Dataset obj for testing.
    """
    train_sampler = BucketBatchSampler(data=train,
            sort_key=lambda i: len(i['source'].split()),
            batch_size=configs['batch_size'], 
            drop_last=False
        )
    test_sampler = BucketBatchSampler(data=test,
            sort_key=lambda i: len(i['source'].split()),
            batch_size=configs.get('batch_size', 4), 
            drop_last=False
        )
    train_iter = sampler_to_iterator(train, train_sampler)
    test_iter = sampler_to_iterator(test, test_sampler)
    
    return train_iter, test_iter
