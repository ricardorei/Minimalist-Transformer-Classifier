import math
import pickle

import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from models import GTransformer
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import WhitespaceEncoder
from utils import get_iterators, prepare_sample, set_seed


def train_loop(
        configs: dict, 
        model: CTransformer, 
        opt: torch.optim.Adam, 
        train: Dataset, 
        test: Dataset, 
        text_encoder: WhitespaceEncoder,
        label_encoder: LabelEncoder) -> CTransformer:
    """
    Main training loop.
    :param configs: Configs defined on the default.yaml file.
    :param model: Transformer Classifier.
    :param opt: Adam optimizer.
    :param train: The dataset used for training.
    :param test: The dataset used for validation.
    :param text_encoder: Torch NLP text encoder for tokenization and vectorization.
    :param label_encoder: Torch NLP label encoder for vectorization of the labels.
    """
     for e in range(configs.get('num_epochs', 8)):
        print(f'\n Epoch {e}')
        model.train()
        
        nr_batches = math.ceil(len(train)/configs.get('batch_size', 8))
        train_iter, test_iter = get_iterators(configs, train, test)
        
        for sample in tqdm.tqdm(train_iter, total=nr_batches):
            # 0) Zero out previous grads
            opt.zero_grad()

            # 1) Prepare Sample
            input_seqs, input_mask, targets = prepare_sample(
                sample, text_encoder, label_encoder, configs.get('max_length', 256)
            )

            # 2) Run model
            out = model(input_seqs.cuda(), input_mask.cuda())

            # 3) Compute loss
            loss = F.nll_loss(out, target_seqs.cuda())
            loss.backward()

            # 4) clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if configs.get('gradient_clipping', -1) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), configs.get('gradient_clipping'))

            # 5) Optim step
            opt.step()

            # 6) Update number of seen examples...
            seen += input_seqs.size(0)
        
        validate(model, text_encoder, label_encoder, configs.get('max_length', 256), test_iter)
    return model


def validate(
        model: CTransformer, 
        text_encoder: WhitespaceEncoder, 
        label_encoder: LabelEncoder, 
        max_length: int,
        iterator) -> None:
    """
    Function that computes the accuracy over the validation set.
    :param model: Transformer Classifier.
    :param text_encoder: Torch NLP text encoder for tokenization and vectorization.
    :param label_encoder: Torch NLP label encoder for label vectorization.
    :param max_length: Max length of the source sequences.
    :param iterator: Iterator object over the test Dataset.
    """
    with torch.no_grad():
        model.train(False)
        tot, cor = 0.0, 0.0
        for sample in iterator:
            # 1) Prepare Sample
            input_seqs, input_mask, targets = prepare_sample(
                sample, text_encoder, label_encoder, configs.get('max_length', 1000)
            )
            # 2) Run model
            out = model(input_seqs.cuda(), input_mask.cuda()).argmax(dim=1).cpu()

            # 3) Compute total number of testing examples and number of correct predictions
            tot += float(input_seqs.size(0))
            cor += float((targets == out).sum().item())
        acc = cor / tot
        print(f'-- Test Accuracy {acc:.3}')
    

def train_manager(configs: dict) -> None:
    """
    Model Training functions.
    :param configs: Dictionary with the configs defined in default.yaml
    """
    with open('.preprocess.pkl', 'rb') as preprocess_file:
            text_encoder, label_encoder, train, test = pickle.load(preprocess_file)

    set_seed(configs.get('seed', 3))
    print(f'- nr. of training examples {len(train)}')
    print(f'- nr. of test examples {len(test)}')

    # Build Transformer model
    model = CTransformer(
                emb_size=configs.get('embedding_size', 128), 
                heads=configs.get('num_heads', 8), 
                depth=configs.get('depth', 6), 
                seq_length=configs.get('max_length', 1000), 
                num_tokens=text_encoder.vocab_size, 
                num_classes=label_encoder.vocab_size, 
                max_pool=configs.get('max_pool', False)
            )
    model.cuda()

    # Build Optimizer
    opt = torch.optim.Adam(lr=configs.get('lr', 0.0001), params=model.parameters())

    # Training Loop
    model = train_loop(configs, model, opt, train, test, text_encoder, label_encoder)
