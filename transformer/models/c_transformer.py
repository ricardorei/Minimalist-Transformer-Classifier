import torch
from torch import nn
import torch.nn.functional as F

from models import TransformerBlock
from utils import d

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self,
                 vocab_size: int,
                 num_classes: int, 
                 emb_size: int = 128, 
                 heads: int = 4, 
                 depth: int = 4, 
                 seq_length: int = 256, 
                 max_pool: bool = True, 
                 dropout: float = 0.0) -> None:
        """
        :param vocab_size: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param emb_size: Embedding dimension
        :param heads: Number of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        :param dropout: dropout value to be applied between layers.
        """
        super().__init__()
        self.vocab_size, self.max_pool = vocab_size, max_pool
        self.token_embedding = nn.Embedding(embedding_dim=emb_size, num_embeddings=vocab_size)
        self.pos_embedding = nn.Embedding(embedding_dim=emb_size, num_embeddings=seq_length)

        self.tblocks = nn.ModuleList([
            TransformerBlock(emb_size=emb_size, heads=heads, dropout=dropout)
            for _ in range(depth)])

        self.toprobs = nn.Linear(emb_size, num_classes)
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Function that encodes the source sequence.
        :param x: Our vectorized source sequence. [Batch_size x seq_len]
        :param mask: Mask to be passed to the SelfAttention Block when encoding 
                the x sequence -> check SelfAttention.
        
        :returns:
            -  predicted log-probability vectors for each token based on the preceding tokens.
                 [Batch_size x seq_len x n_classes]
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.do(x)

        for tblock in self.tblocks:
            x = tblock(x, mask)

        mask = mask.squeeze(1).float()
        expanded_mask = torch.repeat_interleave(mask, e, 1).view(b, t, e)
        x = torch.mul(expanded_mask, x)
        
        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension
        x = self.toprobs(x)
        return F.log_softmax(x, dim=1)