import torch
from torch import nn
import torch.nn.functional as F

import random, math

class SelfAttention(nn.Module):
    """
    Self Attention layer.
    
    :param size: Size of the model embeddings.
    :param heads: Number of heads of the model.
    """
    def __init__(self, emb_size: int = 128, heads: int = 6) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads

        self.tokeys    = nn.Linear(emb_size, emb_size * heads, bias=False)
        self.toqueries = nn.Linear(emb_size, emb_size * heads, bias=False)
        self.tovalues  = nn.Linear(emb_size, emb_size * heads, bias=False)
        self.output_layer = nn.Linear(emb_size * heads, emb_size)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param x: Vectors that will be used as keys, values, queries.
                  [batch_size x seq_len x embedding_size]
        :param mask: Mask that will 'remove' the attention from some 
                  of the key, value vectors. [batch_size x 1 x key_len]
        
        :return:
            - Returns a [batch x seq_len x embedding_size] with the contextualized 
                representations of the queries.
        """
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb_size, \
            f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_size})'

        keys    = self.tokeys(x)   .view(b, t, h, e).transpose(1, 2)
        queries = self.toqueries(x).view(b, t, h, e).transpose(1, 2)
        values  = self.tovalues(x) .view(b, t, h, e).transpose(1, 2)

        # compute scaled dot-product self-attention
        queries = queries / math.sqrt(e)

        # for each word Wi the score with all other words Wj 
        # for all heads inside the batch
        # [batch x num_heads x seq_len x seq_len]
        dot = torch.matmul(queries, keys.transpose(2, 3))

        # apply the mask (if we have one)
        # We add a dimension for the heads to it below: [batch, 1, 1, seq_len]
        if mask is not None:
            dot = dot.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention to convert the dot scores into probabilities.
        attention = F.softmax(dot, dim=-1)

        # We multiply the probabilities with the respective values
        context = torch.matmul(attention, values)
        # Finally, we reshape back to [batch x seq_len x num_heads * embedding_size]
        context = context.transpose(1, 2).contiguous().view(b, t, h * e)
        # We unify the heads by appliying a linear transform from:
        # [batch x seq_len x num_heads * embedding_size] -> [batch x seq_len x embedding_size]
        
        return self.output_layer(context)


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block 
        Self attention -> Layer Norm -> Feed Forward -> Layer Norm
    
    :param emb_size: Size of the model embeddings.
    :param heads: Number of heads of the model.
    :param ff_hidden_mult: Int that will specify the size of the 
        feed forward layer as a multiple of the embedding size.
    :param dropout: Dropout value to be applied between layers.
    """
    def __init__(self, 
                 emb_size: int = 128, 
                 heads: int = 6, 
                 ff_hidden_mult: int = 4, 
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.attention = SelfAttention(emb_size, heads=heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_mult * emb_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb_size, emb_size)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Encodes a sequence by passing it through 4 blocks: 
            Self Attention -> Layer Norm -> Feed Forward -> Layer Norm

        :param x: Vectors that will be used as keys, values, queries.
                  [batch_size x seq_len x embedding_size]
        :param mask: Mask that will 'remove' the attention from some 
                  of the key, value vectors. [batch_size x 1 x key_len]
        """
        # Self Attention Block
        attended = self.attention(x, mask)

        # Normalization Block
        x = self.norm1(attended + x)
        x = self.do(x)

        # Feedforward Block
        fedforward = self.ff(x)

        # Normalization Block
        x = self.norm2(fedforward + x)
        x = self.do(x)
        
        return x