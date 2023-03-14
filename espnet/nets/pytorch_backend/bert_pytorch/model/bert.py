import torch.nn as nn

from espnet.nets.pytorch_backend.bert_pytorch.model.transformer import TransformerBlock
from espnet.nets.pytorch_backend.bert_pytorch.model.embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden=512, n_layers=12, attn_heads=8, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        # self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.embedding = BERTEmbedding(embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        self.linear = nn.Linear(hidden, hidden)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, x_masks):
        # attention masking for padded token
        mask = x_masks

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        x = self.linear(x)
        # x = self.softmax(x)
        

        return x
