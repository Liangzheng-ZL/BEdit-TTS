import torch.nn as nn
# from espnet.nets.pytorch_backend.bert_pytorch.model.embedding.token import TokenEmbedding
from espnet.nets.pytorch_backend.bert_pytorch.model.embedding.position import PositionalEmbedding
# from espnet.nets.pytorch_backend.bert_pytorch.model.embedding.segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    # def __init__(self, vocab_size, embed_size, dropout=0.1):
    def __init__(self, embed_size, dropout=0.1):

        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.position = PositionalEmbedding(d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.position(sequence)
        return self.dropout(x)
    # def forward(self, sequence, segment_label):
    #     x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
    #     return self.dropout(x)
