#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BERT Moudle definition."""

import torch

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.gelu import GELU


class BERT(torch.nn.Module):
    """BERT module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(self, idim,
                 attention_dim=512,
                 attention_heads=8,
                 linear_units=2048,
                 num_blocks=12,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 input_layer="embed",
                 pos_enc_class=PositionalEncoding,
                #  pos_enc_activate="gelu",
                 pos_enc_activate=GELU(),
                 normalize_before=True,
                 concat_after=False,
                 positionwise_layer_type="linear",
                 padding_idx=-1):
        """Construct an Encoder object."""
        super(BERT, self).__init__()

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                # torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("BERT Moudle Support only embed.")
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate, pos_enc_activate)
        else:
            raise NotImplementedError("BERT Moudle Support only linear.")
        self.transformer_encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after
            )
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        
    def forward(self, xs, masks):
        """Embed positions in tensor.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """

        xs = self.embed(xs)
        xs, masks = self.transformer_encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
