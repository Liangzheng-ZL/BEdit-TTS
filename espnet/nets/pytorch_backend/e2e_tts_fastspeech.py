#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FastSpeech related modules."""

import logging
from hashlib import new
from operator import index
from tkinter.messagebox import NO
from turtle import shape

import torch
import torch.nn.functional as F
from espnet.asr.asr_utils import get_model_conf, torch_load
from espnet.nets.pytorch_backend.bert.bert_moudle import BERT
from espnet.nets.pytorch_backend.bert.model.bert import BERT
from espnet.nets.pytorch_backend.e2e_tts_transformer import TTSPlot
from espnet.nets.pytorch_backend.fastspeech2.variance_predictor import \
    VariancePredictor
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (
    DurationPredictor, DurationPredictorLoss)
from espnet.nets.pytorch_backend.fastspeech.length_regulator import \
    LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import (make_non_pad_mask,
                                                    make_pad_mask, pad_list)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding, ScaledPositionalEncoding)
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool
from espnet.utils.fill_missing_args import fill_missing_args


class FeedForwardTransformer(TTSInterface, torch.nn.Module):
    """Feed Forward Transformer for TTS a.k.a. FastSpeech.

    This is a module of FastSpeech, feed-forward Transformer with duration predictor described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_, which does not require any auto-regressive
    processing during inference, resulting in fast decoding compared with auto-regressive Transformer.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    @staticmethod
    def add_arguments(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group("feed-forward transformer model setting")
        # network structure related
        group.add_argument("--adim", default=384, type=int,
                           help="Number of attention transformation dimensions")
        group.add_argument("--aheads", default=4, type=int,
                           help="Number of heads for multi head attention")
        group.add_argument("--elayers", default=6, type=int,
                           help="Number of encoder layers")
        group.add_argument("--eunits", default=1536, type=int,
                           help="Number of encoder hidden units")
        group.add_argument("--dlayers", default=6, type=int,
                           help="Number of decoder layers")
        group.add_argument("--dunits", default=1536, type=int,
                           help="Number of decoder hidden units")
        group.add_argument("--positionwise-layer-type", default="linear", type=str,
                           choices=["linear", "conv1d", "conv1d-linear"],
                           help="Positionwise layer type.")
        group.add_argument("--positionwise-conv-kernel-size", default=3, type=int,
                           help="Kernel size of positionwise conv1d layer")
        group.add_argument("--postnet-layers", default=0, type=int,
                           help="Number of postnet layers")
        group.add_argument("--postnet-chans", default=256, type=int,
                           help="Number of postnet channels")
        group.add_argument("--postnet-filts", default=5, type=int,
                           help="Filter size of postnet")
        group.add_argument("--use-batch-norm", default=True, type=strtobool,
                           help="Whether to use batch normalization")
        group.add_argument("--use-scaled-pos-enc", default=True, type=strtobool,
                           help="Use trainable scaled positional encoding instead of the fixed scale one")
        group.add_argument("--encoder-normalize-before", default=False, type=strtobool,
                           help="Whether to apply layer norm before encoder block")
        group.add_argument("--decoder-normalize-before", default=False, type=strtobool,
                           help="Whether to apply layer norm before decoder block")
        group.add_argument("--encoder-concat-after", default=False, type=strtobool,
                           help="Whether to concatenate attention layer's input and output in encoder")
        group.add_argument("--decoder-concat-after", default=False, type=strtobool,
                           help="Whether to concatenate attention layer's input and output in decoder")
        group.add_argument("--duration-predictor-layers", default=2, type=int,
                           help="Number of layers in duration predictor")
        group.add_argument("--duration-predictor-chans", default=384, type=int,
                           help="Number of channels in duration predictor")
        group.add_argument("--duration-predictor-kernel-size", default=3, type=int,
                           help="Kernel size in duration predictor")
        group.add_argument("--reduction-factor", default=1, type=int,
                           help="Reduction factor")
        # training related
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help="How to initialize transformer parameters")
        group.add_argument("--initial-encoder-alpha", type=float, default=1.0,
                           help="Initial alpha value in encoder's ScaledPositionalEncoding")
        group.add_argument("--initial-decoder-alpha", type=float, default=1.0,
                           help="Initial alpha value in decoder's ScaledPositionalEncoding")
        group.add_argument("--transformer-lr", default=1.0, type=float,
                           help="Initial value of learning rate")
        group.add_argument("--transformer-warmup-steps", default=4000, type=int,
                           help="Optimizer warmup steps")
        group.add_argument("--transformer-enc-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder except for attention")
        group.add_argument("--transformer-enc-positional-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder positional encoding")
        group.add_argument("--transformer-enc-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder self-attention")
        group.add_argument("--transformer-dec-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer decoder except for attention and pos encoding")
        group.add_argument("--transformer-dec-positional-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer decoder positional encoding")
        group.add_argument("--transformer-dec-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer decoder self-attention")
        group.add_argument("--transformer-enc-dec-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder-decoder attention")
        group.add_argument("--duration-predictor-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for duration predictor")
        group.add_argument("--postnet-dropout-rate", default=0.5, type=float,
                           help="Dropout rate in postnet")
        # Variance adaptor
        group.add_argument("--variance-predictor-kernel-size", default=7, type=int,
                           help="Kernel size in variance predictor")
        group.add_argument("--variance-predictor-chans", default=384, type=int,
                           help="Number of channels in variance predictor")
        group.add_argument("--variance-predictor-layers", default=3, type=int,
                           help="Number of layers in variance predictor")
        # loss related
        group.add_argument("--use-masking", default=True, type=strtobool,
                           help="Whether to use masking in calculation of loss")
        return parser

    def __init__(
        self, 
        idim, odim, 
        args=None):
        """Initialize feed-forward Transformer module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            args (Namespace, optional):
                - elayers (int): Number of encoder layers.
                - eunits (int): Number of encoder hidden units.
                - adim (int): Number of attention transformation dimensions.
                - aheads (int): Number of heads for multi head attention.
                - dlayers (int): Number of decoder layers.
                - dunits (int): Number of decoder hidden units.
                - use_scaled_pos_enc (bool): Whether to use trainable scaled positional encoding.
                - encoder_normalize_before (bool): Whether to perform layer normalization before encoder block.
                - decoder_normalize_before (bool): Whether to perform layer normalization before decoder block.
                - encoder_concat_after (bool): Whether to concatenate attention layer's input and output in encoder.
                - decoder_concat_after (bool): Whether to concatenate attention layer's input and output in decoder.
                - duration_predictor_layers (int): Number of duration predictor layers.
                - duration_predictor_chans (int): Number of duration predictor channels.
                - duration_predictor_kernel_size (int): Kernel size of duration predictor.
                - spk_embed_dim (int): Number of speaker embedding dimenstions.
                - reduction_factor (int): Reduction factor.
                - transformer_init (float): How to initialize transformer parameters.
                - transformer_lr (float): Initial value of learning rate.
                - transformer_warmup_steps (int): Optimizer warmup steps.
                - transformer_enc_dropout_rate (float): Dropout rate in encoder except attention & positional encoding.
                - transformer_enc_positional_dropout_rate (float): Dropout rate after encoder positional encoding.
                - transformer_enc_attn_dropout_rate (float): Dropout rate in encoder self-attention module.
                - transformer_dec_dropout_rate (float): Dropout rate in decoder except attention & positional encoding.
                - transformer_dec_positional_dropout_rate (float): Dropout rate after decoder positional encoding.
                - transformer_dec_attn_dropout_rate (float): Dropout rate in deocoder self-attention module.
                - transformer_enc_dec_attn_dropout_rate (float): Dropout rate in encoder-deocoder attention module.
                - use_masking (bool): Whether to use masking in calculation of loss.

        """
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # fill missing arguments
        args = fill_missing_args(args, self.add_arguments)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.reduction_factor = args.reduction_factor
        self.use_scaled_pos_enc = args.use_scaled_pos_enc
        self.use_masking = args.use_masking
        mel_idim = odim
        self.mel_idim = mel_idim
        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim,
            embedding_dim=args.adim,
            padding_idx=padding_idx
        )
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=args.transformer_enc_dropout_rate,
            positional_dropout_rate=args.transformer_enc_positional_dropout_rate,
            attention_dropout_rate=args.transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.encoder_normalize_before,
            concat_after=args.encoder_concat_after,
            positionwise_layer_type=args.positionwise_layer_type,
            positionwise_conv_kernel_size=args.positionwise_conv_kernel_size
        )

        # define speech encoder
        self.speechEncoder = Encoder(
            idim=mel_idim,
            attention_dim=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            input_layer='linear', 
            dropout_rate=0.1,
            positional_dropout_rate=0.1, 
            attention_dropout_rate=0.1,
            padding_idx=0
        )

        # variance adaptor
        pitch_dim = 4
        self.pitch_dim = pitch_dim
        self.pitch_predictor = VariancePredictor(idim=args.adim,
                                                 odim=pitch_dim,
                                                 n_layers=args.variance_predictor_layers,
                                                 n_chans=args.variance_predictor_chans,
                                                 kernel_size=args.variance_predictor_kernel_size)
        self.pitch_projection = torch.nn.Linear(pitch_dim, args.adim)
        energy_dim = 1
        self.energy_dim = energy_dim
        self.energy_predictor = VariancePredictor(idim=args.adim,
                                                  odim=energy_dim,
                                                  n_layers=args.variance_predictor_layers,
                                                  n_chans=args.variance_predictor_chans,
                                                  kernel_size=args.variance_predictor_kernel_size)
        self.energy_projection = torch.nn.Linear(energy_dim, args.adim)
        self.bert = BERT(
                        idim=mel_idim,
                        padding_idx=0
                        )

        # pitch and energy
        pitch_energy_dim = 5
        self.pitch_energy_dim = pitch_energy_dim
        self.pitch_energy_Encoder = Encoder(
            idim=pitch_energy_dim,
            attention_dim=512, 
            attention_heads=4,
            linear_units=2048,
            num_blocks=1, 
            input_layer='linear',
            dropout_rate=0.1, 
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1, 
            padding_idx=0
        )
        self.pitch_energy_bert = BERT(
            idim=512,
            num_blocks=2,
            padding_idx=0
        )
        self.pitch_energy_Decoder = Encoder(
            idim=0,
            attention_dim=512,
            attention_heads=4,
            linear_units=2048,
            num_blocks=1, 
            input_layer=None, 
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1, 
            padding_idx=0
        )
        self.pitch_energy_projection = torch.nn.Linear(args.adim, self.pitch_energy_dim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=args.adim,
            n_layers=args.duration_predictor_layers,
            n_chans=args.duration_predictor_chans,
            kernel_size=args.duration_predictor_kernel_size,
            dropout_rate=args.duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder because fastspeech's decoder is the same as encoder
        self.decoder = Encoder(
            idim=0,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            input_layer=None,
            dropout_rate=args.transformer_dec_dropout_rate,
            positional_dropout_rate=args.transformer_dec_positional_dropout_rate,
            attention_dropout_rate=args.transformer_dec_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.decoder_normalize_before,
            concat_after=args.decoder_concat_after,
            positionwise_layer_type=args.positionwise_layer_type,
            positionwise_conv_kernel_size=args.positionwise_conv_kernel_size
        )

        self.ttsbert_decoder = Encoder(
            idim=0,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            input_layer=None,
            dropout_rate=args.transformer_dec_dropout_rate,
            positional_dropout_rate=args.transformer_dec_positional_dropout_rate,
            attention_dropout_rate=args.transformer_dec_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.decoder_normalize_before,
            concat_after=args.decoder_concat_after,
            positionwise_layer_type=args.positionwise_layer_type,
            positionwise_conv_kernel_size=args.positionwise_conv_kernel_size
        )

        # define final projection
        self.feat_out = torch.nn.Linear(args.adim, odim * args.reduction_factor)
        self.tb_feat_out = torch.nn.Linear(args.adim, odim * args.reduction_factor)

        # initialize parameters
        self._reset_parameters(init_type=args.transformer_init,
                               init_enc_alpha=args.initial_encoder_alpha,
                               init_dec_alpha=args.initial_decoder_alpha)

        # define criterions
        self.duration_criterion = DurationPredictorLoss()
        self.mse_criterion = torch.nn.MSELoss()
        self.duration_cross_entropy = torch.nn.CrossEntropyLoss()

    def encoder_forward(self, xs, ilens):
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)
        return hs

    def condition_forward(self, hs_si, hs_mel, ilens, mel_ilens, 
                         pos, ds, ps, es,
                        in_masks, out_masks, in_mel_masks):
        hs = hs_si

        d_outs = self.duration_predictor(hs, ~in_masks)  # (B, Tmax)
        hs = self.length_regulator(hs, ds, ilens)  # (B, Lmax, adim)        

        p_outs = self.pitch_predictor(hs, ~out_masks)
        e_outs = self.energy_predictor(hs, ~out_masks)
        hs += self.pitch_projection(ps) + self.energy_projection(es)   
        hs += hs_mel
        mask_hs, _ = self.bert(hs, ~in_mel_masks)
        return hs, mask_hs, d_outs, p_outs, e_outs   

    def fastspeech_forward(self, hs_si, ilens,  
                         ds, ps, es,
                        in_masks, out_masks, ):
        hs = hs_si

        d_outs = self.duration_predictor(hs, ~in_masks)  # (B, Tmax)
        hs = self.length_regulator(hs, ds, ilens)  # (B, Lmax, adim)        

        p_outs = self.pitch_predictor(hs, ~out_masks)
        e_outs = self.energy_predictor(hs, ~out_masks)
        hs += self.pitch_projection(ps) + self.energy_projection(es)   
        return hs, d_outs, p_outs, e_outs

    def condition_berttts_fastspeech_inference(self, 
                                    hs_si, mask_mel_inputs, 
                                    ilens, mask_ilens,
                                    pos, phn_pos,
                                    ds, ps, es, 
                                    in_masks, out_masks
                                    ):

        hs = hs_si
        d_outs = self.duration_predictor.inference(hs, ~in_masks)  # (B, Tmax)
        new_ds = []
        change_lens = []
        for (d_out, d, position) in zip(d_outs, ds, phn_pos):
            p1, p2 = position
            change = d_out.size()[0] - d.size()[0]
            new_p2 = p2 + change
            mask_part_d_out = d_out[p1:new_p2]
            mask_part_d = d[p1:p2]
            d_temp_1 = d[:p1]
            d_temp_2 = d[p2:]
            new_d = torch.cat([d_temp_1, mask_part_d_out, d_temp_2], dim=0)
            new_d = new_d.unsqueeze(0)
            new_ds.append(new_d)
            before_change = 0
            for bnum in mask_part_d:
                before_change += bnum
            after_change = 0
            for anum in mask_part_d_out:
                after_change += anum
            change_lens.append(after_change - before_change)
        new_ds = torch.cat(new_ds, dim=0)
        change_lens = torch.IntTensor(change_lens)
        hs = self.length_regulator(hs, new_ds, ilens) 
        p_outs = self.pitch_predictor(hs)
        e_outs = self.energy_predictor(hs)

        new_ps = []
        new_es = []
        for i in range(ps.size()[0]):
            p_out = p_outs[i]
            p = ps[i]
            e_out = e_outs[i]
            e = es[i]
            position = pos[i]
            change_len = change_lens[i]
            p1, p2 = position
            new_p2 = p2 + change_len
            # new_p2 = p2

            mask_part_p_out = p_out[p1:new_p2, :]
            mask_part_e_out = e_out[p1:new_p2, :]
            p_temp_1 = p[:p1, :]
            p_temp_2 = p[p2:, :]
            e_temp_1 = e[:p1, :]
            e_temp_2 = e[p2:, :]
            new_p= torch.cat([p_temp_1, mask_part_p_out, p_temp_2], dim=0)
            new_e = torch.cat([e_temp_1, mask_part_e_out, e_temp_2], dim=0)
            new_ps.append(new_p.unsqueeze(0))
            new_es.append(new_e.unsqueeze(0))

        new_ps = torch.cat(new_ps, dim=0)
        new_es = torch.cat(new_es, dim=0)           
        hs += self.pitch_projection(new_ps) + self.energy_projection(new_es)   
        # hs += self.pitch_projection(ps) + self.energy_projection(es)
        after_mask_mel_inputs = torch.zeros(hs.size()[0], hs.size()[1], mask_mel_inputs.size()[-1]).to(hs.device)
        for i in range(mask_mel_inputs.size()[0]):
            mel = mask_mel_inputs[i]
            position = pos[i]
            p1, p2 = position
            change_len = change_lens[i]
            mel_temp_input_1 = mel[:p1,:]
            mel_temp_input_2 = mel[p2:,:]
            mask_zero = torch.zeros((p2+change_len-p1), mel.size()[1]).to(mel.device)
            # mask_zero = torch.zeros((p2-p1), mel.size()[1]).to(mel.device)
            after_mask_mel_input = torch.cat([mel_temp_input_1, mask_zero, mel_temp_input_2], dim=0)
            after_mask_mel_inputs[i] = after_mask_mel_input 
        mask_ilens = torch.tensor([after_mask_mel_inputs.shape[1]], dtype=torch.long, device=after_mask_mel_inputs.device)
        src_mask = (~make_pad_mask(mask_ilens.tolist())).to(mask_mel_inputs.device).unsqueeze(-2)
        hs_mel, _ = self.speechEncoder(after_mask_mel_inputs, src_mask)   
        hs += hs_mel
        in_mel_masks = make_non_pad_mask(mask_ilens).unsqueeze(-1).to(mask_mel_inputs.device)
        mask_hs, _ = self.bert(hs, ~in_mel_masks)
        before_outs = self.decoder_forward(mask_hs)
        new_outs = torch.zeros_like(before_outs).to(before_outs.device)
        for i in range(mask_mel_inputs.size()[0]):
            mel = mask_mel_inputs[i]
            mask_h = before_outs[i]
            position = pos[i]
            p1, p2 = position
            change_len = change_lens[i]
            mel_tmp_1 = mel[:p1,:]
            mel_tmp_2 = mel[p2:,:]
            new_p2 = p2 + change_len
            # new_p2 = p2
            mask_h_tmp = mask_h[p1:new_p2,:]
            after_out = torch.cat([mel_tmp_1, mask_h_tmp, mel_tmp_2], dim=0)
            new_outs[i] = after_out  
        after_postnet_outs = new_outs
        return after_postnet_outs

    def condition_berttts_bert_inference(self, 
                                    hs_si, mask_mel_inputs, 
                                    ilens, mask_ilens,
                                    pos, phn_pos,
                                    ds, pes, 
                                    in_masks, out_masks
                                    ):

        hs = hs_si
        tb_dur_outputs = self.duration_predictor.inference(hs, ~in_masks)  # (B, Tmax)
        new_ds = []
        change_lens = []
        new_phn_position = []
        for (d_out, d, position) in zip(tb_dur_outputs, ds, phn_pos):
            p1, p2 = position
            change = d_out.size()[0] - d.size()[0]
            new_p2 = p2 + change
            new_phn_position.append(p1)
            new_phn_position.append(new_p2)
            mask_part_d_out = d_out[p1:new_p2]
            mask_part_d = d[p1:p2]
            d_temp_1 = d[:p1]
            d_temp_2 = d[p2:]

            adjust_duration = False
            logging.info('Adjust duration:' + str(adjust_duration))
            logging.info('#mask part duration : ' + str(mask_part_d_out))
            if adjust_duration:
                ref_len = torch.sum(d_temp_1)+torch.sum(d_temp_2)
                pred_len = torch.sum(d_out) - torch.sum(mask_part_d_out)
                # change_rate = (ref_len / pred_len) if ref_len == 0 else 1.0
                if ref_len == 0:
                    change_rate = 1.0
                else:
                    change_rate = (ref_len / pred_len)
                # print("change rate:", change_rate)
                mask_part_d_out = torch.round(mask_part_d_out * change_rate).int()
                assert torch.min(mask_part_d_out) > 0, print(mask_part_d_out, change_rate)
                logging.info('#duration change rate:' + str(change_rate))
                logging.info('#After adjjusted mask part duration : ' + str(mask_part_d_out))
            new_d = torch.cat([d_temp_1, mask_part_d_out, d_temp_2], dim=0)
            new_d = new_d.unsqueeze(0)
            new_ds.append(new_d)

            before_change = 0
            for bnum in mask_part_d:
                before_change += bnum

            after_change = 0
            for anum in mask_part_d_out:
                after_change += anum

            change_lens.append(after_change - before_change)
            
        new_ds = torch.cat(new_ds, dim=0)
        change_lens = torch.IntTensor(change_lens)
        hs = self.length_regulator(hs, new_ds, ilens)  # (B, Lmax, adim)   
        # hs = self.length_regulator(hs, ds, ilens)  # (B, Lmax, adim)   
        # mask pitch/energy input
        after_mask_pitch_energy_inputs = torch.zeros(hs.size()[0], hs.size()[1], pes.size()[-1]).to(pes.device)
        for i in range(pes.size()[0]): # mel:[T, 80], position:[2]
            mel = mask_mel_inputs[i]
            pe_source = pes[i]
            position = pos[i]
            p1, p2 = position
            change_len = change_lens[i]
            pe_temp_input_1 = pe_source[:p1,:]
            pe_temp_input_2 = pe_source[p2:,:]
            pe_mask_zero = torch.zeros((p2+change_len-p1), pe_source.size()[1]).to(pe_source.device)
            # mask_zero = torch.zeros((p2-p1), mel.size()[1]).to(mel.device)
            after_mask_pitch_energy_input = torch.cat([pe_temp_input_1, pe_mask_zero, pe_temp_input_2], dim=0)
            after_mask_pitch_energy_inputs[i] = after_mask_pitch_energy_input 
   
        pe_mask_ilens = torch.tensor([after_mask_pitch_energy_inputs.shape[1]], dtype=torch.long, device=after_mask_pitch_energy_inputs.device)
        pitch_energy_masks = make_non_pad_mask(pe_mask_ilens).unsqueeze(-1).to(pes.device)
        src_pitch_energy_mask = (~make_pad_mask(pe_mask_ilens.tolist())).to(pes.device).unsqueeze(-2)
        tb_pitch_energy_inputs, _ = self.pitch_energy_Encoder(after_mask_pitch_energy_inputs, src_pitch_energy_mask)
        tb_pitch_energy_inputs += hs
        tb_pitch_energy_hs, _ = self.pitch_energy_bert(tb_pitch_energy_inputs, ~pitch_energy_masks)
        tb_pitch_energy_outputs = self.tb_pitch_energy_decoder_forward(tb_pitch_energy_hs, pe_mask_ilens)
        new_pes = []
        for i in range(pes.size()[0]):
            pe_out = tb_pitch_energy_outputs[i]
            pe = pes[i]
            position = pos[i]
            change_len = change_lens[i]
            p1, p2 = position
            new_p2 = p2 + change_len
            # new_p2 = p2
            mask_part_pe_out = pe_out[p1:new_p2, :]
            p_temp_1 = pe[:p1, :]
            p_temp_2 = pe[p2:, :]
            new_pe= torch.cat([p_temp_1, mask_part_pe_out, p_temp_2], dim=0)
            new_pes.append(new_pe.unsqueeze(0))
        new_pes = torch.cat(new_pes, dim=0)       
        hs += self.pitch_projection(new_pes[:,:,:4]) + self.energy_projection(new_pes[:,:,4:])   
        after_mask_mel_inputs = torch.zeros(hs.size()[0], hs.size()[1], mask_mel_inputs.size()[-1]).to(hs.device)
        for i in range(mask_mel_inputs.size()[0]): # mel:[T, 80], position:[2]
            mel = mask_mel_inputs[i]
            position = pos[i]
            p1, p2 = position
            change_len = change_lens[i]
            mel_temp_input_1 = mel[:p1,:]
            mel_temp_input_2 = mel[p2:,:]
            mask_zero = torch.zeros((p2+change_len-p1), mel.size()[1]).to(mel.device)
            # mask_zero = torch.zeros((p2-p1), mel.size()[1]).to(mel.device)
            after_mask_mel_input = torch.cat([mel_temp_input_1, mask_zero, mel_temp_input_2], dim=0)
            after_mask_mel_inputs[i] = after_mask_mel_input 
        mask_ilens = torch.tensor([after_mask_mel_inputs.shape[1]], dtype=torch.long, device=after_mask_mel_inputs.device)
        src_mask = (~make_pad_mask(mask_ilens.tolist())).to(mask_mel_inputs.device).unsqueeze(-2)
        hs_mel, _ = self.speechEncoder(after_mask_mel_inputs, src_mask)   
        hs += hs_mel
        in_mel_masks = make_non_pad_mask(mask_ilens).unsqueeze(-1).to(mask_mel_inputs.device)
        mask_hs, _ = self.bert(hs, ~in_mel_masks)
        before_outs = self.ttsbert_decoder_forward(mask_hs)
        new_outs = torch.zeros_like(before_outs).to(before_outs.device)
        new_position = []
        for i in range(mask_mel_inputs.size()[0]): # mel:[T, 80], position:[2]
            mel = mask_mel_inputs[i]
            mask_h = before_outs[i]
            position = pos[i]
            p1, p2 = position
            change_len = change_lens[i]
            mel_tmp_1 = mel[:p1,:]
            mel_tmp_2 = mel[p2:,:]
            new_p2 = p2 + change_len
            new_position.append(p1)
            new_position.append(new_p2)
            # new_p2 = p2
            mask_h_tmp = mask_h[p1:new_p2,:]
            after_out = torch.cat([mel_tmp_1, mask_h_tmp, mel_tmp_2], dim=0)
            new_outs[i] = after_out  
        after_postnet_outs = new_outs
        return after_postnet_outs, new_pes, new_ds, new_phn_position, new_position

    def fs_decoder_forward(self, hs, olens=None):
        # forward decoder
        if olens is not None:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        return before_outs 

    def tb_pitch_energy_decoder_forward(self, hs, olens=None):
        # forward decoder
        if olens is not None:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.pitch_energy_Decoder(hs, h_masks)  # (B, Lmax, adim)
        before_outs = self.pitch_energy_projection(zs).view(zs.size(0), -1, self.pitch_energy_dim)  # (B, Lmax, odim)

        return before_outs 

    def ttsbert_decoder_forward(self, hs, olens=None):
        # forward decoder
        if olens is not None:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.ttsbert_decoder(hs, h_masks)  # (B, Lmax, adim)
        before_outs = self.tb_feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        return before_outs 

    def forward(self, xs, ilens, ys, olens, mask_mel_inputs, mask_ilens, ds, spkids, pos, phn_pos, ps, es, pes, is_extra_speaker=False, *args, **kwargs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).

        Returns:
            Tensor: Loss value.

        """
        # remove unnecessary padded part (for multi-gpus)
        xs = xs[:, :max(ilens)] # [B, L]
        ys = ys[:, :max(olens)] # [B, T, dim=80]
        ds = ds[:, :max(ilens)] # [B, L] 
        ps = ps[:, :max(olens)] # [B, L, dim=4]
        es = es[:, :max(olens)] # [B, L, dim=1]
        pes = pes[:, :max(olens)] # pitch & energy [B, L, dim=5] 

        # mask operation
        # get mask part length
        mask_parts_ilens = []
        for position in pos:
            p1, p2 = position
            part_len = p2-p1
            mask_parts_ilens.append(part_len)
        mask_parts_ilens = torch.Tensor(mask_parts_ilens).to(mask_mel_inputs.device) # [19]       
        # get mask part
        ## mel, pitch and energy
        tmp_mask_parts = []
        tmp_pitch_energy_mask_parts = []
        
        for (mel, pitch_energy_feat, position) in zip(mask_mel_inputs, pes, pos):
            p1, p2 = position
            mask_part = mel[p1:p2]
            pitch_energy_mask_part = pitch_energy_feat[p1:p2]
            tmp_mask_parts.append(mask_part)
            tmp_pitch_energy_mask_parts.append(pitch_energy_mask_part)
        
        ## duration
        tmp_duration_mask_parts = []
        for (dur, phn_position) in zip(ds, phn_pos):
            phn_p1, phn_p2 = phn_position
            dur_mask_part = dur[phn_p1:phn_p2]
            tmp_duration_mask_parts.append(dur_mask_part)
            
        mask_parts = pad_list(tmp_mask_parts, 0).to(mask_mel_inputs.device)
        pitch_energy_mask_parts = pad_list(tmp_pitch_energy_mask_parts, 0).to(pes.device)
        duration_mask_parts = pad_list(tmp_duration_mask_parts, 0).to(ds.device)

        # source feats input 
        mask_mel_inputs = mask_mel_inputs[:, :max(mask_ilens)]

        ## after mask feats input
        after_mask_mel_inputs = torch.zeros_like(mask_mel_inputs).to(mask_mel_inputs.device)
        after_mask_pitch_energy_inputs = torch.zeros_like(pes).to(pes.device)
        for i in range(mask_mel_inputs.size()[0]): # mel:[T, 80], position:[2]
            mel = mask_mel_inputs[i]
            pitch_energy_feat = pes[i]
            position = pos[i]
            phn_position = phn_pos[i]
            p1, p2 = position
            # mel
            mel_temp_input_1 = mel[:p1,:]
            mel_temp_input_2 = mel[p2:,:]
            mask_zero = torch.zeros((p2-p1), mel.size()[1]).to(mel.device)
            after_mask_mel_input = torch.cat([mel_temp_input_1, mask_zero, mel_temp_input_2], dim=0)
            after_mask_mel_inputs[i] = after_mask_mel_input  

            # pitch and energy
            pitch_energy_temp_input_1 = pitch_energy_feat[:p1,:]
            pitch_energy_temp_input_2 = pitch_energy_feat[p2:,:]
            mask_pitch_energy_zero = torch.zeros((p2-p1), pitch_energy_feat.size()[1]).to(pitch_energy_feat.device)
            after_mask_pitch_energy_input = torch.cat([pitch_energy_temp_input_1, mask_pitch_energy_zero, pitch_energy_temp_input_2], dim=0)
            after_mask_pitch_energy_inputs[i] = after_mask_pitch_energy_input
        in_masks = make_non_pad_mask(ilens).to(xs.device)
        in_mel_masks = make_non_pad_mask(mask_ilens).unsqueeze(-1).to(mask_mel_inputs.device)
        out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
        mask_parts_masks = make_non_pad_mask(mask_parts_ilens).unsqueeze(-1).to(mask_parts.device)
        duration_mask_ilens = torch.tensor([duration_mask_parts.shape[1]], dtype=torch.long, device=duration_mask_parts.device)
        duration_mask_parts_masks = make_non_pad_mask(duration_mask_ilens).to(duration_mask_parts.device)
        pe_mask_ilens = torch.tensor([pitch_energy_mask_parts.shape[1]], dtype=torch.long, device=pitch_energy_mask_parts.device)
        pe_mask_parts_masks = make_non_pad_mask(pe_mask_ilens).unsqueeze(-1).to(pitch_energy_mask_parts.device)
        # forward propagation
        # text encoder
        hs_si = self.encoder_forward(xs, ilens) # [19, 32, 512]
        src_mask = (~make_pad_mask(mask_ilens.tolist())).to(mask_mel_inputs.device).unsqueeze(-2)
        hs_mel, _ = self.speechEncoder(after_mask_mel_inputs, src_mask) # [19, 184, 512]

        # fastspeech forward
        fs_d_outs = self.duration_predictor(hs_si, ~in_masks)  # (B, Tmax)
        fs_tmp_dur_mask_part_outs = []
        for (dur_out, phn_position) in zip(fs_d_outs, phn_pos):
            phn_p1, phn_p2 = phn_position
            fs_dur_mask_part = dur_out[phn_p1:phn_p2]
            fs_tmp_dur_mask_part_outs.append(fs_dur_mask_part)
        fs_duration_mask_parts_outs = pad_list(fs_tmp_dur_mask_part_outs, 0).to(fs_d_outs.device)


        hs_si = self.length_regulator(hs_si, ds, ilens)  # (B, Lmax, adim)   
        # pitch/energy bert inputs   
        pitch_energy_masks = make_non_pad_mask(olens).unsqueeze(-1).to(pes.device)
        src_pitch_energy_mask = (~make_pad_mask(olens.tolist())).to(pes.device).unsqueeze(-2)
        tb_pitch_energy_inputs, _ = self.pitch_energy_Encoder(after_mask_pitch_energy_inputs, src_pitch_energy_mask)
        tb_pitch_energy_inputs += hs_si
        tb_pitch_energy_hs, _ = self.pitch_energy_bert(tb_pitch_energy_inputs, ~pitch_energy_masks)
        tb_pitch_energy_outputs = self.tb_pitch_energy_decoder_forward(tb_pitch_energy_hs, olens)
        fs_p_outs = self.pitch_predictor(hs_si, ~out_masks)
        fs_e_outs = self.energy_predictor(hs_si, ~out_masks)
        hs_fs = hs_si + self.pitch_projection(ps) + self.energy_projection(es)  
        fs_mel_outs = self.fs_decoder_forward(hs_fs, olens)
        tb_mel_inputs = hs_fs+hs_mel
        tb_mel_outputs, _ = self.bert(tb_mel_inputs, ~in_mel_masks)
        before_outs = self.ttsbert_decoder_forward(tb_mel_outputs, mask_ilens)
        tmp_mel_mask_part_outs = []
        tmp_pitch_energy_part_outs = []
        for (mel, pitch_energy_out, position, phn_position) in zip(before_outs, tb_pitch_energy_outputs, pos, phn_pos):
            p1, p2 = position
            mask_part = mel[p1:p2]
            pitch_energy_mask_part = pitch_energy_out[p1:p2]
            tmp_mel_mask_part_outs.append(mask_part)
            tmp_pitch_energy_part_outs.append(pitch_energy_mask_part)
        mask_part_outs = pad_list(tmp_mel_mask_part_outs, 0).to(tb_mel_outputs.device)
        pitch_energy_mask_part_outs = pad_list(tmp_pitch_energy_part_outs, 0).to(tb_pitch_energy_outputs.device)
        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]
        # apply mask to remove padded part
        if self.use_masking:
            fs_d_outs = fs_d_outs.masked_select(in_masks)
            ds = ds.masked_select(in_masks)
            ys = ys.masked_select(out_masks)
            fs_mel_outs = fs_mel_outs.masked_select(out_masks)
            fs_p_outs = fs_p_outs.masked_select(out_masks)
            fs_e_outs = fs_e_outs.masked_select(out_masks)
            ps = ps.masked_select(out_masks)
            es = es.masked_select(out_masks)
            mask_part_outs = mask_part_outs.masked_select(mask_parts_masks)
            mask_parts = mask_parts.masked_select(mask_parts_masks)
            pitch_energy_mask_parts = pitch_energy_mask_parts.masked_select(pe_mask_parts_masks)
            pitch_energy_mask_part_outs = pitch_energy_mask_part_outs.masked_select(pe_mask_parts_masks)
            duration_mask_parts = duration_mask_parts.masked_select(duration_mask_parts_masks)
            fs_duration_mask_parts_outs = fs_duration_mask_parts_outs.masked_select(duration_mask_parts_masks)
        # fastspeech2 loss
        fs_l1_loss = F.l1_loss(fs_mel_outs, ys)
        fs_duration_loss = self.duration_criterion(fs_d_outs, ds)
        fs_mask_duration_loss = self.duration_criterion(fs_duration_mask_parts_outs, duration_mask_parts)
        fs_pitch_loss = self.mse_criterion(fs_p_outs, ps)
        fs_energy_loss = self.mse_criterion(fs_e_outs, es)
        fs_loss = fs_l1_loss + \
                  fs_duration_loss +\
                  fs_pitch_loss + \
                  fs_energy_loss

        
        # calculate loss
        tb_l1_loss = F.l1_loss(mask_part_outs, mask_parts)
        
        tb_pitch_energy_loss = self.mse_criterion(pitch_energy_mask_part_outs, pitch_energy_mask_parts)
        tb_loss = tb_l1_loss + \
                  tb_pitch_energy_loss

        loss =  fs_loss + tb_loss

        report_keys = [
            {"fs_l1_loss": fs_l1_loss.item()},
            {"fs_duration_loss": fs_duration_loss.item()},
            {"fs_mask_duration_loss": fs_mask_duration_loss.item()},
            {"fs_pitch_loss": fs_pitch_loss.item()},
            {"fs_energy_loss": fs_energy_loss.item()},
            {"tb_l1_loss": tb_l1_loss.item()},
            {"tb_pitch_energy_loss": tb_pitch_energy_loss.item()},
            {"loss": loss.item()},
        ]
        ##################################################

        # report extra information
        if self.use_scaled_pos_enc:
            report_keys += [
                {"encoder_alpha": self.encoder.embed[-1].alpha.data.item()},
                {"decoder_alpha": self.decoder.embed[-1].alpha.data.item()},
            ]
        self.reporter.report(report_keys)

        return loss
    def inference(self, xs, spkids, is_extra_spk=False, use_gt_condition=False, *args, **kwargs):
        """Generate the sequence of features given the sequences of characters.

        Args:
            xs (Tensor): Input sequence of characters (1, T,).
            inference_args (Namespace): Dummy for compatibility.
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).

        Returns:
            Tensor: Output sequence of features (L, odim).
            None: Dummy for compatibility.
            None: Dummy for compatibility.

        """
        # setup batch axis
        ilens = torch.tensor([xs.shape[1]], dtype=torch.long, device=xs.device)
        in_masks = make_non_pad_mask(ilens).to(xs.device)
        # inference
        hs_si = self.encoder_forward(xs, ilens)

        ys = kwargs['ys']
        ds = kwargs['ds']
        es = kwargs['es']
        ps = kwargs['ps']
        pes = kwargs['pes']
        pos = kwargs['pos']
        phn_pos = kwargs['phn_pos']

        mask_mel_inputs = kwargs['mask_mel_inputs']
        mask_ilens = torch.tensor([mask_mel_inputs.shape[1]], dtype=torch.long, device=mask_mel_inputs.device)

        olens = torch.tensor([ys.shape[1]], dtype=torch.long, device=ys.device)
        out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
        in_mel_masks = make_non_pad_mask(mask_ilens).unsqueeze(-1).to(mask_mel_inputs.device)
        fastspeech_banch = False
        if fastspeech_banch:
            after_postnet_outs = self.condition_berttts_fastspeech_inference(
                                                hs_si, mask_mel_inputs, 
                                                ilens, mask_ilens,
                                                # spembs,
                                                pos, phn_pos,
                                                ds, ps, es, 
                                                in_masks, out_masks
                                                )    
        else:
            after_postnet_outs, pe_outs, dur_outs, new_phn_position, new_position = self.condition_berttts_bert_inference(
                                    hs_si, mask_mel_inputs, 
                                    ilens, mask_ilens,
                                    pos, phn_pos,
                                    ds, pes, 
                                    in_masks, out_masks
                                    ) 
        return after_postnet_outs[0], None, None, pe_outs[0], dur_outs[0], new_phn_position, new_position

    def _source_mask(self, ilens):
        """Make masks for self-attention.

        Args:
            ilens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)

        """
        # x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        x_masks = make_non_pad_mask(ilens).to(self.feat_out.weight.device)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)

    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0, init_rb_dec_alpha=1.0):
        # initialize parameters
        initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)
            self.ttsbert_decoder.embed[-1].alpha.data = torch.tensor(init_rb_dec_alpha)

    @property
    def attention_plot_class(self):
        """Return plot class for attention weight plot."""
        return TTSPlot

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training. keys should match what `chainer.reporter` reports.

        If you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ["loss", "tb_pitch_energy_loss", "fs_mask_duration_loss", "fs_duration_loss", "fs_pitch_loss", "fs_energy_loss"]
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]

        return plot_keys
