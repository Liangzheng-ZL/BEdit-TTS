#!/usr/bin/env python3

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Variance predictor related modules."""

import torch
import torch.nn.functional as F
import numpy as np
import math

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
import logging

class FineGrainedConditionPredictor(torch.nn.Module):

    def __init__(
        self,
        idim: int,
        odim: int,
        num_gaussian: int,
        n_layers: int = 2,
        n_chans: int = 1024,
        kernel_size: int = 3,
        bias: bool = True,
        dropout_rate: float = 0.5,
        si_gru_size: int = 32,
    ):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        assert check_argument_types()
        super().__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.odim = odim
        self.num_gaussian = num_gaussian
        self.sd_gru = torch.nn.GRU(input_size=n_chans+odim,
                                   hidden_size=n_chans,
                                   batch_first=True)
        self.weight_projection = torch.nn.Linear(n_chans, num_gaussian)
        self.mean_A_projection = torch.nn.Linear(n_chans, odim)
        self.mean_b_projection = torch.nn.Linear(n_chans, odim)
        self.logvar_A_projection = torch.nn.Linear(n_chans, odim)
        self.logvar_b_projection = torch.nn.Linear(n_chans, odim)
        self.sd_mean_projection = torch.nn.Linear(odim, odim)
        self.sd_logvar_projection = torch.nn.Linear(odim, odim)
        self.si_gru = torch.nn.GRU(input_size=idim,
                                       hidden_size=si_gru_size,
                                       batch_first=True,
                                       bidirectional=True)
        self.mean_projection = torch.nn.Linear(si_gru_size*2, num_gaussian*odim)
        self.logvar_projection = torch.nn.Linear(si_gru_size*2, num_gaussian*odim)

    def forward(self, xs_si, xs, pl_condition, x_masks=None) -> torch.Tensor:

        B, L, _ = xs_si.size()

        # calculate transformation params(the A and b in Ax+b) for mean and logvar
        prev_pl_condition = torch.cat([torch.zeros_like(pl_condition[:, 0:1, :]), pl_condition[:, :-1, :]], dim=1)
        xs = xs.transpose(1, -1)  # (B, idim, L)
        for f in self.conv:
            xs = f(xs)  # (B, C, L)
        xs = torch.cat([xs.transpose(1, 2), prev_pl_condition], dim=-1)
        sd_gru_outputs, _ = self.sd_gru(xs)  # (B, L, n_chans)
        weights = F.softmax(self.weight_projection(sd_gru_outputs), dim=-1)  # (B, L, num_gaussian)
        mean_A = self.mean_A_projection(sd_gru_outputs).unsqueeze(-1)  # (B, L, odim, 1)
        mean_b = self.mean_b_projection(sd_gru_outputs).unsqueeze(-1)  # (B, L, odim, 1)
        logvar_A = self.logvar_A_projection(sd_gru_outputs).unsqueeze(-1)  # (B, L, odim, 1)
        logvar_b = self.logvar_b_projection(sd_gru_outputs).unsqueeze(-1)  # (B, L, odim, 1)

        # calculate speaker-independet params
        si_gru_output, _ = self.si_gru(xs_si)
        si_mean = self.mean_projection(si_gru_output).view(B, L, self.odim, self.num_gaussian)       # (B, L, odim, num_gaussian)
        si_logvar = self.logvar_projection(si_gru_output).view(B, L, self.odim, self.num_gaussian)   # (B, L, odim, num_gaussian)

        # apply speaker-dependent transformation to mean and logvar
        sd_mean = mean_A * si_mean + mean_b                 # (B, L, odim, num_gaussian)
        sd_mean = self.sd_mean_projection(F.tanh(sd_mean).transpose(-1, -2)).transpose(-1, -2).contiguous().view(B, L, -1)         # (B, L, odim*num_gaussian)
        sd_logvar = logvar_A * si_logvar + logvar_b         # (B, L, odim, num_gaussian)
        sd_logvar = self.sd_logvar_projection(F.tanh(sd_logvar).transpose(-1, -2)).transpose(-1, -2).contiguous().view(B, L, -1)   # (B, L, odim*num_gaussian)
        gmm_params = torch.cat([weights, sd_mean, sd_logvar], dim=-1)  # (B, L, num_gaussian + odim*num_gaussian*2)

        # mask the value
        if x_masks is not None:
            gmm_params = gmm_params.masked_fill(x_masks, 0.0)

        return gmm_params

    def inference(self, xs_si,
                  xs,
                  std_rescale=1,
                  use_top_k_gaussian_mixture=-1,
                  x_masks=None):

        B, L, _ = xs.shape

        # calculate speaker-independent params
        si_gru_output, _ = self.si_gru(xs_si)
        si_mean = self.mean_projection(si_gru_output).view(B, L, self.odim, self.num_gaussian)       # (B, L, odim, num_gaussian)
        si_logvar = self.logvar_projection(si_gru_output).view(B, L, self.odim, self.num_gaussian)   # (B, L, odim, num_gaussian)

        xs = xs.transpose(1, -1)  # (B, idim, L)
        for f in self.conv:
            xs = f(xs)  # (B, C, L)
        xs = xs.transpose(1, 2) # (B, L, C)
        sd_gru_input = torch.cat([xs[:, 0:1, :], torch.zeros(B, 1, self.odim).to(xs.device)], dim=-1)
        conditions = []
        sd_gru_hidden = None
        for t in range(1, L+1):
            # calculate transformation params
            sd_gru_output, sd_gru_hidden = self.sd_gru(sd_gru_input, sd_gru_hidden) # (B, 1, C)
            logit = self.weight_projection(sd_gru_output)                    # (B, 1, num_gaussian)
            mean_A = self.mean_A_projection(sd_gru_output).unsqueeze(-1)      # (B, 1, odim, 1)
            mean_b = self.mean_b_projection(sd_gru_output).unsqueeze(-1)      # (B, 1, odim, 1)
            logvar_A = self.logvar_A_projection(sd_gru_output).unsqueeze(-1)  # (B, 1, odim, 1)
            logvar_b = self.logvar_b_projection(sd_gru_output).unsqueeze(-1)  # (B, 1, odim, 1)

            # apply transformation to mean and logvar
            sd_mean = mean_A * si_mean[:, t-1:t] + mean_b          # (B, 1, odim, num_gaussian)
            sd_mean = self.sd_mean_projection(F.tanh(sd_mean).transpose(-1, -2)).transpose(-1, -2)
            sd_logvar = logvar_A * si_logvar[:, t-1:t] + logvar_b  # (B, 1, odim, num_gaussian)
            sd_logvar = self.sd_logvar_projection(F.tanh(sd_logvar).transpose(-1, -2)).transpose(-1, -2)

            # sample the condition for the t-th token
            condition = self.sample_from_gmm(logit, sd_mean, sd_logvar, std_rescale, use_top_k_gaussian_mixture)
            conditions.append(condition)

            # update value for recurrent prediction
            sd_gru_input = torch.cat([xs[:, t:t+1, :], condition], dim=-1) if t < L else None

        conditions = torch.cat(conditions, dim=1)

        if x_masks is not None:
            conditions = conditions.masked_fill(x_masks, 0.0)

        return conditions

    def transfer(self, xs_si, xs_tgt, xs_src, src_pl_condition, x_masks=None):
        B, L, _ = xs_tgt.shape

        # calculate speaker-independent params
        si_gru_output, _ = self.si_gru(xs_si)
        si_mean = self.mean_projection(si_gru_output).view(B, L, self.odim, self.num_gaussian)       # (B, L, odim, num_gaussian)
        si_logvar = self.logvar_projection(si_gru_output).view(B, L, self.odim, self.num_gaussian)   # (B, L, odim, num_gaussian)

        # calculate transformation params of source speaker
        prev_src_pl_condition = torch.cat([torch.zeros_like(src_pl_condition[:, 0:1, :]), src_pl_condition[:, :-1, :]], dim=1)
        xs_src = xs_src.transpose(1, -1)  # (B, idim, L)
        for f in self.conv:
            xs_src = f(xs_src)  # (B, C, L)
        xs_src = torch.cat([xs_src.transpose(1, 2), prev_src_pl_condition], dim=-1)
        src_gru_outputs, _ = self.sd_gru(xs_src)  # (B, L, n_chans)
        src_weights = F.softmax(self.weight_projection(src_gru_outputs), dim=-1)    # (B, L, num_gaussian)
        src_mean_A = self.mean_A_projection(src_gru_outputs).unsqueeze(-1)      # (B, L, odim, 1)
        src_mean_b = self.mean_b_projection(src_gru_outputs).unsqueeze(-1)      # (B, L, odim, 1)
        src_logvar_A = self.logvar_A_projection(src_gru_outputs).unsqueeze(-1)  # (B, L, odim, 1)
        src_logvar_b = self.logvar_b_projection(src_gru_outputs).unsqueeze(-1)  # (B, L, odim, 1)

        # apply speaker-dependent transformation to mean and logvar
        src_mean = src_mean_A * si_mean + src_mean_b           # (B, L, odim, num_gaussian)
        src_mean = self.sd_mean_projection(F.tanh(src_mean).transpose(-1, -2)).transpose(-1, -2)
        src_logvar = src_logvar_A * si_logvar + src_logvar_b   # (B, L, odim, num_gaussian)
        src_logvar = self.sd_logvar_projection(F.tanh(src_logvar).transpose(-1, -2)).transpose(-1, -2)

        # calculate log-likelihood and posterior of source condition
        components_logp = self.components_log_prob(src_pl_condition, src_mean, src_logvar)  # (B, L, num_gaussian)
        weights_posterior = src_weights * torch.exp(components_logp)  # (B, L, num_gaussian)
        weights_posterior /= torch.sum(weights_posterior, dim=-1).unsqueeze(-1)

        # transfer generation
        xs_tgt = xs_tgt.transpose(1, -1)  # (B, idim, L)
        for f in self.conv:
            xs_tgt = f(xs_tgt)  # (B, C, L)
        xs_tgt = xs_tgt.transpose(1, 2) # (B, L, C)
        tgt_gru_input = torch.cat([xs_tgt[:, 0:1, :], torch.zeros(B, 1, self.odim).to(xs_tgt.device)], dim=-1)
        tgt_gru_hidden = None
        conditions = []
        for t in range(1, L+1):
            # calculate transformation params of target speaker
            tgt_gru_output, tgt_gru_hidden = self.sd_gru(tgt_gru_input, tgt_gru_hidden) # (B, 1, C)
            tgt_mean_A = self.mean_A_projection(tgt_gru_output).unsqueeze(-1)      # (B, 1, odim, 1)
            tgt_mean_b = self.mean_b_projection(tgt_gru_output).unsqueeze(-1)      # (B, 1, odim, 1)
            tgt_logvar_A = self.logvar_A_projection(tgt_gru_output).unsqueeze(-1)  # (B, 1, odim, 1)
            tgt_logvar_b = self.logvar_b_projection(tgt_gru_output).unsqueeze(-1)  # (B, 1, odim, 1)

            # apply transformation of target speaker to mean and logvar
            tgt_mean = tgt_mean_A * si_mean[:, t-1:t] + tgt_mean_b          # (B, 1, odim, num_gaussian)
            tgt_mean = self.sd_mean_projection(F.tanh(tgt_mean).transpose(-1, -2)).transpose(-1, -2)
            tgt_logvar = tgt_logvar_A * si_logvar[:, t-1:t] + tgt_logvar_b  # (B, 1, odim, num_gaussian)
            tgt_logvar = self.sd_logvar_projection(F.tanh(tgt_logvar).transpose(-1, -2)).transpose(-1, -2)

            # sample the condition for the t-th token
            condition = self.sample_from_gmm(weights_posterior[:,t-1:t], tgt_mean, tgt_logvar, std_rescale=0.0, use_top_k_gaussian_mixture=1)
            conditions.append(condition)

            # update value for recurrent prediction
            tgt_gru_input = torch.cat([xs_tgt[:, t:t+1, :], condition], dim=-1) if t < L else None

        conditions = torch.cat(conditions, dim=1)

        if x_masks is not None:
            conditions = conditions.masked_fill(x_masks, 0.0)

        return conditions

    def gmm_weight_entropy(self, gmm_params, in_masks=None):
        weights, _, _ = self.parse_gmm_params(gmm_params)
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)
        if in_masks is not None:
            entropy = entropy.masked_select(in_masks)
        return entropy

    def parse_mean_logvar(self, mean_logvar):
        mean, logvar = torch.split(mean_logvar, mean_logvar.size(-1) // 2, dim=-1)  # (B, L, D x Num_Gaussian)
        mean = mean.view(mean.size(0), mean.size(1), self.odim, self.num_gaussian)  # (B, L, D, Num_Gaussian)
        logvar = logvar.view(logvar.size(0), logvar.size(1), self.odim, self.num_gaussian)
        return mean, logvar


    def parse_gmm_params(self, gmm_params):
        """
        gmm_param: shape:  ... , Num_Gaussian + Dim x Num_Gaussian x 2

        return:
        weights: shape:  ..., Num_Gaussian
        mean: shape:  ..., Dim, Num_Gaussian
        logvar: shape:  ..., Dim, Num_Gaussian
        """
        weights = gmm_params[:, :, :self.num_gaussian]  # (B, L, Num_Gaussian)
        mean_logvar = gmm_params[:, :, self.num_gaussian:]  # (B, L, D x Num_Gaussian x 2)
        mean, logvar = self.parse_mean_logvar(mean_logvar)
        return weights, mean, logvar

    def components_log_prob(self, x, mean, logvar):

        var = torch.exp(logvar)
        x = x.unsqueeze(-1).expand(mean.size())
        logp = - 0.5 * (self.odim * math.log(2 * math.pi) +
                        torch.sum(logvar + (x - mean) ** 2 / var, dim=-2))  # (B, L, Num_Gaussian)
        return logp

    def gmm_log_prob(self, x, gmm_params):

        weights, mean, logvar = self.parse_gmm_params(gmm_params)  # (B, L, Num_Gaussian), (B, L, D, Num_Gaussian), (B, L, D, Num_Gaussian)
        logp = self.components_log_prob(x, mean, logvar)  # (B, L, Num_Gaussian)

        # for stability, equivalent to:
        # logp = torch.log(torch.sum(gaussian_weights * torch.exp(logp), dim=-1))
        max_logp, _ = torch.max(logp, dim=-1)
        logp = torch.log(torch.sum(weights * torch.exp(logp-max_logp.unsqueeze(-1)), dim=-1)) + max_logp  # (B, L)

        return logp

    def sample_from_gmm(self, logits, mean, logvar, std_rescale=1, use_top_k_gaussian_mixture=-1):
        """
        return:
        condition: shape:  ..., Dim
        """

        if use_top_k_gaussian_mixture > 0:
            logits, sort_idx = torch.sort(logits, dim=-1, descending=True)
            sort_idx = sort_idx.unsqueeze(-2).expand_as(mean)
            mean = mean.gather(dim=-1, index=sort_idx)
            logvar = logvar.gather(dim=-1, index=sort_idx)
            logits[:,:,use_top_k_gaussian_mixture:] = -np.inf

        gaussian_idx = torch.argmax(logits + (-torch.log(-torch.log(torch.rand_like(logits)))), dim=-1) # (B, L)
        gaussian_idx = gaussian_idx.unsqueeze(-1).expand(mean.size()[:-1]).unsqueeze(-1)  # (B, L, D, 1)
        mean = mean.gather(dim=-1, index=gaussian_idx).squeeze(-1)  # (B, L, D)
        logvar = logvar.gather(dim=-1, index=gaussian_idx).squeeze(-1)

        std = torch.exp(0.5 * logvar)
        condition = mean + torch.randn_like(mean) * std * std_rescale # (B, L, D)
        return condition

class GlobalConditionPredictor(torch.nn.Module):

    def __init__(
        self,
        idim: int,
        spemb_dim: int,
        odim: int,
        num_gaussian: int,
        n_layers: int = 1,
        n_chans: int = 384,
        kernel_size: int = 3,
        bias: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        assert check_argument_types()
        super().__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.odim = odim
        self.num_gaussian = num_gaussian
        self.gru = torch.nn.GRU(input_size=n_chans,
                                hidden_size=n_chans,
                                bidirectional=True,
                                batch_first=True)
        self.projection = torch.nn.Linear(spemb_dim+n_chans*2, num_gaussian+odim*num_gaussian*2)

    def forward(self, hs: torch.Tensor, spembs, ilens) -> torch.Tensor:

        hs = hs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            hs = f(hs)  # (B, C, Tmax)
        hs = hs.transpose(1, 2)

        hs = torch.nn.utils.rnn.pack_padded_sequence(hs, ilens.cpu(), batch_first=True)
        _, out = self.gru(hs)  # (2, B, C)
        batch_size = out.size(1)
        out = out.permute(1, 2, 0).contiguous().view(batch_size, -1)

        condition_gmm_param = self.projection(torch.cat([spembs, out], dim=-1))  # (B, gmm_param_dim)

        return condition_gmm_param

    def inference(self, hs, spembs, ilens, std_rescale=1, use_top_k_gaussian_mixture=-1):
        gmm_param = self(hs, spembs, ilens)
        condition = self.sample_from_gmm(gmm_param, std_rescale, use_top_k_gaussian_mixture)
        return condition

    def parse_gmm_params(self, gmm_params):
        """
        gmm_param: shape:  ... , Num_Gaussian + Dim x Num_Gaussian x 2

        return:
        weights: shape:  ..., Num_Gaussian
        mean: shape:  ..., Dim, Num_Gaussian
        logvar: shape:  ..., Dim, Num_Gaussian
        """
        weights = F.softmax(gmm_params[:, :self.num_gaussian], dim=-1)  # (B, Num_Gaussian)
        mean_logvar = gmm_params[:, self.num_gaussian:]  # (B, D x Num_Gaussian x 2)
        mean, logvar = torch.split(mean_logvar, mean_logvar.size(-1) // 2, dim=-1)  # (B, D x Num_Gaussian)
        mean = mean.view(mean.size(0), self.odim, self.num_gaussian)  # (B, D, Num_Gaussian)
        logvar = logvar.view(logvar.size(0), self.odim, self.num_gaussian)
        return weights, mean, logvar

    def gmm_weight_entropy(self, gmm_params):
        weights, _, _ = self.parse_gmm_params(gmm_params)
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)
        return entropy

    def gmm_log_prob(self, x, gmm_params):
        """
        x: shape: ... , Dim
        weights: shape:  ..., Num_Gaussian
        mean: shape:  ..., Dim, Num_Gaussian
        logvar: shape:  ..., Dim, Num_Gaussian

        return:
        log_prob: shape:  ...
        """
        weights, mean, logvar = self.parse_gmm_params(gmm_params)  # (B, Num_Gaussian), (B, D, Num_Gaussian), (B, D, Num_Gaussian)

        var = torch.exp(logvar)
        x = x.unsqueeze(-1).expand(mean.size())
        logp = - 0.5 * (self.odim * math.log(2 * math.pi) +
                        torch.sum(logvar + (x - mean) ** 2 / var, dim=-2))  # (B, Num_Gaussian)

        # for stability, equivalent to:
        # logp = torch.log(torch.sum(gaussian_weights * torch.exp(logp), dim=-1))
        max_logp, _ = torch.max(logp, dim=-1)
        logp = torch.log(torch.sum(weights * torch.exp(logp-max_logp.unsqueeze(-1)), dim=-1)) + max_logp

        return logp


    def sample_from_gmm(self, gmm_param, std_rescale=1, use_top_k_gaussian_mixture=-1):
        """
        gmm_param: shape:  ... , Num_Gaussian + Dim x Num_Gaussian x 2

        return:
        condition: shape:  ..., Dim
        """
        logits = gmm_param[:, :self.num_gaussian]  # (B, Num_Gaussian)
        mean_logvar = gmm_param[:, self.num_gaussian:]  # (B, D x Num_Gaussian x 2)
        mean, logvar = torch.split(mean_logvar, mean_logvar.size(-1) // 2, dim=-1)  # (B, D x Num_Gaussian)
        mean = mean.view(mean.size(0), self.odim, self.num_gaussian)  # (B, D, Num_Gaussian)
        logvar = logvar.view(logvar.size(0), self.odim, self.num_gaussian)

        if use_top_k_gaussian_mixture > 0:
            logits, sort_idx = torch.sort(logits, dim=-1, descending=True)
            sort_idx = sort_idx.unsqueeze(-2).expand_as(mean)
            mean = mean.gather(dim=-1, index=sort_idx)
            logvar = logvar.gather(dim=-1, index=sort_idx)
            logits[:, use_top_k_gaussian_mixture:] = -np.inf

        gaussian_idx = torch.argmax(logits + (-torch.log(-torch.log(torch.rand_like(logits)))), dim=-1)  # (B, )
        gaussian_idx = gaussian_idx.unsqueeze(-1).expand(mean.size()[:-1]).unsqueeze(-1)  # (B, D, 1)
        mean = mean.gather(dim=-1, index=gaussian_idx).squeeze(-1)  # (B, D)
        logvar = logvar.gather(dim=-1, index=gaussian_idx).squeeze(-1)

        std = torch.exp(0.5 * logvar)
        condition = mean + torch.randn_like(mean) * std * std_rescale # (B, D)
        return condition

# class GlobalConditionPredictor(torch.nn.Module):
# 
#     def __init__(
#         self,
#         idim: int,
#         spemb_dim: int,
#         odim: int,
#         num_gaussian: int,
#         n_layers: int = 1,
#         n_chans: int = 384,
#         kernel_size: int = 3,
#         bias: bool = True,
#         dropout_rate: float = 0.5,
#     ):
#         """Initilize duration predictor module.
#         Args:
#             idim (int): Input dimension.
#             n_layers (int, optional): Number of convolutional layers.
#             n_chans (int, optional): Number of channels of convolutional layers.
#             kernel_size (int, optional): Kernel size of convolutional layers.
#             dropout_rate (float, optional): Dropout rate.
#         """
#         assert check_argument_types()
#         super().__init__()
#         self.conv = torch.nn.ModuleList()
#         for idx in range(n_layers):
#             in_chans = idim if idx == 0 else n_chans
#             self.conv += [
#                 torch.nn.Sequential(
#                     torch.nn.Conv1d(
#                         in_chans,
#                         n_chans,
#                         kernel_size,
#                         stride=1,
#                         padding=(kernel_size - 1) // 2,
#                         bias=bias,
#                     ),
#                     torch.nn.ReLU(),
#                     LayerNorm(n_chans, dim=1),
#                     torch.nn.Dropout(dropout_rate),
#                 )
#             ]
#         self.odim = odim
#         self.num_gaussian = num_gaussian
#         self.gru = torch.nn.GRU(input_size=n_chans,
#                                 hidden_size=n_chans,
#                                 bidirectional=True,
#                                 batch_first=True)
#         self.weight_projection = torch.nn.Linear(n_chans*2, num_gaussian)
#         self.mean_logvar_projection = torch.nn.Linear(spemb_dim+n_chans*2, odim*num_gaussian*2)
# 
#     def _forward(self, hs, spembs, ilens) -> torch.Tensor:
# 
#         hs = hs.transpose(1, -1)  # (B, idim, Tmax)
#         for f in self.conv:
#             hs = f(hs)  # (B, C, Tmax)
#         hs = hs.transpose(1, 2)
# 
#         hs = torch.nn.utils.rnn.pack_padded_sequence(hs, ilens, batch_first=True)
#         _, out = self.gru(hs)  # (2, B, C)
#         batch_size = out.size(1)
#         out = out.permute(1, 2, 0).contiguous().view(batch_size, -1)
# 
#         logit = self.weight_projection(out)  # (B, num_gaussian)
#         mean_logvar = self.mean_logvar_projection(torch.cat([spembs, out], dim=-1))  # (B, odim*num_gaussian*2)
# 
#         return logit, mean_logvar
# 
#     def forward(self, hs, spembs, ilens):
#         logit, mean_logvar = self._forward(hs, spembs, ilens)
#         return torch.cat([F.softmax(logit, dim=-1), mean_logvar], dim=-1)
# 
#     def inference(self, hs, spembs, ilens, std_rescale=1, use_top_k_gaussian_mixture=-1):
#         logit, mean_logvar = self._forward(hs, spembs, ilens)
#         mean, logvar = self.parse_mean_logvar(mean_logvar)
#         condition = self.sample_from_gmm(logit, mean, logvar, std_rescale, use_top_k_gaussian_mixture)
#         return condition
# 
#     def parse_mean_logvar(self, mean_logvar):
#         mean, logvar = torch.split(mean_logvar, mean_logvar.size(-1) // 2, dim=-1)  # (B, D x Num_Gaussian)
#         mean = mean.view(mean.size(0), self.odim, self.num_gaussian)  # (B, D, Num_Gaussian)
#         logvar = logvar.view(logvar.size(0), self.odim, self.num_gaussian)
#         return mean, logvar
# 
#     def parse_gmm_params(self, gmm_params):
#         """
#         gmm_param: shape:  ... , Num_Gaussian + Dim x Num_Gaussian x 2
# 
#         return:
#         weights: shape:  ..., Num_Gaussian
#         mean: shape:  ..., Dim, Num_Gaussian
#         logvar: shape:  ..., Dim, Num_Gaussian
#         """
#         weights = gmm_params[:, :self.num_gaussian]  # (B, Num_Gaussian)
#         mean_logvar = gmm_params[:, self.num_gaussian:]  # (B, D x Num_Gaussian x 2)
#         mean, logvar = self.parse_mean_logvar(mean_logvar)
#         return weights, mean, logvar
# 
#     def gmm_weight_entropy(self, gmm_params):
#         weights, _, _ = self.parse_gmm_params(gmm_params)
#         entropy = -torch.sum(weights * torch.log(weights), dim=-1)
#         return torch.mean(entropy)
# 
#     def gmm_log_prob(self, x, gmm_params):
#         """
#         x: shape: ... , Dim
#         weights: shape:  ..., Num_Gaussian
#         mean: shape:  ..., Dim, Num_Gaussian
#         logvar: shape:  ..., Dim, Num_Gaussian
# 
#         return:
#         log_prob: shape:  ...
#         """
#         weights, mean, logvar = self.parse_gmm_params(gmm_params)  # (B, Num_Gaussian), (B, D, Num_Gaussian), (B, D, Num_Gaussian)
# 
#         var = torch.exp(logvar)
#         x = x.unsqueeze(-1).expand(mean.size())
#         logp = - 0.5 * (self.odim * math.log(2 * math.pi) +
#                         torch.sum(logvar + (x - mean) ** 2 / var, dim=-2))  # (B, Num_Gaussian)
# 
#         # for stability, equivalent to:
#         # logp = torch.log(torch.sum(gaussian_weights * torch.exp(logp), dim=-1))
#         max_logp, _ = torch.max(logp, dim=-1)
#         logp = torch.log(torch.sum(weights * torch.exp(logp-max_logp.unsqueeze(-1)), dim=-1)) + max_logp
# 
#         return logp
# 
# 
#     def sample_from_gmm(self, logits, mean, logvar, std_rescale=1, use_top_k_gaussian_mixture=-1):
#         """
#         return:
#         condition: shape:  ..., Dim
#         """
#         if use_top_k_gaussian_mixture > 0:
#             logits, sort_idx = torch.sort(logits, dim=-1, descending=True)
#             sort_idx = sort_idx.unsqueeze(-2).expand_as(mean)
#             mean = mean.gather(dim=-1, index=sort_idx)
#             logvar = logvar.gather(dim=-1, index=sort_idx)
#             logits[:, use_top_k_gaussian_mixture:] = -np.inf
# 
#         gaussian_idx = torch.argmax(logits + (-torch.log(-torch.log(torch.rand_like(logits)))), dim=-1)  # (B, )
#         gaussian_idx = gaussian_idx.unsqueeze(-1).expand(mean.size()[:-1]).unsqueeze(-1)  # (B, D, 1)
#         mean = mean.gather(dim=-1, index=gaussian_idx).squeeze(-1)  # (B, D)
#         logvar = logvar.gather(dim=-1, index=gaussian_idx).squeeze(-1)
# 
#         std = torch.exp(0.5 * logvar)
#         condition = mean + torch.randn_like(mean) * std * std_rescale # (B, D)
#         return condition
