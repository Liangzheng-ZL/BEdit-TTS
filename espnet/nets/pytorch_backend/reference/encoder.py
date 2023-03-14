import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import pad_list

def calculate_channels(L, kernel_size, stride, pad, n_convs):
    for _ in range(n_convs):
        L = (L - kernel_size + 2 * pad) // stride + 1
    return L

class ReferenceGlobalEncoder(nn.Module):
    '''
    inputs --- [N, T, n_mels]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, idim, filters, gru_size):

        super().__init__()
        K = len(filters)
        filters_ = [1] + filters

        convs = [nn.Conv2d(in_channels=filters_[i],
                           out_channels=filters_[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=filters[i])
             for i in range(K)])

        out_channels = calculate_channels(idim, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=filters[-1] * out_channels,
                          hidden_size=gru_size,
                          bidirectional=True,
                          batch_first=True)
        self.n_mel_channels = idim
        self.gru_size = gru_size

    def forward(self, inputs, lengths):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        K = len(self.bns)
        for conv, bn in zip(self.convs, self.bns):
            lengths = (lengths / 2 + 0.5).long()
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, T//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, T//2^K, 128*n_mels//2^K]

        gru_input = torch.nn.utils.rnn.pack_padded_sequence(out,
                                                            lengths.cpu(),
                                                            batch_first=True,
                                                            enforce_sorted=False)
        _, out = self.gru(gru_input)
        out = out.permute(1,2,0).contiguous().view(N, -1)
        return out


class ReferenceFineGrainedEncoder(nn.Module):

    def __init__(self, idim, filters, gru_size):

        super().__init__()
        
        K = len(filters)
        filters_ = [1] + filters

        convs = [nn.Conv2d(in_channels=filters_[i],
                           out_channels=filters_[i + 1],
                           kernel_size=(3, 3),
                           stride=(1, 1),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=filters[i])
             for i in range(K)])

        out_channels = calculate_channels(idim, 3, 1, 1, K)
        self.gru = nn.GRU(input_size=filters[-1] * out_channels,
                          hidden_size=gru_size,
                          bidirectional=True,
                          batch_first=True)
        self.n_mel_channels = idim
        self.gru_size = gru_size

    def forward(self, inputs, ds, lens):

        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, T, 128, n_mels]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, T, 32*n_mels]

        # batch_conditions = []
        # P = ds.size(1)
        # ds = ds.cpu().tolist()
        # for n in range(N):
        #     end = 0
        #     conditions = []
        #     for p in range(P):
        #         start = end
        #         end += ds[n][p]
        #         if start < end:
        #             _, condition = self.gru(out[n:n+1, start:end, :])
        #         else:
        #             condition = torch.zeros(self.gru_size*2, device=inputs.device)
        #         conditions.append(condition.view(-1))
        #     conditions = torch.stack(conditions)
        #     batch_conditions.append(conditions)
        # batch_conditions = torch.stack(batch_conditions)


        segments = []
        for n in range(N):
            end = 0
            for p in range(lens[n].item()):
                start = end
                end += ds[n, p].item()
                if start < end:
                    segment = out[n, start:end, :]  # (phn_dur, C)
                else:
                    segment = torch.zeros_like(out[0, :1])
                segments.append(segment)

        unsorted_ds = torch.LongTensor([segment.size(0) for segment in segments]).to(ds.device)  # (num_phns, )
        sorted_ds, sorted_indices = torch.sort(unsorted_ds, descending=True)
        unsorted_indices = nn.utils.rnn.invert_permutation(sorted_indices)
        segments = pad_list(segments, 0)  # (num_phns, max_phn_dur, C)
        segments = segments.index_select(0, sorted_indices)

        segments = nn.utils.rnn.pack_padded_sequence(segments,
                                                     sorted_ds.cpu(),
                                                     batch_first=True)
        _, conditions = self.gru(segments)  # (2, num_phns, *)
        conditions = conditions.transpose(0, 1).reshape(sorted_ds.size(0), -1)  # (num_phns, 2*)
        conditions = conditions[unsorted_indices, :]

        batch_conditions = torch.zeros(*ds.size(), conditions.size(-1), device=inputs.device)
        end = 0
        for i, num_phns in enumerate(lens):
            start = end
            end += num_phns.item()
            batch_conditions[i, :num_phns] = conditions[start:end]

        return batch_conditions
