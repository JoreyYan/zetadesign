

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.drop = nn.Dropout(p_drop, inplace=True)
        #
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) *
                             -(math.log(10000.0) / d_model_half))
        self.register_buffer('div_term', div_term)

    def forward(self, x, idx_s):
        B, L, _, K = x.shape
        K_half = K // 2
        pe = torch.zeros_like(x)
        i_batch = -1
        for idx in idx_s:
            i_batch += 1
            sin_inp = idx.unsqueeze(1) * self.div_term
            emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)  # (L, K//2)
            pe[i_batch, :, :, :K_half] = emb.unsqueeze(1)
            pe[i_batch, :, :, K_half:] = emb.unsqueeze(0)
        x = x + torch.autograd.Variable(pe, requires_grad=False)
        return self.drop(x)

class Pair_emb_wo_templ(nn.Module):
    #TODO: embedding without template info
    def __init__(self, d_model=128, d_seq=128, p_drop=0.1):
        super(Pair_emb_wo_templ, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Linear(d_seq, self.d_emb)
        self.projection = nn.Linear(d_model + 1, d_model)
        self.pos = PositionalEncoding2D(d_model, p_drop=p_drop)
    def forward(self, seq, idx):
        # input:
        #   seq: target sequence (B, L, 20)
        B = seq.shape[0]
        L = seq.shape[1]
        seq = self.emb(seq) # (B, L, d_model//2)
        left  = seq.unsqueeze(2).expand(-1,-1,L,-1)
        right = seq.unsqueeze(1).expand(-1,L,-1,-1)
        seqsep = torch.abs(idx[:,:,None]-idx[:,None,:])+1
        seqsep = torch.log(seqsep.float()).view(B,L,L,1)
        #
        pair = torch.cat((left, right, seqsep), dim=-1)
        pair = self.projection(pair)
        return self.pos(pair, idx)