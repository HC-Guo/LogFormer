import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(0)
        x = x + weight
        return self.dropout(x)


class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        z = self.gelu(self.linear1(x))
        z = self.linear2(z)

        return x+z


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, adapter_size=64, dim_feedforward=3072, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.adapter1 = Adapter(d_model, d_model, hidden_dim=adapter_size)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.adapter2 = Adapter(d_model, d_model, hidden_dim=adapter_size)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.dropout1(src2)
        src2 = self.adapter1(src2)
        src = self.norm1(src + src2)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)
        src2 = self.adapter2(src2)
        src = self.norm2(src + src2)

        return src

    def activate_adapter(self):
        tune_layers = [self.adapter1, self.adapter2, self.norm1, self.norm2]
        for layer in tune_layers:
            for param in layer.parameters():
                param.requires_grad = True


class Model(nn.Module):
    def __init__(self, mode, num_layers=4, adapter_size=64, dim=768, window_size=100, nhead=8, dim_feedforward=3072, dropout=0.1):
        super(Model, self).__init__()
        if mode == 'adapter':
            encoder_layer = TransformerEncoderLayer(
                dim, nhead, adapter_size=adapter_size, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                dim, nhead, dim_feedforward, dropout, batch_first=True)

        self.trans_encder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers)
        self.pos_encoder1 = PositionalEncoding(d_model=768)
        # self.pos_encoder2 = LearnedPositionEncoding(
        #    d_model=768, max_len=window_size)
        self.fc1 = nn.Linear(dim * window_size, 2)

    def forward(self, x):
        B, _, _ = x.size()
        # x = x*math.sqrt(self.dim)

        x = self.pos_encoder1(x)
        # x = self.pos_encoder2(x)

        x = self.trans_encder(x)  # mask默认None

        x = x.contiguous().view(B, -1)

        x = self.fc1(x)

        return x

    def train_adapter(self):
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.trans_encder.layers:
            layer.activate_adapter()
        for param in self.fc1.parameters():
            param.requires_grad = True

    def train_classifier(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.fc1.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    model = Model('adapter')
    pass
