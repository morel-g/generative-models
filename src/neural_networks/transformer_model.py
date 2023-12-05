# From https://github.com/pytorch/examples/blob/main/word_language_model/model.py
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import numpy as np
import torch.nn.functional as F
from src.neural_networks.ncsnpp.layerspp import GaussianFourierProjection
import math


class TimeEmbedBlock(nn.Module):
    def __init__(
        self, emb_dim, activation_fn=nn.GELU(), add_log=True, batch_first=True
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.activation_fn = activation_fn
        self.dense1 = nn.Linear(emb_dim, emb_dim)
        self.dense2 = nn.Linear(emb_dim, emb_dim)
        self.add_log = add_log
        self.batch_first = batch_first
        self.eps = 1e-4

        self.fourier_emb = GaussianFourierProjection(embedding_size=emb_dim // 2)

    def forward(self, t):
        if self.add_log:
            t_new = torch.log(t + self.eps)
        else:
            t_new = t
        embed = self.fourier_emb(t_new)
        embed = self.dense1(embed)
        embed = self.dense2(self.activation_fn(embed))
        unsqueeze_dim = 0 if not self.batch_first else 1
        return embed.unsqueeze(unsqueeze_dim)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the embeddings,
    so that the two can be summed. Here, we use sine and cosine functions of different
    frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        if self.batch_first:
            x = x + self.pe[:, : x.size(1), :]
        else:
            x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


# class TransformerModel(nn.Transformer):
class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self,
        nb_tokens,
        emb_dim,
        nb_heads,
        hidden_dim,
        nb_layers,
        batch_first=False,
        dropout=0.5,
    ):
        super(TransformerModel, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nb_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=nb_layers
        )

        self.model_type = "Transformer"
        self.src_mask = None
        self.batch_first = batch_first
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, batch_first=batch_first)
        self.time_embed = TimeEmbedBlock(emb_dim, batch_first=batch_first)

        self.input_emb = nn.Embedding(nb_tokens, emb_dim)
        self.emb_dim = emb_dim
        self.decoder = nn.Linear(emb_dim, nb_tokens)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, t, has_mask=False):
        if has_mask:
            device = x.device
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                seq_len = x.shape[0] if not self.batch_first else x.shape[1]
                mask = self._generate_square_subsequent_mask(seq_len).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        x = self.input_emb(x) * math.sqrt(self.emb_dim)
        x = x + self.time_embed(t)
        x = self.pos_encoder(x)
        # output = self.encoder(x, mask=self.src_mask)
        output = self.transformer_encoder(x, src_key_padding_mask=self.src_mask)
        output = self.decoder(output)
        # output = F.softmax(output, dim=-1)

        return output
