import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class AdvancedTransformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_layers, dropout=0.1):
        super(AdvancedTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model)
        encoder_layers = nn.TransformerEncoderLayer(dim_model, num_heads, dim_feedforward=4*dim_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(dim_model, num_tokens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src_emb = self.dropout(self.pos_encoder(self.embedding(src)))
        transformer_output = self.transformer_encoder(src_emb)
        output = self.fc_out(transformer_output)
        return output