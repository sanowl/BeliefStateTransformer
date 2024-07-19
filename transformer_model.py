import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc_out = nn.Linear(dim_model, num_tokens)

    def forward(self, src):
        src_emb = self.embedding(src)
        transformer_output = self.transformer(src_emb, src_emb)
        output = self.fc_out(transformer_output)
        return output

    def get_residual_stream(self, src):
        src_emb = self.embedding(src)
        return self.transformer.encoder(src_emb)