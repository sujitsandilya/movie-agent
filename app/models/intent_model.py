import torch
import torch.nn as nn


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes, num_layers=4, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # x: (batch_size, seq_len) or (seq_len,) for single sample
        if x.dim() == 1:
            x = x.unsqueeze(0)  # add batch dim

        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)

        # Token + positional embeddings
        x = self.dropout(self.embedding(x) + self.pos_encoding(positions))  # (batch, seq_len, embed_dim)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (batch, seq_len, embed_dim)

        # Use [CLS] token (first token) for classification
        cls_output = x[:, 0, :]  # (batch, embed_dim)

        return self.classifier(self.dropout(cls_output))  # (batch, num_classes)
