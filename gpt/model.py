import torch
import torch.nn as nn
import math
import torch.nn.functional as F

torch.manual_seed(42)

B, T, C = 4, 8, 32 # batch, time, channels
x = torch.randn(B,T,C)



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        # Embedding vector size. Each token in the input sequence is mapped 
        # to an embedding vector of the size
        self.d_model = d_model 
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model is not divisible by h"

        self.d_k = d_model // num_heads # dimension of vector seen by each head

        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv

        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wv

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)


        if mask is not None:
            # small value to indicate not to tend to those positions
            attention_scores.masked_fil_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)            

        return (attention_scores @ value)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch. seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)

        x = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine the heads together
        # (batch, h, seq_len, d_k) --> () --> (batch, seq_len,d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        return self.w_o(x)