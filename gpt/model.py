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
    
class ResidualConnection(nn.Module):
  def __init__(self, features: int, dropout: float) -> None:
     super().__init__()
     self.dropout = nn.Dropout(dropout)
     self.norm = nn.LayerNorm(features)

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))
  

class FeedForwardBlock(nn.Module):
  def __init__(self, d_model: int, d_ff: int, ) -> None:
     super().__init__()
     self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
     self.dropout = nn.Dropout()
     self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

  def forward(self, x):
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
  

class EncoderBlock(nn.Module):
  def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
      super().__init__()
      self.self_attention_block = self_attention_block
      self.feed_forward_block = feed_forward_block
      self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
  def forward(self, x, src_mask):
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
    x = self.residual_connections[1](x, self.feed_forward_block)
    return x


class Encoder(nn.Module):
  def __init__(self, features: int, layers: nn.ModuleList) -> None:
     super().__init__()
     self.layers = layers
     self.norm = nn.LayerNorm(features)

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)

    return self.norm(x)


class DecoderBlock(nn.Module):
  def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    self.norm = nn.LayerNorm(features)

  """
  src_mask: mask the input source sequence 
  tgt_mask: used in the decoder to mask out padding tokens 
            to prevent them from seeing the target sequence
  """
  def forward(self, x, encoder_output, src_mask, tgt_mask):
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
    x = self.residual_connections[1](x, lambda x: self.self_attention_block(x,encoder_output, encoder_output, src_mask))
    x = self.residual_connections[2](x, lambda x: self.feed_forward_block(x))


class DecoderBlock(nn.Module):
  def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    self.norm = nn.LayerNorm(features)

  """
  src_mask: mask the input source sequence 
  tgt_mask: used in the decoder to mask out padding tokens 
            to prevent them from seeing the target sequence
  """
  def forward(self, x, encoder_output, src_mask, tgt_mask):
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
    x = self.residual_connections[1](x, lambda x: self.self_attention_block(x,encoder_output, encoder_output, src_mask))
    x = self.residual_connections[2](x, lambda x: self.feed_forward_block(x))

class InputEmbeddings(nn.Module):
  def __init__(self, d_model: int, vocab_size: int) -> None:
     super().__init__()
     self.d_model = d_model
     self.vocab_size = vocab_size
     self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model)
  
class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_length = seq_length
    self.dropout = nn.Dropout(dropout)

    # (seq_len, d_model)
    pe = torch.zeros(seq_length, d_model)

    # seq_length
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1) # (seq_len, 1)

    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)

    # apply sine to even indices
    pe[:, 0::2] = torch.sin(position * div_term)

    # apply cos to odd indices
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)

    self.register_buffer("pe", pe)
  
  def forward(self, x):
    x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
    return self.dropout(x)


class ProjectionLayer(nn.Module):
  def __init__(self, d_model, vocab_size) -> None:
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)
  
  def forward(self, x) -> None:
    return self.proj(x)
  
class Transformer(nn.Module):
  def __init__(self, 
               encoder: Encoder, 
               decoder: Decoder, 
               src_embed: InputEmbeddings, 
               tgt_embed: InputEmbeddings,
               src_pos: PositionalEncoding,
               tgt_pos: PositionalEncoding,
               projection_layer: ProjectionLayer):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.src_pos = src_pos
    self.tgt_pos = tgt_pos
    self.project_layer = projection_layer
  
  def encode(self, src, src_mask):
    src = self.src_embed(src)
    src = self.src_pos(src)
    return self.encoder(src, src_mask)
  
  def decode(self, 
             encoder_output: torch.Tensor, 
             src_mask: torch.Tensor, 
             tgt: torch.Tensor,
             tgt_mask: torch.Tensor):
    tgt = self.tgt_embed(tgt)
    tgt = self.tgt_pos(tgt)
    
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

  def project(self, x):
    return self.projection_layer(x)
