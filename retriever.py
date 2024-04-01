import math
import torch
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast

def make_causal_mask(bsz, tgt_len, device, dtype):
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=1024, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size * 2.68)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        gate_proj = nn.functional.silu(self.gate_proj(x))
        up_proj = self.up_proj(x)
        down_proj = self.down_proj(gate_proj * up_proj)
        return down_proj

class Attention(nn.Module):
    def __init__(self, config, attention_mask, position_ids, flash=True):
        super(Attention, self).__init__()
        self.flash=flash
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.max_position_embeddings = config.sequence_length
        self.attention_mask = attention_mask
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.position_ids = position_ids
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        position_ids = self.position_ids[:, :seq_len]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        if self.flash:
            output = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        else:
            att = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert att.size() == (bsz, self.num_heads, seq_len, seq_len), "Attention weights shape error"
            att = att + self.attention_mask[:bsz, :, :seq_len, :seq_len]
            att = nn.functional.softmax(att, dim=-1)
            output = torch.matmul(att, value)

        output = output.transpose(1, 2).contiguous()
        assert output.size() == (bsz, seq_len, self.num_heads, self.head_dim), "Attention output shape error"
        output = output.reshape(bsz, seq_len, self.hidden_size)

        output = self.o_proj(output)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, config, attention_mask, position_ids, flash=True):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size)
        self.attn = Attention(config, attention_mask, position_ids, flash)
        self.ln_2 = RMSNorm(config.hidden_size)
        self.mlp = MLP(config.hidden_size)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Retriever(nn.Module):
    def __init__(self, device, ptdtype, config, flash=True):
        super(Retriever, self).__init__()
        self.device = device
        self.ptdtype = ptdtype
        self.config = config
        self.vocab_size = config.vocab_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        position_ids = torch.arange(0, config.sequence_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, config.sequence_length)
        attention_mask = make_causal_mask(config.batch_size, config.sequence_length, device, ptdtype)
        self.layers = nn.ModuleList([DecoderLayer(config, attention_mask, position_ids, flash) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for decoder_layer in self.layers:
            x = decoder_layer(x)
        x = self.norm(x)
        return x

def retriever(device, ptdtype, config, pretrained=False, model_path=None, flash=True):
    model = Retriever(device, ptdtype, config, flash)
    if pretrained:
        state_dict = torch.load(model_path)['state_dict']
        replacer = '_orig_mod.'
        if "module" == list(state_dict.keys())[0][:6]:
            replacer = 'module._orig_mod.'
        model.load_state_dict({k.replace(replacer, ''): v for k, v in state_dict.items()})
    return model

