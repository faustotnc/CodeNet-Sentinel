import torch
import copy
import lightning as L
from torch.nn import MultiheadAttention
from torch.utils.data import Dataset
from . import LayerNormWithBias, FeedForward, PositionalEncoder, Tokenizer


class DecoderDataset(Dataset):
    def __init__(self, data, vocab_size):
        self.inputs = torch.stack(data["inputs"])
        self.attn_masks = torch.stack(data["attn_masks"])
        self.targets = torch.stack(data["targets"])

        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "attn_masks": self.attn_masks[idx],
            "targets": self.targets[idx],
        }

    def save_to(self, filename):
        torch.save(self, filename)

    def load_from(filename) -> Dataset:
        return torch.load(filename)


class DecoderBlock(torch.nn.Module):
    def __init__(self, n_head, n_dim, max_seq_len, mlp_dropout=0.0, attn_dropout=0.0, bias=False):
        super().__init__()

        # Layers for causal multi-head attention
        self.causal_mha_norm = LayerNormWithBias(n_dim, bias)
        self.causal_mha = MultiheadAttention(n_dim, n_head, dropout=attn_dropout, batch_first=True)
        self.causal_mha_dropout = torch.nn.Dropout(mlp_dropout)

        # Layers for memory multi-head attention
        self.mem_mha_norm = LayerNormWithBias(n_dim, bias)
        self.mem_mha = MultiheadAttention(n_dim, n_head, dropout=attn_dropout, batch_first=True)
        self.mem_mha_dropout = torch.nn.Dropout(mlp_dropout)

        # Layers for feedforward network
        self.feed_forward_norm = LayerNormWithBias(n_dim, bias)
        self.feed_forward = FeedForward(n_dim, n_dim, mlp_dropout, bias)
        self.feed_forward_dropout = torch.nn.Dropout(mlp_dropout)

        # Registering the causal mask as a buffer
        self.register_buffer('causal_mask', torch.triu(
            torch.ones((max_seq_len, max_seq_len)),
            diagonal=1
        ).bool())

    def forward(self, x, pad_mask, memory=None, mem_pad_mask=None, is_causal=True):
        # Causal multi-head attention with residual connection and dropout
        causal_mha_norm = self.causal_mha_norm(x)
        causal_mha_out, _ = self.causal_mha(
            query=causal_mha_norm,  # What we're looking for
            value=causal_mha_norm,  # Everything we have available
            key=causal_mha_norm,  # What we actually have access to (masked)
            key_padding_mask=pad_mask,
            attn_mask=self.causal_mask[
                :x.shape[1],
                :x.shape[1]
            ] if is_causal else None,
        )
        causal_mha = x + self.causal_mha_dropout(causal_mha_out)

        # Memory multi-head attention with residual connection and dropout
        if memory is not None:
            mem_mha_out, _ = self.mem_mha(
                query=self.mem_mha_norm(causal_mha),
                key=memory,
                value=memory,
                key_padding_mask=mem_pad_mask
            )
            decoder_out = causal_mha + self.mem_mha_dropout(mem_mha_out)
        else:
            decoder_out = causal_mha

        # Feedforward network with residual connection and dropout
        feed_forward_norm = self.feed_forward_norm(decoder_out)
        feed_forward_out = self.feed_forward(feed_forward_norm)
        feed_forward = decoder_out + self.feed_forward_dropout(feed_forward_out)

        return feed_forward
