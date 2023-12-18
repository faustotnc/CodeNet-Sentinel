import torch
from torch.nn import MultiheadAttention
from . import LayerNormWithBias, FeedForward
from torch.utils.data import Dataset


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
        """
        Initialize the DecoderBlock.

        Parameters:
            n_head (int): The number of attention heads.
            n_dim (int): The dimension of input features.
            max_seq_len (int): The maximum sequence length for creating the causal mask.
            mlp_dropout (float, optional): The dropout rate for the MLP layers. Defaults to 0.0.
            attn_dropout (float, optional): The dropout rate for the attention layers. Defaults to 0.0.
            bias (bool, optional): Whether to include a bias term. Defaults to False.
        """
        super().__init__()

        # Layers for causal multi-head attention
        self.causal_multihead_attn_norm = LayerNormWithBias(n_dim, bias)
        self.causal_multihead_attention = MultiheadAttention(
            n_dim, n_head, dropout=attn_dropout, batch_first=True)
        self.causal_multihead_attn_dropout = torch.nn.Dropout(mlp_dropout)

        # Layers for standard multi-head attention
        self.multihead_attn_norm = LayerNormWithBias(n_dim, bias)
        self.multihead_attention = MultiheadAttention(
            n_dim, n_head, batch_first=True)
        self.multihead_attn_dropout = torch.nn.Dropout(mlp_dropout)

        # Layers for feedforward network
        self.feed_forward_norm = LayerNormWithBias(n_dim, bias)
        self.feed_forward = FeedForward(n_dim, n_dim, mlp_dropout, bias)
        self.feed_forward_dropout = torch.nn.Dropout(mlp_dropout)

        # Registering the causal mask as a buffer
        self.register_buffer('causal_mask', torch.triu(
            torch.ones((max_seq_len, max_seq_len)), diagonal=1).bool())

    def forward(self, x, tgt_key_pad_mask, memory=None, memory_key_pad_mask=None, is_causal=True):
        """
        Forward pass of the decoder block.

        Parameters:
            x (torch.Tensor): The input tensor to the decoder block.
            tgt_key_pad_mask (torch.Tensor): The target key padding mask tensor for self-attention.
            memory (torch.Tensor, optional): The tensor from the encoder to be used in the 
                                              standard attention layer.
            memory_key_pad_mask (torch.Tensor, optional): The key padding mask for the encoder's 
                                                          tensor.
            is_causal (bool, optional): Whether to apply the causal mask for self-attention. 
                                        Defaults to True.

        Returns:
            torch.Tensor: The output tensor of the decoder block.
        """
        # Causal multi-head attention with residual connection and dropout
        causal_mha_norm = self.causal_multihead_attn_norm(x)
        causal_mha_out, _ = self.causal_multihead_attention(
            query=causal_mha_norm,
            key=causal_mha_norm,
            value=causal_mha_norm,
            key_padding_mask=tgt_key_pad_mask,
            attn_mask=self.causal_mask[:x.shape[1],
                                       :x.shape[1]] if is_causal else None,
        )
        causal_multi_head_attn = x + \
            self.causal_multihead_attn_dropout(causal_mha_out)

        # Standard multi-head attention with residual connection and dropout
        mha_norm = self.multihead_attn_norm(causal_multi_head_attn)
        mha_out, _ = self.multihead_attention(
            query=mha_norm,
            key=mha_norm if memory is None else memory,
            value=mha_norm if memory is None else memory,
            key_padding_mask=tgt_key_pad_mask if memory is None else memory_key_pad_mask
        )
        multi_head_attn = causal_multi_head_attn + \
            self.multihead_attn_dropout(mha_out)

        # Feedforward network with residual connection and dropout
        ff_norm = self.feed_forward_norm(multi_head_attn)
        ff_out = self.feed_forward(ff_norm)
        output = multi_head_attn + self.feed_forward_dropout(ff_out)

        return output
    

