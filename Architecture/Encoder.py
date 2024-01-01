import torch
from torch.nn import MultiheadAttention
from . import LayerNormWithBias, FeedForward


class EncoderBlock(torch.nn.Module):
    def __init__(self, n_head: int, n_dim: int, mlp_dropout: float = 0.0, bias: bool = False):
        """
        Initialize the EncoderBlock.

        Parameters:
            n_head (int): The number of attention heads.
            n_dim (int): The dimension of input features.
            mlp_dropout (float, optional): The dropout rate for the MLP layers. Defaults to 0.0.
            bias (bool, optional): Whether to include a bias term. Defaults to False.
        """
        super().__init__()

        # Layer normalization and multi-head attention components
        self.multihead_attn_norm = LayerNormWithBias(n_dim, bias)
        self.multihead_attention = MultiheadAttention(n_dim, n_head, batch_first=True)
        self.multihead_attn_dropout = torch.nn.Dropout(mlp_dropout)

        # Layer normalization and feedforward network components
        self.feed_forward_norm = LayerNormWithBias(n_dim, bias)
        self.feed_forward = FeedForward(n_dim, n_dim, mlp_dropout, bias)
        self.feed_forward_dropout = torch.nn.Dropout(mlp_dropout)

    def forward(self, x, key_pad_mask):
        """
        Forward pass of the encoder block.

        Parameters:
            x (torch.Tensor): The input tensor to the encoder block.
            key_pad_mask (torch.Tensor): The key padding mask tensor for self-attention.

        Returns:
            torch.Tensor: The output tensor of the encoder block.
        """
        # Multi-head attention with residual connection and dropout
        mha_norm = self.multihead_attn_norm(x)
        mha_out, _ = self.multihead_attention(mha_norm, mha_norm, mha_norm, key_padding_mask=key_pad_mask)
        multi_head_attn = x + self.multihead_attn_dropout(mha_out)

        # Feedforward network with residual connection and dropout
        ff_norm = self.feed_forward_norm(multi_head_attn)
        ff_out = self.feed_forward(ff_norm)
        output = multi_head_attn + self.feed_forward_dropout(ff_out)

        return output


class QnAConcatBlock(torch.nn.Module):
    def __init__(self, n_head: int, n_dim: int, mlp_dropout: float = 0.0, bias: bool = False):
        """
        Initialize the QnAConcatBlock.

        Parameters:
            n_head (int): The number of attention heads.
            n_dim (int): The dimension of input features.
            mlp_dropout (float, optional): The dropout rate for the MLP layers. Defaults to 0.0.
            bias (bool, optional): Whether to include a bias term. Defaults to False.
        """
        super().__init__()
        self.n_head = n_head

        # Layer normalization components
        self.question_attn_norm = LayerNormWithBias(n_dim, bias)
        self.answer_attn_norm = LayerNormWithBias(n_dim, bias)

        # The multi-head attention components
        self.multihead_attention = MultiheadAttention(n_dim, n_head, batch_first=True)
        self.multihead_attn_dropout = torch.nn.Dropout(mlp_dropout)

        # Layer normalization and feedforward network components
        self.feed_forward_norm = LayerNormWithBias(n_dim, bias)
        self.feed_forward = FeedForward(n_dim, n_dim, mlp_dropout, bias)
        self.feed_forward_dropout = torch.nn.Dropout(mlp_dropout)

    def forward(self, question_memory, answer_memory, question_attn_mask, answer_key_pad_mask):
        question_norm = self.question_attn_norm(question_memory)
        answer_norm = self.answer_attn_norm(answer_memory)

        # Create attention masks from the question padding masks
        attn_mask = question_attn_mask.unsqueeze(1).expand(-1, question_attn_mask.shape[1], -1)
        attn_mask = attn_mask + attn_mask.transpose(1, 2)
        attn_mask = attn_mask.repeat(self.n_head, 1, 1)

        mha_out, _ = self.multihead_attention(
            query=question_norm,  # What we're looking for
            value=answer_norm,  # Everything we have available
            key=answer_norm,  # What we actually have access to (masked)
            attn_mask=attn_mask,
            key_padding_mask=answer_key_pad_mask,
            is_causal=False
        )

        multi_head_attn = question_memory + self.multihead_attn_dropout(mha_out)

        # Feedforward network with residual connection and dropout
        ff_norm = self.feed_forward_norm(multi_head_attn)
        ff_out = self.feed_forward(ff_norm)
        output = multi_head_attn + self.feed_forward_dropout(ff_out)

        return output
