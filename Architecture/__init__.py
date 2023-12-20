import torch
import math
from transformers import AutoTokenizer

# Initialize the GPT tokenizer and add special tokens
Tokenizer = AutoTokenizer.from_pretrained("gpt2")
num_new_tokens = Tokenizer.add_special_tokens({
    'pad_token': '[PAD]',
    'bos_token': '[BOS]',
    'eos_token': '[EOS]',
    'sep_token': '[SEP]',
    'cls_token': '[CLS]',
    'mask_token': '[MASK]',
})
VOCAB_SIZE = Tokenizer.vocab_size + num_new_tokens


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size, warmup_steps, factor=1.0, last_epoch=-1):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self._step_count, 1)
        scale = self.factor * (self.model_size ** (-0.5) *
                               min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        return [base_lr * scale for base_lr in self.base_lrs]


class PositionalEncoder(torch.nn.Module):
    def __init__(self, n_dim, max_len=1024):
        """
        Initialize the PositionalEncoder.

        Parameters:
            n_dim (int): The dimensionality of the model's input and output.
            max_len (int, optional): The maximum length of the input sequences. Defaults to 1024.
        """
        super(PositionalEncoder, self).__init__()

        # Compute the positional encodings once in advance
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, n_dim, 2).float() * -(math.log(10000.0) / n_dim)
        )
        pos_enc = torch.zeros((max_len, n_dim))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer so it moves with the model during training and evaluation
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        """
        Add positional encodings to the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The output tensor with positional encodings added. The output has the
                          same shape as the input tensor.
        """
        return x + self.pos_enc[:x.size(1), :].unsqueeze(0)


class LayerNormWithBias(torch.nn.Module):
    def __init__(self, n_dim, bias):
        """
        Initialize the LayerNormWithBias module.

        Parameters:
            n_dim (int): The dimension of the input features.
            bias (bool): If True, a bias term is used in the normalization. If False, 
                         no bias term is used.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(n_dim))
        self.bias = torch.nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, input):
        """
        Apply layer normalization to the input tensor.

        Parameters:
            input (torch.Tensor): The input tensor to be normalized.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return torch.nn.functional.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class FeedForward(torch.nn.Module):
    def __init__(self, n_dim_in, d_dim_out, dropout=0.0, bias=False):
        """
        Initialize the FeedForward module.

        Parameters:
            n_dim_in (int): The number of input dimensions.
            d_dim_out (int): The number of output dimensions.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            bias (bool, optional): Whether to include a bias term in the linear layers. 
                                   Defaults to False.
        """
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_dim_in, 4 * n_dim_in, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4 * n_dim_in, d_dim_out, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Pass the input through the feedforward network.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        return self.net(x)
