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


class DecoderModel(L.LightningModule):
    def __init__(
            self, decoder_block,
            # Hyperparameters and Config
            n_layers, n_head, n_dim, max_seq_len, mlp_dropout, attn_dropout,
            vocab_size, learning_rate, min_learning_rate,
            weight_decay, beta1, beta2, bias=False, log_interval=1
    ):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=Tokenizer.pad_token_id
        )

        dec = decoder_block(
            n_head,
            n_dim,
            max_seq_len,
            mlp_dropout,
            attn_dropout,
            bias
        )

        self.decoder_layers = torch.nn.ModuleList(
            [copy.deepcopy(dec) for _ in range(n_layers)]
        )
        self.embedding = torch.nn.Embedding(vocab_size, n_dim)
        self.pos_encoder = PositionalEncoder(n_dim, max_seq_len)
        self.final_linear = torch.nn.Linear(n_dim, vocab_size)

    def forward(self, x, tgt_key_pad_mask, memory=None, memory_key_pad_mask=None):
        x = self.pos_encoder(self.embedding(x))

        for layer in self.decoder_layers:
            x = layer(x, tgt_key_pad_mask, memory, memory_key_pad_mask)

        logits = self.final_linear(x)
        return logits

    def training_step(self, batch):
        _, loss, _ = self._compute_and_log_metrics(batch, "train")
        return loss

    def validation_step(self, batch):
        _, loss, _ = self._compute_and_log_metrics(
            batch, "validation", on_step=False)
        return loss

    def test_step(self, batch):
        self._compute_and_log_metrics(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=10,
            min_lr=self.hparams.min_learning_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self.hparams.log_interval,
                "monitor": "validation_loss",
                "strict": True,
            }
        }

    def _compute_and_log_metrics(self, batch, prefix, on_step=True, on_epoch=True):
        logits = self(batch["inputs"], batch["attn_masks"])
        loss = self._compute_loss(logits, batch["targets"])
        acc = self._compute_accuracy(logits, batch["targets"])

        self.log_dict(
            {f"{prefix}_loss": loss, f"{prefix}_accuracy":  acc},
            on_step=on_step, on_epoch=on_epoch, logger=True
        )

        return logits, loss, acc

    def _compute_loss(self, logits, targets):
        return self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

    def _compute_accuracy(self, logits, targets):
        # Get the index of the maximum logit as the predicted token
        _, predicted = torch.max(logits, dim=-1)

        # Mask out padding positions
        non_padding_mask = (targets != Tokenizer.pad_token_id)
        total_non_padding = non_padding_mask.sum().item()

        correct_predictions = (
            predicted[non_padding_mask] == targets[non_padding_mask]
        ).sum().item()

        accuracy = correct_predictions / total_non_padding if total_non_padding > 0 else 0.0

        return accuracy

    def _generate(self, src, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = src[-self.hparams.max_seq_len:]

            logits = self(idx_cond, None)
            logits = logits[:, -1]

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence
            src = torch.cat((src, idx_next), dim=1)

            if idx_next[0][0] == Tokenizer.eos_token_id:
                break

        return src
