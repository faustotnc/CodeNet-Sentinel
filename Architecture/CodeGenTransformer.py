import copy
import lightning
import torch

from Architecture import PositionalEncoder, Tokenizer
from Architecture.Decoder import DecoderBlock
from Architecture.Encoder import EncoderBlock
from torch.utils.data import Dataset


class CodeGenDataset(Dataset):
    def __init__(self, data, vocab_size):
        self.instructions = torch.stack(data["instructions"])
        self.instructions_pad_masks = torch.stack(data["instructions_pad_masks"])
        self.responses = torch.stack(data["responses"])
        self.responses_pad_masks = torch.stack(data["responses_pad_masks"])
        self.targets = torch.stack(data["targets"])

        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        return {
            "instructions": self.instructions[idx],
            "instructions_pad_masks": self.instructions_pad_masks[idx],
            "responses": self.responses[idx],
            "responses_pad_masks": self.responses_pad_masks[idx],
            "targets": self.targets[idx],
        }

    def save_to(self, filename):
        torch.save(self, filename)

    def load_from(filename) -> Dataset:
        return torch.load(filename)


class CodeGenModel(lightning.LightningModule):
    def __init__(
        self, decoder_block: DecoderBlock, encoder_block: EncoderBlock,
        # Hyperparameters and Config
        n_layers: int, n_head: int, n_dim: int, max_instruct_len: int, max_seq_len: int, vocab_size: int,
        mlp_dropout: float, attn_dropout: float, learning_rate: float, min_learning_rate: float,
        weight_decay: float, beta1: float, beta2: float, max_iters: int, bias: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.criterion=torch.nn.CrossEntropyLoss(
            ignore_index=Tokenizer.pad_token_id
        )

        # The Encoder
        self.encoder_embedding = torch.nn.Embedding(vocab_size, n_dim)
        self.encoder_pos_encoder = PositionalEncoder(n_dim, max_seq_len)
        enc = encoder_block(n_head, n_dim, mlp_dropout, attn_dropout, bias)
        self.encoder_layers = torch.nn.ModuleList([copy.deepcopy(enc) for _ in range(n_layers)])

        # The Decoder
        self.decoder_embedding = torch.nn.Embedding(vocab_size, n_dim)
        self.decoder_pos_encoder = PositionalEncoder(n_dim, max_seq_len)
        dec = decoder_block(n_head, n_dim, max_seq_len, mlp_dropout, attn_dropout, bias)
        self.decoder_layers = torch.nn.ModuleList([copy.deepcopy(dec) for _ in range(n_layers)])

        # The autoregressive language model head
        self.autoregressive_lm_head = torch.nn.Sequential(
            torch.nn.Linear(n_dim, n_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(mlp_dropout),
            torch.nn.Linear(n_dim, n_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(mlp_dropout),
            torch.nn.Linear(n_dim, vocab_size, bias=bias),
        )

    def forward(self, instruction, instruction_pad_mask, response, response_pad_mask):
        # Encode the instruction
        encoder_embed = self.encoder_embedding(instruction)
        encoder_out = self.encoder_pos_encoder(encoder_embed)
        for encoder in self.encoder_layers:
            encoder_out = encoder(encoder_out, instruction_pad_mask)

        # Decode the response
        decoder_embed = self.decoder_embedding(response)
        decoder_out = self.decoder_pos_encoder(decoder_embed)
        for decoder in self.decoder_layers:
            decoder_out = decoder(decoder_out, response_pad_mask, encoder_out, instruction_pad_mask)

        # Final pass through the autoregressive language model head
        logits = self.autoregressive_lm_head(decoder_out)
        return logits

    def training_step(self, batch):
        _, loss, _ = self._compute_and_log_metrics(batch, "train")
        return loss

    def validation_step(self, batch):
        _, loss, _ = self._compute_and_log_metrics(
            batch=batch,
            prefix="validation",
            on_step=False
        )

        return loss

    def test_step(self, batch):
        self._compute_and_log_metrics(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,  # placeholder
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.max_iters,
            eta_min=self.hparams.min_learning_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "train_loss_step",
                "strict": True,
            }
        }

    def _compute_and_log_metrics(self, batch, prefix, on_step=True, on_epoch=True):
        logits = self(
            batch["instructions"],
            batch["instructions_pad_masks"],
            batch["responses"],
            batch["responses_pad_masks"]
        )

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

    def _generate(self, instruction, instruction_pad_mask, response_seed, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = response_seed[:, -self.hparams.max_seq_len:]

            logits = self(instruction, instruction_pad_mask, idx_cond, None)
            logits = logits[:, -1]

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence
            response_seed = torch.cat((response_seed, idx_next), dim=1)

            if idx_next[0][0] == Tokenizer.eos_token_id:
                break

        return response_seed
