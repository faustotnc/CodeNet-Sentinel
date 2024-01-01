import lightning
import torch

from Architecture import PositionalEncoder, Tokenizer
from Architecture.Encoder import EncoderBlock, QnAConcatBlock
from torch.utils.data import Dataset


class SentinelDataset(Dataset):
    def __init__(self, data, status_values):
        self.question = torch.stack(data["question"])
        self.question_pad_mask = torch.stack(data["question_pad_mask"])
        self.answer = torch.stack(data["answer"])
        self.answer_pad_mask = torch.stack(data["answer_pad_mask"])
        self.status = torch.tensor(data["status"])

        self.status_values = status_values

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        return {
            "question": self.question[idx],
            "question_pad_mask": self.question_pad_mask[idx],
            "answer": self.answer[idx],
            "answer_pad_mask": self.answer_pad_mask[idx],
            "status": self.status[idx],
        }

    def save_to(self, filename):
        torch.save(self, filename)

    def load_from(filename) -> Dataset:
        return torch.load(filename)


class SentinelModel(lightning.LightningModule):
    def __init__(
        self, encoder_block: EncoderBlock, qna_concat_block: QnAConcatBlock,
        # Hyperparameters and Config
        n_layers: int, n_head: int, n_dim: int, max_seq_len: int, vocab_size: int, n_logits: int,
        mlp_dropout: float, learning_rate: float, min_learning_rate: float, weight_decay: float,
        beta1: float, beta2: float, max_iters: int, bias: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=Tokenizer.pad_token_id
        )

        # The Question Encoder
        self.question_embedding = torch.nn.Embedding(vocab_size, n_dim)
        self.question_pos_enc = PositionalEncoder(n_dim, max_seq_len)
        self.question_encoder_layers = torch.nn.ModuleList([
            encoder_block(n_head, n_dim, mlp_dropout, bias) for _ in range(n_layers)
        ])

        # The Answer Encoder
        self.answer_embedding = torch.nn.Embedding(vocab_size, n_dim)
        self.answer_pos_enc = PositionalEncoder(n_dim, max_seq_len)
        self.answer_encoder_layers = torch.nn.ModuleList([
            encoder_block(n_head, n_dim, mlp_dropout, bias) for _ in range(n_layers)
        ])

        # The QnA Concatenation Layer
        self.qna_concat_layer = qna_concat_block(n_head, n_dim, mlp_dropout, bias)

        # The autoregressive language model head
        self.classification_lm_head = torch.nn.Sequential(
            torch.nn.Linear(n_dim, n_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(mlp_dropout),
            torch.nn.Linear(n_dim, n_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(mlp_dropout),
            torch.nn.Linear(n_dim, n_logits, bias=bias)
        )

    def forward(self, question, question_pad_mask, answer, answer_pad_mask):
        # Encode the question
        question_embed = self.question_embedding(question)
        question_encoder_out = self.question_pos_enc(question_embed)
        for q_enc in self.question_encoder_layers:
            question_encoder_out = q_enc(question_encoder_out, question_pad_mask)

        # Encode the answer
        answer_embed = self.answer_embedding(answer)
        answer_encoder_out = self.answer_pos_enc(answer_embed)
        for a_enc in self.answer_encoder_layers:
            answer_encoder_out = a_enc(answer_encoder_out, answer_pad_mask)

        # Concatenate the question and answer
        qna_concat_out = self.qna_concat_layer(
            question_memory=question_encoder_out,
            answer_memory=answer_encoder_out,
            question_attn_mask=question_pad_mask,
            answer_key_pad_mask=answer_pad_mask
        )

        # Final pass through the classification language model head
        avg_output = qna_concat_out.mean(dim=1)
        logits = self.classification_lm_head(avg_output)
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
            batch["question"],
            batch["question_pad_mask"],
            batch["answer"],
            batch["answer_pad_mask"]
        )

        loss = self.criterion(logits, batch["status"])
        acc = self._compute_accuracy(logits, batch["status"])

        self.log_dict(
            {f"{prefix}_loss": loss, f"{prefix}_accuracy":  acc},
            on_step=on_step, on_epoch=on_epoch, logger=True
        )

        return logits, loss, acc

    def _compute_accuracy(self, logits, labels):
        # Get the index of the maximum logit as the predicted token
        _, predicted = torch.max(logits, dim=-1)

        correct = (predicted == labels).sum().item()
        total = labels.size(0)  # Total number of labels
        accuracy = 100 * correct / total

        return accuracy
