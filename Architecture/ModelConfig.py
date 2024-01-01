from dataclasses import dataclass, asdict
import datetime


@dataclass
class ModelConfig:
    random_seed = 1337

    # WandB Logging
    wandb_log = True
    wandb_project_name = 'CodeNet-Sentinel-v3'
    wandb_run_name = datetime.datetime.now().strftime("%b-%d-%y @ %I:%M %p")

    # The Data
    batch_size: int = 10
    max_seq_len: int = 768
    test_split: float = 0.1
    val_split: float = 0.1
    shuffle_dataset: bool = True

    # The Model
    n_layers: int = 6
    n_head: int = 8
    n_dim: int = 512
    mlp_dropout: float = 0.1
    attn_dropout: float = 0.0
    bias: bool = False

    # Training
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-8
    warmup_iters: int = 1000
    num_epochs: int = 7
    weight_decay = 0.01
    beta1 = 0.9
    beta2 = 0.98
    grad_accumulation = 8
    grad_clip = 1.0
    log_interval = 1000

    def asdict(self):
        return asdict(self)
