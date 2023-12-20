from dataclasses import dataclass, asdict
import datetime


@dataclass
class ModelConfig:
    random_seed = 1337

    # WandB Logging
    wandb_log = True
    wandb_project_name = 'CodeNet-Sentinel-v1'
    wandb_run_name = datetime.datetime.now().strftime("%b-%d-%y @ %I:%M %p")

    # The Data
    batch_size: int = 10
    max_instruction_len: int = 768
    max_seq_len: int = 768
    overlap_size: int = 8 # Number of overlapped tokens in tokenization
    val_split: float = 0.1
    shuffle_dataset: bool = True
    save_data_to: str = None  # Specified in notebook
    load_data_from: str = None  # Specified in notebook
    train_data_file_name: str = "sentinel_train.pt"
    val_data_file_name: str = "sentinel_val.pt",
    tokenized_stride: int = 128

    # The Model
    n_layers: int = 5
    n_head: int = 8
    n_dim: int = 320
    mlp_dropout: float = 0.1
    attn_dropout: float = 0.0
    bias: bool = False

    # n_layer: int = 2
    # n_head: int = 2
    # n_dim: int = 128
    # dropout: float = 0
    # attn_dropout: float = 0.0
    # bias: bool = False

    # Training
    learning_rate: float = 0.001
    min_learning_rate: float = 1e-5
    warmup_iters: int = 1000
    num_epochs: int = 10
    weight_decay = 0.01
    beta1 = 0.9
    beta2 = 0.98
    grad_accumulation = 8
    grad_clip = 1.0
    log_interval = 2000

    def asdict(self):
        return asdict(self)