import torch
from dataclasses import dataclass

@dataclass
class ModelParams:
    """Model parameters"""
    TEST: bool = True

    sunshine_path: str = 'Datasets/sunshine.csv'
    temp_path: str = 'Datasets/temp.csv'
    wind_path: str = 'Datasets/wind.csv'

    model_path: str = None  # 'last_model.pth'
    model_path: str = 'last_model.pth'

    # Data parameters
    # 时间范围 （天）(前闭后开)
    train_day_range: tuple = (0, 300)
    val_day_range: tuple = (290, 300)
    infer_day_range: tuple = (300, 310)

    look_back_len: int = 5  # 可读取的历史信息长度（天）

    # Training parameters
    model_name: str = 'UNiLAB3'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    mode: str = 'CNN'
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-4
    max_grad_norm: float = 0.2

