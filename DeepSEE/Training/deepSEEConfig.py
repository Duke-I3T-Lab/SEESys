from transformers import PretrainedConfig

import torch

class DeepSEEConfig(PretrainedConfig):
    def __init__(
        self,
        epochs: int = 10,
        lr: float = 0.01,
        batchsize: int = 64,
        seed: int = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        data_dir: str = './data/',
        datasets: list = ['SenseTime'],
        n_proc: int = 8,
        ## Time Series Sliding Window
        window_length: int = 30,
        window_step: int = 10,
        target_transform: bool = False,
        **kwargs,
    ):

        ## General cross attention configuration
        self.epochs = epochs
        self.lr = lr
        self.batchsize = batchsize
        self.seed = seed
        self.device = device
        self.data_dir = data_dir
        self.datasets = datasets
        self.n_proc = n_proc
        ## Time Series Sliding Window
        self.window_length = window_length
        self.window_step = window_step
        ## Target Label Transform
        self.target_transform = target_transform
        super().__init__(**kwargs)