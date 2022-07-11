import torch


class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_classes = 3
        self.pad_size = 50

        # uer/bart-base-chinese-cluecorpussmall
        self.ptm_name = 'bert-base-chinese'
        self.ptm_path = 'models'

        self.epoch_size = 10
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.weight_decay = 1e-4
        self.dropout = 0.5
        self.cv = 0.15

        self.seed = 42