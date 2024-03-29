import torch
import numpy as np

from .wvTool import Vocab, S2STokenizer


class Config4cls:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_classes = 3
        self.pad_size = 150

        # uer/bart-base-chinese-cluecorpussmall
        # bert-base-chinese
        self.ptm_name = 'uer/bart-base-chinese-cluecorpussmall'
        self.ptm_path = 'models'

        self.emb_size = 768
        self.emb = torch.tensor(np.load(f'wv_{self.emb_size}.npz')['embeddings'].astype('float32'))
        self.vocab = Vocab()
        self.vocab.load_from_pkl('vocab.pkl')
        self.tokenizer = S2STokenizer(self.vocab)

        self.epoch_size = 10
        self.batch_size = 64
        self.learning_rate = 2e-5
        self.weight_decay = 1e-4
        self.dropout = 0.5
        self.cv = 0.15

        self.num_beams = 6
        self.num_beam_groups = 3
        self.diversity_penalty = 0.5

        self.seed = 42


class Config4gen:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_classes = 3
        self.content_pad_size = 150
        self.title_pad_size = 30

        # uer/bart-base-chinese-cluecorpussmall
        # fnlp/bart-base-chinese
        # uer/t5-base-chinese-cluecorpussmall
        self.ptm_name = 'uer/bart-base-chinese-cluecorpussmall'
        self.ptm_path = 'models'

        self.emb_size = 768
        self.emb = torch.tensor(np.load(f'wv_{self.emb_size}.npz')['embeddings'].astype('float32'))
        self.vocab = Vocab()
        self.vocab.load_from_pkl('vocab.pkl')
        self.tokenizer = S2STokenizer(self.vocab)

        self.epoch_size = 10
        self.batch_size = 64
        self.learning_rate = 2e-5
        self.weight_decay = 1e-4
        self.dropout = 0.5
        self.cv = 0.15

        self.T_0 = 6
        self.T_multi = 2

        self.num_beams = 6
        self.num_beam_groups = 3
        self.diversity_penalty = 0.5
        self.seed = 42

        self.lamb = 5