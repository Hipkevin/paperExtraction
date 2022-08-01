import jieba
import gensim
import torch
import pickle as pkl
import numpy as np

from typing import Union, List, Optional, Tuple, Dict
from transformers import PreTrainedTokenizer


class Vocab:
    def __init__(self):
        super(Vocab, self).__init__()

        UNK, PAD, MASK = '[UNK]', '[PAD]', '[MASK]'
        vocab_dic = {PAD: 0, UNK: 1, MASK: 2}

        self.token2id = vocab_dic
        self.id2token = {0: PAD, 1: UNK, 2: MASK}

    def __getitem__(self, item: Union[int, str]) -> Union[int, str]:
        if isinstance(item, str):
            return self.token2id.get(item)
        else:
            return self.id2token.get(item)

    def __len__(self):
        return len(self.token2id)

    def add_from_corpus(self, file_path, min_freq, user_dict=None):
        if user_dict is not None:
            jieba.load_userdict(user_dict)

        vocab = {}
        with open(file_path, 'r', encoding='UTF-8') as file:

            for line in file.read().strip().split('\n'):
                content = jieba.cut(line)
                for word in content:
                    vocab[word] = vocab.get(word, 0) + 1

            vocab_list = [_ for _ in vocab.items() if _[1] >= min_freq]

            new_vocab = {word_count[0]: idx + 3 for idx, word_count in enumerate(vocab_list)}
            self.token2id.update(new_vocab)
            self.id2token.update(dict(zip(new_vocab.values(), new_vocab.keys())))

    def load_from_pkl(self, file_path):
        self.token2id = pkl.load(open(file_path, 'rb'))
        self.id2token.update(dict(zip(self.token2id.values(), self.token2id.keys())))

    def save_to_pkl(self, path):
        pkl.dump(self.token2id, open(path, 'wb'))

    def add_special_token(self, new_tokens: Union[str, List[str]]):
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]

        length = len(self.token2id)
        new_vocab = {word: idx + length for idx, word in enumerate(new_tokens)}

        self.token2id.update(new_vocab)
        self.id2token.update(dict(zip(new_vocab.values(), new_vocab.keys())))


class S2STokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Vocab):
        super(S2STokenizer, self).__init__()

        self.vocab = vocab
        self.add_special_tokens({'pad_token': vocab[0],
                                 'unk_token': vocab[1],
                                 'mask_token': vocab[2]})
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text, **kwargs):
        return list(jieba.cut(text))

    def _convert_token_to_id(self, token):
        return self.vocab[token] if token in self.vocab.token2id else 1

    def _convert_id_to_token(self, index: int) -> str:
        return self.vocab[index]

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.token2id

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        pass


class WVHandle:
    def __init__(self, vocab):
        super(WVHandle, self).__init__()

        self.vocab = vocab
        self.embedding = None

    def train_from_corpus(self, corpus_path, wv_dim, save_path, seed):
        with open(corpus_path, 'r', encoding='utf8') as file:
            content = file.read().strip().split('\n')

        text = [list(jieba.cut(c)) for c in content]
        wv_model = gensim.models.word2vec.Word2Vec(text, vector_size=wv_dim, min_count=2, seed=seed)

        embeddings = np.zeros((len(self.vocab), wv_dim))

        for word in self.vocab.token2id.keys():
            if word in wv_model.wv:
                idx = self.vocab[word]
                embeddings[idx] = np.asarray(wv_model.wv.get_vector(word), dtype='float32')

        np.savez_compressed(save_path, embeddings=embeddings)

        self.embedding = torch.tensor(embeddings.astype('float32'))
