import torch
import re
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from typing import Tuple, List
from transformers import AutoTokenizer

from . import timer
from .config import Config4cls, Config4gen


@timer
def getStandard4cls(path) -> Tuple[List, List, List, List]:
    """
    获取带标注的标准数据，用于句子分类（语步划分）
    :param path: excel文件路径
    :return: (目的，方法，其他，标题)
    """

    data = pd.read_excel(path)

    titles = data['title']
    abstracts = data['abstract']

    # 规则提取
    purpose, method, other = list(), list(), list()
    for abstract in abstracts:
        abstract = re.split('\n|_x000D_', abstract)[0].split('。【')

        for a in abstract:
            if '【目的】' in a:
                purpose.append(a.strip('【目的】。'))

            elif '方法】' in a:
                method.append(a.strip('方法】。'))

            else:
                a = a.split('作者简介')[0]
                other.append(a.split('】')[1].strip(' 。'))

    return purpose, method, other, titles

@timer
def getStandard4gen(path) -> Tuple[List, List]:
    """
    提取摘要中的目的和方法，用于标题生成
    :param path: 文件路径
    :return: （目的+方法，标题）
    """
    data = pd.read_excel(path)

    titles = data['title']
    abstracts = data['abstract']

    # 规则提取
    contents = list()
    for abstract in abstracts:
        abstract = re.split('\n|_x000D_', abstract)[0].split('。【')

        content = ''
        for a in abstract:
            if '【目的】' in a:
                content += a.strip('【目的】。')
            elif '方法】' in a:
                content += a.strip('方法】。')

        contents.append(content)

    return contents, titles

class StandardDataset(Dataset):
    def __init__(self):
        super(StandardDataset, self).__init__()

    @staticmethod
    def buildSamplerIndex(X, Y, cv=0.15, random_seed=0):
        # 对数据集进行分层抽样，返回划分后的索引
        # 通过该索引构造SubsetRandomSampler，完成数据集划分
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=cv, random_state=random_seed)
        train_index, test_index = list(stratified_split.split(X, Y))[0]

        return train_index.tolist(), test_index.tolist()

    def getSampler(self):
        assert hasattr(self, 'train_index') and hasattr(self, 'test_index'),\
            print('necessary class member default: index')

        return SubsetRandomSampler(self.train_index), SubsetRandomSampler(self.test_index)


class StandardDataset4cls(StandardDataset):
    def __init__(self, path, config: Config4cls):
        super(StandardDataset4cls, self).__init__()

        purpose, method, other, _ = getStandard4cls(path)

        data = purpose + method + other
        label = [0] * len(purpose) + [1] * len(method) + [2] * len(other)

        print(f"purpose: {len(purpose)}\n"
              f"method: {len(method)}\n"
              f"other: {len(other)}")

        tokenizer = AutoTokenizer.from_pretrained(config.ptm_name, cache_dir=config.ptm_path)

        self.X = torch.tensor([tokenizer.encode(text=d,
                                                truncation=True,  # 截断
                                                padding='max_length',  # 填充到max_length
                                                max_length=config.pad_size,
                                                add_special_tokens=True)
                               for d in data], dtype=torch.long)
        self.Y = torch.tensor(label, dtype=torch.long)

        # 训练-测试集划分的采样器
        self.train_index, self.test_index = self.buildSamplerIndex(self.X, self.Y, config.cv, config.seed)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return len(self.X)

class StandardDataset4gen(StandardDataset):
    def __init__(self, path, config: Config4gen):
        super(StandardDataset4gen, self).__init__()

        contents, self.titles = getStandard4gen(path)

        tokenizer = AutoTokenizer.from_pretrained(config.ptm_name, cache_dir=config.ptm_path)

        self.X = torch.tensor([tokenizer.encode(text=content,
                                                truncation=True,  # 截断
                                                padding='max_length',  # 填充到max_length
                                                max_length=config.content_pad_size,
                                                add_special_tokens=True)
                               for content in contents], dtype=torch.long)

        self.Y = torch.tensor([tokenizer.encode(text=title,
                                                truncation=True,  # 截断
                                                padding='max_length',  # 填充到max_length
                                                max_length=config.title_pad_size,
                                                add_special_tokens=True)
                               for title in self.titles], dtype=torch.long)

        # 训练-测试集划分的采样器
        self.train_index, self.test_index = self.buildSamplerIndex(self.X, self.Y, config.cv, config.seed)

    @staticmethod
    def buildSamplerIndex(X, Y, cv=0.15, random_seed=0):
        # 对数据集进行随机抽样，返回划分后的索引
        # 通过该索引构造SubsetRandomSampler，完成数据集划分
        stratified_split = ShuffleSplit(n_splits=1, test_size=cv, random_state=random_seed)
        train_index, test_index = list(stratified_split.split(X))[0]

        return train_index.tolist(), test_index.tolist()

    def __getitem__(self, item):
        return self.X[item], self.Y[item], self.titles[item]

    def __len__(self):
        return len(self.X)