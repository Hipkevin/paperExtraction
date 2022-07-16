import torch
import random
import numpy as np

from torch.utils.data import DataLoader

from util.dataTool import StandardDataset4gen, StandardDataset4cls
from util.model import PTMGeneration
from util.trainer import train4gen, test4gen, train4cls, test4cls

from util.config import Config4gen, Config4cls

# seed everything
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def pretrain(model_encoder, config_cls: Config4cls):
    print('Data Loading...')
    dataset = StandardDataset4cls('data/standard.xlsx', config_cls)

    train_sampler, test_sampler = dataset.getSampler()  # 使用sampler划分数据集

    train_loader = DataLoader(dataset, batch_size=config_cls.batch_size, sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=config_cls.batch_size, sampler=test_sampler, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model_encoder.parameters(),
                                  lr=config_cls.learning_rate, weight_decay=config_cls.weight_decay)

    print("Training...")
    model_encoder = train4cls(model_encoder,
                              train_loader=train_loader,
                              val_loader=test_loader,
                              optimizer=optimizer,
                              criterion=criterion,
                              config=config_cls)

    print("Testing...")
    test4cls(test_loader, model_encoder, config_cls)

    return model_encoder

def train(model_all, config_gen: Config4gen):
    print('Data Loading...')
    dataset4gen = StandardDataset4gen('data/standard.xlsx', config_gen)
    print(f'dataset size: {len(dataset4gen)}')

    train_sampler, test_sampler = dataset4gen.getSampler()  # 使用sampler划分数据集

    train_loader = DataLoader(dataset4gen, batch_size=config_gen.batch_size, sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(dataset4gen, batch_size=config_gen.batch_size, sampler=test_sampler, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model_all.parameters(),
                                  lr=config_gen.learning_rate, weight_decay=config_gen.weight_decay)

    print("Training...")
    model_all = train4gen(model_all,
                      train_loader=train_loader,
                      val_loader=test_loader,
                      optimizer=optimizer,
                      criterion=criterion,
                      config=config_gen)

    print("Testing...")
    test4gen(test_loader, model_all, config_gen)

    return model_all


if __name__ == '__main__':
    config_g = Config4gen()
    config_g.seed = seed

    config_c = Config4cls()
    config_c.seed = seed

    model = PTMGeneration(config_g).to(config_g.device)

    print("----- Pre-train -----")
    model = pretrain(model, config_c)

    print("----- Train -----")
    model = train(model, config_g)