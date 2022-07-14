import torch
import random
import numpy as np

from torch.utils.data import DataLoader

from util.dataTool import StandardDataset4gen
from util.model import BartGeneration
from util.trainer import train4gen, test4gen

from util.config import Config4gen

# seed everything
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    config = Config4gen()
    config.seed = seed

    print('Data Loading...')
    model = BartGeneration(config).to(config.device)

    dataset = StandardDataset4gen('data/standard.xlsx', config)

    train_sampler, test_sampler = dataset.getSampler()  # 使用sampler划分数据集

    train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=test_sampler, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Training...")
    # model = train4gen(model,
    #                   train_loader=train_loader,
    #                   val_loader=test_loader,
    #                   optimizer=optimizer,
    #                   criterion=criterion,
    #                   config=config)

    print("Testing...")
    test4gen(test_loader, model, config)