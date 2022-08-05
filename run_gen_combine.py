import torch
import random
import numpy as np

from torch.utils.data import DataLoader

from util.dataTool import StandardDataset4gen, StandardDataset4gen_combine
from util.model import PTMGeneration, Seq2SeqModel, BartSeq2SeqModel, BartSeq2SeqModel_combine
from util.trainer import train4gen_combine, test4gen_combine

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
    model = BartSeq2SeqModel_combine(config).to(config.device)

    dataset_train = StandardDataset4gen_combine('data/standard_train.xlsx', config)
    dataset_test = StandardDataset4gen_combine('data/standard_test.xlsx', config)
    print(f'train size: {len(dataset_train)}')
    print(f'test size: {len(dataset_test)}')

    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Training...")
    model = train4gen_combine(model,
                      train_loader=train_loader,
                      val_loader=test_loader,
                      optimizer=optimizer,
                      criterion=criterion,
                      config=config)

    print("Testing...")
    test4gen_combine(test_loader, model, config)