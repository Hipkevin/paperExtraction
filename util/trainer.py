from . import timer
from .config import Config4cls, Config4gen

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import Accuracy, Precision, Recall, F1, ROUGEScore, BLEUScore
from pprint import pprint


@timer
def train4cls(model, train_loader, val_loader, optimizer, criterion, config: Config4cls):
    model.train()

    val_f1_metric = F1(num_classes=config.num_classes, average='macro').to(config.device)

    for epoch in range(config.epoch_size):

        for idx, data in enumerate(train_loader):
            x, y = data[0].to(config.device), data[1].to(config.device)
            out = model(x)

            loss = criterion(input=out, target=y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation
            if idx % 10 == 0:
                for data in val_loader:
                    x, y = data[0].to(config.device), data[1].to(config.device)
                    out = model(x)

                    pre = out.softmax(dim=-1)
                    val_f1_metric(pre, y)

                val_f1 = val_f1_metric.compute()
                print(f'epoch: {epoch + 1} batch: {idx} loss: {loss} | f1: {val_f1}')
                val_f1_metric.reset()

    return model


@timer
def test4cls(test_loader, model, config):
    model.eval()

    acc_metric = Accuracy(num_classes=config.num_classes, average='macro').to(config.device)
    p_metric = Precision(num_classes=config.num_classes, average='macro').to(config.device)
    r_metric = Recall(num_classes=config.num_classes, average='macro').to(config.device)
    f1_metric = F1(num_classes=config.num_classes, average='macro').to(config.device)

    for data in test_loader:
        x, y = data[0].to(config.device), data[1].to(config.device)
        out = model(x)

        pre = out.softmax(dim=-1)

        acc_metric(pre, y)
        p_metric(pre, y)
        r_metric(pre, y)
        f1_metric(pre, y)

    acc = acc_metric.compute()
    p = p_metric.compute()
    r = r_metric.compute()
    f1 = f1_metric.compute()

    print(f'accuracy: {acc}\n'
          f'precision: {p}\n'
          f'recall: {r}\n'
          f'f1: {f1}\n')


@timer
def train4gen(model, train_loader, val_loader, optimizer, criterion, config: Config4gen):
    val_BLUE_metric = BLEUScore(n_gram=1).to(config.device)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_multi, eta_min=1e-6)

    for epoch in range(config.epoch_size):

        for idx, data in enumerate(train_loader):
            model.train()
            x, y, _ = data[0].to(config.device), data[1].to(config.device), data[2]
            logits, loss = model(x, y)

            loss = criterion(input=logits.flatten(end_dim=-2), target=y.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation
            if idx % 10 == 0:
                model.eval()
                for data in val_loader:
                    x, _, title_text = data[0].to(config.device), data[1].to(config.device), data[2]

                    summary_ids = model.PTM.generate(x,
                                                     num_beams=config.num_beams,
                                                     no_repeat_ngram_size=1,
                                                     min_length=0,
                                                     max_length=config.title_pad_size,
                                                     early_stopping=True)

                    pre = [p.replace(' ', '') for p in
                           model.tokenizer.batch_decode(summary_ids,
                                                        skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)]

                    val_BLUE_metric(pre, title_text)

                val_BLUE = val_BLUE_metric.compute()
                print(f'epoch: {epoch + 1} batch: {idx} loss: {loss} | BLUE-1: {val_BLUE}')
                val_BLUE_metric.reset()
        scheduler.step()

    return model


@timer
def test4gen(test_loader, model, config: Config4gen):
    model.eval()

    ROUGE_metric = ROUGEScore(rouge_keys='rougeL').to(config.device)
    BLUE1_metric = BLEUScore(n_gram=1).to(config.device)
    BLUE2_metric = BLEUScore(n_gram=2).to(config.device)
    BLUE3_metric = BLEUScore(n_gram=3).to(config.device)

    for data in test_loader:
        x, _, title_text = data[0].to(config.device), data[1].to(config.device), data[2]

        summary_ids = model.PTM.generate(x,
                                         num_beams=config.num_beams,
                                         no_repeat_ngram_size=1,
                                         min_length=0,
                                         max_length=config.title_pad_size,
                                         early_stopping=True)
        pre = [p.replace(' ', '') for p in
               model.tokenizer.batch_decode(summary_ids,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)]

        ROUGE_metric(pre, title_text)
        BLUE1_metric(pre, title_text)
        BLUE2_metric(pre, title_text)
        BLUE3_metric(pre, title_text)

    ROUGE = ROUGE_metric.compute()
    BLUE1 = BLUE1_metric.compute()
    BLUE2 = BLUE2_metric.compute()
    BLUE3 = BLUE3_metric.compute()

    pprint(ROUGE)
    print(f'BLUE-1: {BLUE1}')
    print(f'BLUE-2: {BLUE2}')
    print(f'BLUE-3: {BLUE3}')