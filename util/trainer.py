from . import timer
from .config import Config4cls, Config4gen

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from sacrebleu.metrics import BLEU
from torchmetrics import Accuracy, Precision, Recall, F1, ROUGEScore, BLEUScore
from pprint import pprint

import numpy as np
import jieba


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
    scheduler = StepLR(optimizer, step_size=500, gamma=0.95)

    for epoch in range(config.epoch_size):

        for idx, data in enumerate(train_loader):
            model.train()
            x, y, _ = data[0].to(config.device), data[1].to(config.device), data[2]

            #             pre = model.tokenizer.batch_decode(x,
            #                                                 skip_special_tokens=True,
            #                                                 clean_up_tokenization_spaces=True)

            #             print(pre)

            logits, loss = model(x, y)

            loss = criterion(input=logits.flatten(end_dim=-2), target=y.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if idx % 10 == 0:
                print(f'epoch: {epoch + 1} batch: {idx} loss: {loss}')

    return model


@timer
def test4gen(test_loader, model, config: Config4gen):
    model.eval()
    BLEU_metric = BLEU(effective_order=True)

    scores = np.zeros(4)
    bp = 0
    count = 0

    for data in test_loader:
        x, y, title_text = data[0].to(config.device), data[1].to(config.device), data[2]

        summary_ids = model.PTM.generate(x,
                                         num_beams=config.num_beams,
                                         no_repeat_ngram_size=1,
                                         num_beam_groups=config.num_beam_groups,
                                         diversity_penalty=config.diversity_penalty,
                                         length_penalty=0.7,
                                         min_length=0,
                                         max_length=config.title_pad_size,
                                         early_stopping=True)

        pre = model.tokenizer.batch_decode(summary_ids,
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)
        for p, t in zip(pre, title_text):
            s = BLEU_metric.sentence_score(p, [' '.join(jieba.cut(t))])
            scores += s.precisions
            bp += s.score
            count += 1

    for p, t in zip(pre[0: 5], title_text[0: 5]):
        print('gen:', p.replace(' ', ''))
        print('tgt:', t)

    print(f'BP_-BLEU: {bp / count}')
    print(f'BLEU-1/2/3/4: {scores / count}')


@timer
def train4gen_combine(model, train_loader, val_loader, optimizer, criterion, config: Config4gen):
    scheduler = StepLR(optimizer, step_size=500, gamma=0.95)

    for epoch in range(config.epoch_size):

        for idx, data in enumerate(train_loader):
            model.train()
            x, y, t, s = data[0].to(config.device), data[1].to(config.device), data[2], data[3].to(config.device)

            logits, loss, e_out = model(x, y)

            gen_loss = criterion(input=logits.flatten(end_dim=-2), target=y.flatten())
            cls_loss = config.lamb * criterion(input=e_out.flatten(end_dim=-2), target=s.flatten())

            loss = gen_loss + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if idx % 10 == 0:
                print(f'epoch: {epoch + 1} batch: {idx} loss: {loss} gen_loss:{gen_loss} cls_loss:{cls_loss}')

    return model


@timer
def test4gen_combine(test_loader, model, config: Config4gen):
    model.eval()
    BLEU_metric = BLEU(effective_order=True)
    f1_metric = F1(num_classes=config.num_classes + 1, average='macro').to(config.device)

    scores = np.zeros(4)
    bp = 0
    count = 0

    for data in test_loader:
        x, y, title_text, s = data[0].to(config.device), data[1].to(config.device), data[2], data[3].to(config.device)

        logits, loss, e_out = model(x, y)
        # print(e_out.softmax(dim=-1).flatten(end_dim=-2).size())
        f1_metric(e_out.softmax(dim=-1).flatten(end_dim=-2), s.flatten())

        summary_ids = model.PTM.generate(x,
                                         num_beams=config.num_beams,
                                         no_repeat_ngram_size=1,
                                         num_beam_groups=config.num_beam_groups,
                                         diversity_penalty=config.diversity_penalty,
                                         length_penalty=0.7,
                                         min_length=0,
                                         max_length=config.title_pad_size,
                                         early_stopping=True)

        pre = model.tokenizer.batch_decode(summary_ids,
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)
        for p, t in zip(pre, title_text):
            s = BLEU_metric.sentence_score(p, [' '.join(jieba.cut(t))])
            scores += s.precisions
            bp += s.score
            count += 1

    for p, t in zip(pre[0: 5], title_text[0: 5]):
        print('gen:', p.replace(' ', ''))
        print('tgt:', t)

    print(f'BP-BLEU: {bp / count}')
    print(f'BLEU-1/2/3/4: {scores / count}')

    f1 = f1_metric.compute()
    print(f'f1: {f1}')