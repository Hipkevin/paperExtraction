from . import timer
from .config import Config4cls, Config4gen

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

    val_ROUGE_metric = ROUGEScore(rouge_keys='rougeL')

    for epoch in range(config.epoch_size):

        for idx, data in enumerate(train_loader):
            model.train()
            x, y, _ = data[0].to(config.device), data[1].to(config.device), data[2]
            out = model(x, y)

            loss = criterion(input=out.flatten(end_dim=-2), target=y.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation
            if idx % 10 == 0:
                model.eval()
                for data in val_loader:
                    x, _, title_text = data[0].to(config.device), data[1].to(config.device), data[2]

                    summary_ids = model.bart.generate(x,
                                                      num_beams=2,
                                                      no_repeat_ngram_size=3,
                                                      length_penalty=2.0,
                                                      min_length=0,
                                                      max_length=config.title_pad_size,
                                                      early_stopping=True)

                    pre = [p.replace(' ', '') for p in
                           model.tokenizer.batch_decode(summary_ids,
                                                        skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)]

                    val_ROUGE_metric(pre, title_text)

                val_ROUGE = val_ROUGE_metric.compute()['rougeL_fmeasure']
                print(f'epoch: {epoch + 1} batch: {idx} loss: {loss} | ROUGE: {val_ROUGE}')
                val_ROUGE_metric.reset()

    return model

@timer
def test4gen(test_loader, model, config: Config4gen):
    model.eval()

    ROUGE_metric = ROUGEScore(rouge_keys='rougeL').to(config.device)
    BLUE_metric = BLEUScore(n_gram=4).to(config.device)

    for data in test_loader:
        x, _, title_text = data[0].to(config.device), data[1].to(config.device), data[2]

        summary_ids = model.bart.generate(x,
                                          num_beams=2,
                                          no_repeat_ngram_size=3,
                                          length_penalty=2.0,
                                          min_length=0,
                                          max_length=config.title_pad_size,
                                          early_stopping=True)
        pre = [p.replace(' ', '') for p in
                           model.tokenizer.batch_decode(summary_ids,
                                                        skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)]

        ROUGE_metric(pre, title_text)
        BLUE_metric(pre, title_text)

    ROUGE = ROUGE_metric.compute()
    BLUE = BLUE_metric.compute()

    pprint(ROUGE)
    print(f'BLUE: {BLUE}')