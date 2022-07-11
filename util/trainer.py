from . import timer

from torchmetrics import Accuracy, Precision, Recall, F1

@timer
def train(model, train_loader, val_loader, optimizer, criterion, config):
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
def test(test_loader, model, config):
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