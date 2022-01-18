import torch
from model import *
from model.ssd import MultiboxLoss, PredConvNet

TOTAL_PRIORS_NUM = 9128


def train(model: PredConvNet, train_loader, validation_loader, optimizer, criterion: MultiboxLoss, epochs, device):

    total_loss = []
    total_accuracy = []

    model.to(device)
    b_size = train_loader.batch_size

    for epoch in range(epochs):
        batch_loss = 0
        for images, bboxes, labels in train_loader:

            optimizer.zero_grad()
            loc_hat, conf_hat = model(images)

            loc_hat = loc_hat.view(b_size, TOTAL_PRIORS_NUM, -1)
            conf_hat = conf_hat.view(b_size, TOTAL_PRIORS_NUM, -1)

            loss = criterion(loc_hat, conf_hat, bboxes, labels)

            print("loss: ", loss.item())
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()

        total_loss.append(batch_loss)

        val_acc = 0
        for images, bboxes, labels in validation_loader:
            loc_hat, conf_hat = model(images)

            pred_labels_pos = torch.softmax(conf_hat, 1).argmax(1)  # 0 --> 90
            acc = torch.sum(labels == (pred_labels_pos + 1))
            val_acc += acc

        total_accuracy.append(
            val_acc / (len(validation_loader) * validation_loader.batch_size))

        print(
            f"Epoch {epoch} : loss = {total_loss[-1]} - validation accuracy = {total_accuracy[-1]}"
        )


def get_items(y, item):
    labels = []
    for annot in y:
        labels.append(annot[item])

    return torch.tensor(labels)
