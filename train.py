import imp
import torch
from model import *
from model.ssd import MultiboxLoss, PredConvNet
from utils.utils import mAP_score
TOTAL_PRIORS_NUM = 9128
TOTAL_NUM_OF_CLASSES = 91
clip_value = 0.9

grad_clip = None

lr_decay_scale = 0.1


def train(model: PredConvNet, train_loader, validation_loader, optimizer, criterion: MultiboxLoss, epochs, device):

    total_loss = []
    total_batch_loss = []
    total_accuracy = []

    model.to(device)

    for epoch in range(epochs):
        batch_loss = 0
        it = 0

        for images, bboxes, labels in train_loader:
            it += 1
            optimizer.zero_grad()
            loc_hat, conf_hat = model(images)

            loc_hat = loc_hat.view(len(bboxes), TOTAL_PRIORS_NUM, -1)
            conf_hat = conf_hat.view(len(bboxes), TOTAL_PRIORS_NUM, -1)

            loss = criterion(loc_hat, conf_hat, bboxes, labels)

            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            total_batch_loss.append(loss.item())
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()

        total_loss.append(batch_loss)

        path = f"model_time_steps/model_epoch{epoch}"

        PATH = open(path, 'wb')
        torch.save({
            'model_state_dict': model.state_dict(),
        }, PATH)
        PATH.close()

        print(
            f"Epoch {epoch} : loss = {total_loss[-1]}"
        )

    return total_loss


def get_items(y, item):
    labels = []
    for annot in y:
        labels.append(annot[item])

    return torch.tensor(labels)


#################################################################
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" %
          (optimizer.param_groups[1]['lr'],))
