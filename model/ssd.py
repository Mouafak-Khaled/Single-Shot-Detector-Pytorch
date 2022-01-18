from pyexpat import features
from tkinter import BOTTOM
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch import relu
import numpy as np
from utils.prior_utils import *
from utils.priors import *

DEFAULT_IMAGE_SIZE = 300
NUM_OF_CLASSES = 91

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# A pretrained model trained with ImageNet:
base_model = models.resnet18(pretrained=True)

# Removing the Fully Connected Layer from the ResNet:
# base_model = nn.Sequential(*list(base_model.children())[:-1])

# Removing the Fully Connected Layer from the ResNet:
base_model = nn.Sequential(*list(base_model.children())[:-2])


class Conv(nn.Module):
    def __init__(self, channels, paddings, strides):
        super(Conv, self).__init__()

        kernel = 1
        self.conv1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                               kernel_size=kernel, stride=strides[0], padding=paddings[0])

        kernel = 3
        self.conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2],
                               kernel_size=kernel, stride=strides[1], padding=paddings[1])

        self.bn1 = nn.BatchNorm2d(channels[1])
        self.bn2 = nn.BatchNorm2d(channels[2])

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

        nn.init.constant_(self.conv1.bias, 0.)
        nn.init.constant_(self.conv2.bias, 0.)

    def forward(self, x):
        x = self.bn1(relu(self.conv1(x)))
        x = self.bn2(relu(self.conv2(x)))

        return x


class FeatureNet(nn.Module):
    def __init__(self, base_resnet):
        super(FeatureNet, self).__init__()

        modelList = list(base_resnet.children())

        self.features = nn.ModuleList()

        self.feature1 = nn.Sequential(
            *modelList[:6],
            # nn.BatchNorm2d(128)
        )  # 128 x 38 x 38

        self.features.append(self.feature1)

        self.feature2 = nn.Sequential(
            self.feature1,
            *modelList[6],
            # nn.BatchNorm2d(256)
        )  # 256 x 19 x 19
        self.features.append(self.feature2)

        channels = [256, 256, 512]
        paddings = [0, 1]
        strides = [1, 2]

        self.feature3 = nn.Sequential(
            self.feature2,
            Conv(channels=channels, paddings=paddings, strides=strides),
            # nn.BatchNorm2d(512)
        )  # 512 x 10 x 10
        self.features.append(self.feature3)

        channels = [512, 128, 256]
        paddings = [0, 1]
        strides = [1, 2]

        self.feature4 = nn.Sequential(
            self.feature3,
            Conv(channels=channels, paddings=paddings, strides=strides),
            # nn.BatchNorm2d(256)
        )  # 256 x 5 x 5
        self.features.append(self.feature4)

        channels = [256, 128, 256]
        paddings = [0, 0]
        strides = [1, 1]

        self.feature5 = nn.Sequential(
            self.feature4,
            Conv(channels=channels, paddings=paddings, strides=strides),
            # nn.BatchNorm2d(256)
        )  # 256 x 3 x 3
        self.features.append(self.feature5)

        channels = [256, 128, 256]
        paddings = [0, 0]
        strides = [1, 1]

        self.feature6 = nn.Sequential(
            self.feature5,
            Conv(channels=channels, paddings=paddings, strides=strides),
            # nn.BatchNorm2d(256)
        )  # 256 x 1 x 1
        self.features.append(self.feature6)

    def forward(self, x):
        return [pred_conv(x) for pred_conv in self.features]


class PredConv(nn.Module):
    def __init__(self, channel_in, loc_channel_out, pred_channel_out, feature):
        super(PredConv, self).__init__()

        kernel, pad = 3, 1

        self.feature = feature

        self.loc_conv = nn.Conv2d(in_channels=channel_in, out_channels=loc_channel_out,
                                  kernel_size=kernel, padding=pad)

        self.pred_conv = nn.Conv2d(in_channels=channel_in, out_channels=pred_channel_out,
                                   kernel_size=kernel, padding=pad)

        nn.init.kaiming_normal_(self.loc_conv.weight)
        nn.init.constant_(self.loc_conv.bias, 0.)

        nn.init.kaiming_normal_(self.pred_conv.weight)
        nn.init.constant_(self.pred_conv.bias, 0.)

    def forward(self, x):
        x1 = self.feature(x)
        x_loc = self.loc_conv(x1)
        x_pred = self.pred_conv(x1)

        return x_loc, x_pred


class PredConvNet(nn.Module):
    def __init__(self, channel_in, loc_channel_out, pred_channel_out, featureNet):
        super(PredConvNet, self).__init__()

        self.pred_convs = nn.ModuleList()

        for i, feature in enumerate(featureNet):
            self.pred_convs.append(
                PredConv(channel_in[i], loc_channel_out[i], pred_channel_out[i], feature))

    def forward(self, x):
        loc_v = torch.empty(0, 4).to(device)
        conf_v = torch.empty(0, NUM_OF_CLASSES).to(device)

        out_conv = [pred_conv(x) for pred_conv in self.pred_convs]

        for loc_out, conf_out in out_conv:
            loc_v = torch.vstack((loc_v, loc_out.view(-1, 4))).to(device)
            conf_v = torch.vstack(
                (conf_v, conf_out.view(-1, NUM_OF_CLASSES))).to(device)
        return loc_v, conf_v


class MultiboxLoss(nn.Module):
    def __init__(self, n_priors, n_classes, threshold=0.5, alpha=1):
        super(MultiboxLoss, self).__init__()

        self.smooth_l1 = nn.SmoothL1Loss()  # For bboxes
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')  # For confs
        self.alpha = alpha
        self.n_priors = n_priors
        self.n_classes = n_classes
        self.priors = create_priors()
        self.priors_xy = cxcy_to_xy(self.priors)

        self.threshold = threshold

    ####The below unction (forward) is incorperated from the code of https://github.com/sgrvinod########
    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.n_priors
        n_classes = self.n_classes

        true_locs = torch.zeros((batch_size, n_priors, 4),
                                dtype=torch.float).to(device)
        true_classes = torch.zeros(
            (batch_size, n_priors), dtype=torch.long).to(device)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)

            overlap_for_each_prior, object_for_each_prior = overlap.max(
                dim=0)

            _, prior_for_each_object = overlap.max(dim=1)

            object_for_each_prior[prior_for_each_object] = torch.LongTensor(
                range(n_objects))

            overlap_for_each_prior[prior_for_each_object] = 1.

            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[overlap_for_each_prior <
                                 self.threshold] = 0

            true_classes[i] = label_for_each_prior

            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(
                boxes[i][object_for_each_prior]), self.priors)

        positive_priors = true_classes != 0
        predicted_locs_pos = predicted_locs[positive_priors]
        true_locs_pos = true_locs[positive_priors]

        nan_mask1 = predicted_locs_pos != predicted_locs_pos
        nan_mask2 = true_locs_pos != true_locs_pos

        predicted_locs_pos[nan_mask1] = 0
        true_locs_pos[nan_mask2] = 0

        loc_loss = self.smooth_l1(predicted_locs_pos, true_locs_pos)

        n_positives = positive_priors.sum(dim=1)
        n_hard_negatives = 3 * n_positives

        conf_loss_all = self.cross_entropy(
            predicted_scores.view(-1, n_classes), true_classes.view(-1))

        conf_loss_all = conf_loss_all.view(batch_size, n_priors)

        conf_loss_pos = conf_loss_all[positive_priors]

        conf_loss_neg = conf_loss_all.clone()

        conf_loss_neg[positive_priors] = 0.

        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(
            0).expand_as(conf_loss_neg).to(device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()
                     ) / n_positives.sum().float()
        return conf_loss + self.alpha * loc_loss


def main():

    # channels_in = [128, 256, 512, 256, 256, 256]

    # priors = [4, 7, 6, 7, 5, 5]

    # loc_channel_out = [4 * prior for prior in priors]
    # pred_channel_out = [NUM_OF_CLASSES * prior for prior in priors]

    # feature_net = FeatureNet(base_model)

    # x = torch.randn((1, 3, 300, 300))

    # pred_convnet = PredConvNet(
    #     channels_in, loc_channel_out, pred_channel_out, feature_net.features)

    # i, j =  pred_convnet(x)
    # print(j.size())

    gt_locs = np.abs(np.random.random((5, 4)))
    pred_locs = np.abs(np.random.random((25, 4)))

    gt_ids = torch.tensor([11, 3231, 753, 423, 847])

    pred_locs = torch.from_numpy(pred_locs)
    gt_locs = torch.from_numpy(gt_locs)

    pred_confs = torch.randn((25, 5))


if __name__ == "__main__":
    main()
