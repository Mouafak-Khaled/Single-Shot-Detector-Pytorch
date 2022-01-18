import enum
from tkinter import BOTTOM
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torch import relu
import numpy as np

DEFAULT_IMAGE_SIZE = 300
NUM_OF_CLASSES = 91

# A pretrained model trained with ImageNet:
base_model = models.resnet18(pretrained=True)

# Removing the Fully Connected Layer from the ResNet:
# base_model = nn.Sequential(*list(base_model.children())[:-1])

# Removing the Fully Connected Layer from the ResNet:
base_model = nn.Sequential(*list(base_model.children())[:-2])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

composed = transforms.Compose([transforms.Resize(DEFAULT_IMAGE_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])


class Conv(nn.Module):
    def __init__(self, channels, paddings, strides):
        super(Conv, self).__init__()

        kernel = 1
        self.conv1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                               kernel_size=kernel, stride=strides[0], padding=paddings[0])

        kernel = 3
        self.conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2],
                               kernel_size=kernel, stride=strides[1], padding=paddings[1])

        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)

    def forward(self, x):

        x = relu(self.conv1(x))
        x = relu(self.conv2(x))

        return x


class FeatureNet():
    def __init__(self, base_resnet):

        modelList = list(base_resnet.children())

        self.feature1 = nn.Sequential(*modelList[:6])  # 128 x 38 x 38

        self.feature2 = nn.Sequential(
            self.feature1,
            *modelList[6]
        )  # 256 x 19 x 19

        channels = [256, 256, 512]
        paddings = [0, 1]
        strides = [1, 2]

        self.feature3 = nn.Sequential(
            self.feature2,
            Conv(channels=channels, paddings=paddings, strides=strides)
        )  # 512 x 10 x 10

        channels = [512, 128, 256]
        paddings = [0, 1]
        strides = [1, 2]

        self.feature4 = nn.Sequential(
            self.feature3,
            Conv(channels=channels, paddings=paddings, strides=strides)
        )  # 256 x 5 x 5

        channels = [256, 128, 256]
        paddings = [0, 0]
        strides = [1, 1]

        self.feature5 = nn.Sequential(
            self.feature4,
            Conv(channels=channels, paddings=paddings, strides=strides)
        )  # 256 x 3 x 3

        channels = [256, 128, 256]
        paddings = [0, 0]
        strides = [1, 1]

        self.feature6 = nn.Sequential(
            self.feature5,
            Conv(channels=channels, paddings=paddings, strides=strides)
        )  # 256 x 1 x 1

        self.features = [self.feature1, self.feature2,
                         self.feature3, self.feature4,
                         self.feature5, self.feature6]


class PredConv(nn.Module):
    def __init__(self, channel_in, loc_channel_out, pred_channel_out, feature):
        super(PredConv, self).__init__()

        kernel, pad = 3, 1

        loc_conv_tmp = nn.Conv2d(in_channels=channel_in, out_channels=loc_channel_out,
                                 kernel_size=kernel, padding=pad)

        pred_conv_tmp = nn.Conv2d(in_channels=channel_in, out_channels=pred_channel_out,
                                  kernel_size=kernel, padding=pad)

        torch.nn.init.kaiming_uniform_(loc_conv_tmp.weight)
        torch.nn.init.kaiming_uniform_(pred_conv_tmp.weight)

        self.loc_conv = nn.Sequential(feature, loc_conv_tmp)
        self.pred_conv = nn.Sequential(feature, pred_conv_tmp)

    def forward(self, x):
        x_loc = self.loc_conv(x)
        x_pred = self.pred_conv(x)

        return x_loc, x_pred


class PredConvNet(nn.Module):
    def __init__(self, channel_in, loc_channel_out, pred_channel_out, features):
        super(PredConvNet, self).__init__()

        self.pred_convs = nn.ModuleList()

        for i in range(len(features)):
            self.pred_convs.append(PredConv(
                channel_in[i], loc_channel_out[i], pred_channel_out[i], features[i]))

    def forward(self, x):
        loc_v = torch.empty(0, 4)
        conf_v = torch.empty(0, NUM_OF_CLASSES)

        out_conv = [pred_conv(x) for pred_conv in self.pred_convs]

        for loc_out, conf_out in out_conv:
            loc_v = torch.vstack((loc_v, loc_out.view(-1, 4)))
            conf_v = torch.vstack((conf_v, conf_out.view(-1, NUM_OF_CLASSES)))
        return loc_v, conf_v


class MultiboxLoss(nn.Module):
    def __init__(self, threshold=0.5, alpha=1):
        super(MultiboxLoss, self).__init__()

        self.smooth_l1 = nn.SmoothL1Loss()  # For bboxes
        self.cross_entropy = nn.CrossEntropyLoss()  # For confs
        self.alpha = alpha

        self.threshold = threshold

    def forward(self, gt_locs, pred_locs, gt_ids, pred_confs):
        # --- Localization loss: -----------------------------------------------------

        LEFT, TOP, W, H = pred_locs[:, 0], pred_locs[:,
                                                     1], pred_locs[:, 2], pred_locs[:, 3]
        BOTTOM = LEFT + W
        RIGHT = TOP + H

        intersection = torch.empty((pred_locs.size()[0], 0))

        for gt_loc in gt_locs:

            gt_left, gt_top, w, h = gt_loc
            gt_right = gt_top + h
            gt_bottom = gt_left + w

            x_overlap = torch.maximum(torch.tensor([0.0]), torch.minimum(
                RIGHT, gt_right) - torch.maximum(LEFT, gt_left))
            y_overlap = torch.maximum(torch.tensor([0.0]), torch.minimum(
                BOTTOM, gt_bottom) - torch.maximum(TOP, gt_top))
            intersection_area = x_overlap * y_overlap

            intersection = torch.hstack(
                (intersection, intersection_area.reshape(-1, 1)))

        intersection_mask = intersection > self.threshold
        positives_mask = torch.any(intersection_mask, dim=1)

        positives = torch.sum(positives_mask)

        num_positives = positives.item()

        intersection = intersection * intersection_mask
        intersection_indecies = intersection.argmax(1)

        positive_preds = pred_locs[positives_mask]

        positive_gts = gt_locs[intersection_indecies][positives_mask]

        positive_grads = torch.empty(positive_preds.size())

        positive_grads[:, 0] = (positive_gts[:, 0] -
                                positive_preds[:, 0]) / positive_preds[:, 2]
        positive_grads[:, 1] = (positive_gts[:, 1] -
                                positive_preds[:, 1]) / positive_preds[:, 3]
        positive_grads[:, 2] = torch.log(
            positive_gts[:, 2] / positive_preds[:, 2])
        positive_grads[:, 3] = torch.log(
            positive_gts[:, 3] / positive_preds[:, 3])

        loss_loc = self.smooth_l1(positive_grads, positive_gts) / num_positives

        # --- Confidence loss: -----------------------------------------------------
        # scores of positive priors
        positive_confs = pred_confs[positives_mask]
        positive_gt_ids = gt_ids[positives_mask]

        # scores of (some) negative priors
        num_hard_negatives = 3 * num_positives

        intersection_sums = torch.sum(intersection, 1)

        negative_threshold = torch.sort(intersection_sums)[:num_hard_negatives]

        intersection_sums_mask = intersection_sums <= negative_threshold[0]

        negative_confs = pred_confs[intersection_sums_mask][:num_hard_negatives]

        negative_gt_ids = gt_ids[intersection_sums_mask][:num_hard_negatives]

        loss_conf = self.cross_entropy(positive_confs, positive_gt_ids) + \
            self.cross_entropy(negative_confs, negative_gt_ids)

        loss_val = loss_conf + self.alpha * loss_loc

        return loss_val


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
