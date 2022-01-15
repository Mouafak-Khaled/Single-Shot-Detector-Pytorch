import enum
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torch import relu

DEFAULT_IMAGE_SIZE = 300


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

        self.pred_convs = []
        for i in range(len(features)):
            self.pred_convs.append(PredConv(
                channel_in[i], loc_channel_out[i], pred_channel_out[i], features[i]))

    def forward(self, x):
        out_conv = [pred_conv(x) for pred_conv in self.pred_convs]
        return out_conv

# loss = nn.CrossEntropyLoss(conf)
# loss.item()

# loss2 = nn.Smooth(loc)
# loss2.item()

# general_loss = loss + loss2

# general_loss.backward()

# optimizer.step()


def main():

    channels_in = [128, 256, 512, 256, 256, 256]

    priors = [4, 7, 6, 7, 4, 4]

    NUM_OF_CLASSES = 91

    loc_channel_out = [4 * prior for prior in priors]
    pred_channel_out = [NUM_OF_CLASSES * prior for prior in priors]

    feature_net = FeatureNet(base_model)

    x = torch.randn((1, 3, 300, 300))
    # out = feature_net(x)

    pred_convnet = PredConvNet(
        channels_in, loc_channel_out, pred_channel_out, feature_net.features)


if __name__ == "__main__":
    main()
