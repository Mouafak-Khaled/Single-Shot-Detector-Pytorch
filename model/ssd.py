import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torch import relu

DEFAULT_IMAGE_SIZE = 300


# A pretrained model trained with ImageNet:
base_model = models.resnet50(pretrained=True)

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
                                kernel_size=kernel, stride=strides[0], pad=paddings[0])

        kernel = 3
        self.conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], 
                                kernel_size=kernel, stride=strides[1], pad=paddings[1])

        torch.nn.init.kaiming_uniform(self.conv1.weight)
        torch.nn.init.kaiming_uniform(self.conv2.weight)

    def forward(self, x):

        x = relu(self.conv1(x))
        x = relu(self.conv2(x))

        return x

class FeatureNet(nn.Module):
    def __init__(self, base_resnet):
        super(FeatureNet, self).__init__()

        modelList = list(base_resnet.children())

        self.feature1 = nn.Sequential(modelList[:5])
        self.feature2 = nn.Sequential(modelList[:7])

        channels = [256, 256, 512]
        paddings = [0, 1]
        strides = [1, 2]

        self.feature3 = Conv(channels=channels , paddings=paddings , strides=strides)

        channels = [512, 128, 256]
        paddings = [0, 1]
        strides = [1, 2]

        self.feature4 = Conv(channels=channels , paddings=paddings , strides=strides)

        channels = [256, 128, 256]
        paddings = [0, 0]
        strides = [1, 1]

        self.feature5 = Conv(channels=channels , paddings=paddings , strides=strides)

        channels = [256, 128, 256]
        paddings = [0, 0]
        strides = [1, 1]

        self.feature6 = Conv(channels=channels , paddings=paddings , strides=strides)
        
        self.features = [self.feature1, self.feature2, self.feature3, self.feature4, self.feature5, self.feature6] 

    def forward(self, x):
        return [feature(x) for feature in self.features]


class PredConv(nn.Module):
    def __init__(self, channel_in, loc_channel_out, pre_channel_out, feature):
        super(PredConv, self).__init__()

        self.feature = feature

        kernel, pad = 3, 1
        
        self.loc_conv = nn.Conv2d(in_channels=channel_in, out_channels=loc_channel_out, 
                                kernel_size=kernel, pad=pad)

        self.pred_conv = nn.Conv2d(in_channels=channel_in, out_channels=pre_channel_out, 
                                kernel_size=kernel, pad=pad)

        torch.nn.init.kaiming_uniform(self.loc_conv.weight)
        torch.nn.init.kaiming_uniform(self.pred_conv.weight)

    def forward(self, x):
        feature_out = self.feature(x)
        x_loc = self.loc_conv(feature_out)
        x_pred = self.pred_conv(feature_out)

        return x_loc, x_pred


class PredConvNet(nn.Module):
    def __init__(self, channel_in, loc_channel_out, pred_channel_out, features):
        super(PredConvNet, self).__init__()

        self.pred_conv1 = PredConv(channel_in[0], loc_channel_out[0], pred_channel_out[0] , features[0])  
        self.pred_conv2 = PredConv(channel_in[1], loc_channel_out[1], pred_channel_out[1] , features[1]) 
 
        self.pred_conv3 = PredConv(channel_in[2], loc_channel_out[2], pred_channel_out[2] , features[2]) 
        self.pred_conv4 = PredConv(channel_in[3], loc_channel_out[3], pred_channel_out[3] , features[3]) 
 
        self.pred_conv5 = PredConv(channel_in[4], loc_channel_out[4], pred_channel_out[4] , features[4]) 
        self.pred_conv6 = PredConv(channel_in[5], loc_channel_out[5], pred_channel_out[5] , features[5]) 

        self.pred_convs = [self.pred_conv1, self.pred_conv2, self.pred_conv3, self.pred_conv4, self.pred_conv5, self.pred_conv6]         

    def forward(self, x):
        return [pred_conv(x) for pred_conv in self.pred_convs]

# loss = nn.CrossEntropyLoss(conf)
# loss.item()

# loss2 = nn.Smooth(loc)
# loss2.item()

# general_loss = loss + loss2

# general_loss.backward()

# optimizer.step()

def main():

    pass