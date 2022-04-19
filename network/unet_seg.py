import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(residualBlock, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False))
        else:
          self.skip = None

        self.block = nn.Sequential(nn.BatchNorm2d(in_channels, track_running_stats=False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                   nn.BatchNorm2d(out_channels, track_running_stats=False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 3, padding=1)
                                   )   

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, c=4, sigmod=True):
        super(UNet, self).__init__()
        
        self.c = c
        self.sigmoid = sigmod
        
        size = self.c * np.array([2,4,8,16], dtype = np.intc)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_down1 = residualBlock(in_channels, size[0])
        self.dconv_down2 = residualBlock(size[0], size[1])
        self.dconv_down3 = residualBlock(size[1], size[2])
        self.dconv_down4 = residualBlock(size[2], size[3])
        
        self.bottleneck = residualBlock(size[3], size[3])
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up4 = residualBlock(2 * size[3] , size[3])
        self.dconv_up3 = residualBlock(size[3] + size[2], size[2])
        self.dconv_up2 = residualBlock(size[2] + size[1], size[1])
        self.dconv_up1 = residualBlock(size[1] + size[0], size[0])
        self.conv_last = nn.Conv2d(size[0], n_classes, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        x = self.bottleneck(x)
        
        x = self.upsample(x)
        x = torch.cat((x, conv4), dim=1)
        x = self.dconv_up4(x)
        
        x = self.upsample(x) 
        x = torch.cat((x, conv3), dim=1)
        x = self.dconv_up3(x)
      
        x = self.upsample(x)
        x = torch.cat((x, conv2), dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample(x)
        x = torch.cat((x, conv1), dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        if self.sigmoid:
          out = torch.sigmoid(out)
        
        return out