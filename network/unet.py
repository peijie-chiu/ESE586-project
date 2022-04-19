import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        padding= 1,
        bias = True,
        act = 'lrelu',
        norm = "bnorm"):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]

        if norm == 'bnorm':
            layers.append(nn.BatchNorm2d(out_channels))

        if act =='lrelu':
            layers.append(nn.LeakyReLU(inplace=True))
        elif act == 'relu':
            layers.append(nn.ReLU(inplace=True))

        self.convblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.convblock(x)


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1))

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        # input dim x = Bx1x255x255
        assert x.size(-2) % 2 == 1 and x.size(-1) % 2 == 1
        x = F.pad(x, (0, 1, 0, 1), "constant", -0.5)
       
        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1).squeeze(1)[:, :-1, :-1]


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


if __name__ == '__main__':
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(4, 1, 255, 255)

    print(model(x).size())

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         # Encoder Part
#         self.enc1_1 = ConvBlock(1, 48, 3, 1, 1, bias=True, norm=None)
#         self.enc1_2 = ConvBlock(48, 48, 3, 1, 1, bias=True)
#         self.enc2 = ConvBlock(48, 48, 3, 1, 1, bias=True)
#         self.enc3 = ConvBlock(48, 48, 3, 1, 1, bias=True)
#         self.enc4 = ConvBlock(48, 48, 3, 1, 1, bias=True)
#         self.enc5 = ConvBlock(48, 48, 3, 1, 1, bias=True)
#         self.enc6 = ConvBlock(48, 48, 3, 1, 1, bias=True)

#         # Decoder Part
#         self.dec1_1 = ConvBlock(96, 96, 3, 1, 1, bias=True)
#         self.dec1_2 = ConvBlock(96, 96, 3, 1, 1, bias=True)

#         self.dec2_1 = ConvBlock(96+48, 96, 3, 1, 1, bias=True)
#         self.dec2_2 = ConvBlock(96, 96, 3, 1, 1, bias=True)

#         self.dec3_1 = ConvBlock(96+48, 96, 3, 1, 1, bias=True)
#         self.dec3_2 = ConvBlock(96, 96, 3, 1, 1, bias=True)

#         self.dec4_1 = ConvBlock(96+48, 96, 3, 1, 1, bias=True)
#         self.dec4_2 = ConvBlock(96, 96, 3, 1, 1, bias=True)

#         self.dec5_1 = ConvBlock(96+1, 64, 3, 1, 1, bias=True)
#         self.dec5_2 = ConvBlock(64, 32, 3, 1, 1, bias=True)

#         self.conv_out = nn.Conv2d(32, 1, 3, 1, 1, bias=True)

        
#     def forward(self, x):
#         # input dim x = Bx1x255x255
#         assert x.size(-2) % 2 == 1 and x.size(-1) % 2 == 1
#         x = F.pad(x, (0, 1, 0, 1))

#         input = x

#         # Encoder
#         x = self.enc1_1(x)
#         x = self.enc1_2(x)
#         x = self.maxpool(x)
#         pool1 = x

#         x = self.enc2(x)
#         x = self.maxpool(x)
#         pool2 = x

#         x = self.enc3(x)
#         x = self.maxpool(x)
#         pool3 = x

#         x = self.enc4(x)
#         x = self.maxpool(x)
#         pool4 = x

#         x = self.enc5(x)
#         x = self.maxpool(x)

#         x = self.enc6(x)

#         # Decoder
#         x = self.upsample(x)
#         x = torch.cat([x, pool4], axis=1)
#         x = self.dec1_1(x)
#         x = self.dec1_2(x)

#         x = self.upsample(x)
#         x = torch.cat([x, pool3], axis=1)
#         x = self.dec2_1(x)
#         x = self.dec2_2(x)

#         x = self.upsample(x)
#         x = torch.cat([x, pool2], axis=1)
#         x = self.dec3_1(x)
#         x = self.dec3_2(x)

#         x = self.upsample(x)
#         x = torch.cat([x, pool1], axis=1)
#         x = self.dec4_1(x)
#         x = self.dec4_2(x)

#         x = self.upsample(x)
#         x = torch.cat([x, input], axis=1)
#         x = self.dec5_1(x)
#         x = self.dec5_2(x)

#         x = F.leaky_relu(self.conv_out(x))

#         return x.squeeze(1)[:, :-1, :-1]

#     def _init_weights(self):
#         """Initializes weights using He et al. (2015)."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight.data)
#                 m.bias.data.zero_()
