import torch 
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.down_conv(self.in_channel, 32, bias=False, batchnorm=False)
        self.ec1 = self.down_conv(32, 64, bias=False, batchnorm=False)
        self.ec2 = self.down_conv(64, 64, bias=False, batchnorm=False)
        self.ec3 = self.down_conv(64, 128, bias=False, batchnorm=False)
        self.ec4 = self.down_conv(128, 128, bias=False, batchnorm=False)
        self.ec5 = self.down_conv(128, 256, bias=False, batchnorm=False)
        self.ec6 = self.down_conv(256, 256, bias=False, batchnorm=False)
        self.ec7 = self.down_conv(256, 512, bias=False, batchnorm=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.up_conv(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.down_conv(256 + 512, 256, bias=False)
        self.dc7 = self.down_conv(256, 256, bias=False)
        self.dc6 = self.up_conv(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.down_conv(128 + 256, 128, bias=False)
        self.dc4 = self.down_conv(128, 128, bias=False)
        self.dc3 = self.up_conv(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.down_conv(64 + 128, 64, bias=False)
        self.dc1 = self.down_conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.down_conv(64, n_classes, kernel_size=1, stride=1, padding=0, bias=False)


    def down_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def up_conv(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        print(self.dc9(e7).size(), syn2.size())
        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0
