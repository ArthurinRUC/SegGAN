import torch
import torch.nn as nn
from model_unit import Up, Down, DoubleConv, OutConv
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import config as cf

# **the first layer in the model cannot use checkpoint**

class netD(nn.Module):
    def __init__(self, nc, ndf):
        super(netD, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.type = cf.loss
        if self.type == "gan":
            self.main = nn.Sequential(
                nn.Conv2d(in_channels=self.nc, out_channels=self.ndf,
                          kernel_size=4, bias=False),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool2d(512),
                nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf*2,
                          kernel_size=4, bias=False),
                nn.BatchNorm2d(num_features=self.ndf*2),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool2d(256),
                nn.Conv2d(in_channels=self.ndf*2, out_channels=self.ndf*4,
                          kernel_size=4, bias=False),
                nn.BatchNorm2d(num_features=self.ndf*4),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool2d(128),
                nn.Conv2d(in_channels=self.ndf*4, out_channels=self.ndf*8,
                          kernel_size=4, bias=False),
                nn.BatchNorm2d(num_features=self.ndf*8),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool2d(64),
                nn.Conv2d(in_channels=self.ndf*8, out_channels=1,
                          kernel_size=4, bias=False),
                nn.AdaptiveAvgPool2d(1)
            )
        elif self.type == "wgan" or self.type == "wgan-gp":
            # when use wgan(-gp), remove BN layers
                        self.main = nn.Sequential(
                nn.Conv2d(in_channels=self.nc, out_channels=self.ndf,
                          kernel_size=4, bias=False),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool2d(512),
                nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf*2,
                          kernel_size=4, bias=False),
                nn.InstanceNorm2d(num_features=self.ndf*2),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool2d(256),
                nn.Conv2d(in_channels=self.ndf*2, out_channels=self.ndf*4,
                          kernel_size=4, bias=False),
                nn.InstanceNorm2d(num_features=self.ndf*4),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool2d(128),
                nn.Conv2d(in_channels=self.ndf*4, out_channels=self.ndf*8,
                          kernel_size=4, bias=False),
                nn.InstanceNorm2d(num_features=self.ndf*8),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool2d(64),
                nn.Conv2d(in_channels=self.ndf*8, out_channels=1,
                          kernel_size=4, bias=False),
                nn.AdaptiveAvgPool2d(1)
            )

    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, 1).squeeze(1)
        return x


class netG(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(netG, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.nz, out_channels=self.ngf,
                      kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=self.ngf),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.ngf, out_channels=self.ngf *
                      2, kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=self.ngf * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.ngf * 2, out_channels=self.ngf *
                      4, kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=self.ngf * 4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.ngf * 4, out_channels=self.ngf*8,
                      kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=self.ngf*8),
            nn.LeakyReLU(0.1),)
        self.trans = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=self.ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=self.ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=self.ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.ngf,
                               out_channels=self.nc, kernel_size=4, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.conv(input)
        x = self.trans(x)
        return x


class unetG(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(unetG, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
