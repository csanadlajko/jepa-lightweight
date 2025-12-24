import torch
import torch.nn as nn
from torch.nn.functional import relu

# standard unet model used for tumor segmentation
# code credit: https://medium.com/data-science/cook-your-first-u-net-in-pytorch-b3297a844cf3

class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # encoder
        self.enc11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # in_chan size 3 might be useless in case of cancer dataset
        self.enc12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # pooling for dimension reduction

        self.enc21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.enc52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        
        # decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        xe11 = relu(self.enc11(x))
        xe12 = relu(self.enc12(xe11))
        xpool1 = self.pool1(xe12)

        xe21 = relu(self.enc21(xpool1))
        xe22 = relu(self.enc22(xe21))
        xpool2 = self.pool2(xe22)

        xe31 = relu(self.enc31(xpool2))
        xe32 = relu(self.enc32(xe31))
        xpool3 = self.pool3(xe32)

        xe41 = relu(self.enc41(xpool3))
        xe42 = relu(self.enc42(xe41))
        xpool4 = self.pool4(xe42)

        xe51 = relu(self.enc51(xpool4))
        xe52 = relu(self.enc52(xe51))


        # decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.dec11(xu11))
        xd12 = relu(self.dec12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.dec21(xu22))
        xd22 = relu(self.dec22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.dec31(xu33))
        xd32 = relu(self.dec32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.dec41(xu44))
        xd42 = relu(self.dec42(xd41))

        out = self.outconv(xd42)

        return out


