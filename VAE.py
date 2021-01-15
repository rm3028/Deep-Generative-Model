
import torch
from torch.nn import *
import torch.nn.functional as F

class VAE(torch.nn.Module):
    def __init__(self, imagSize, codeSize):
        super(VAE, self).__init__()
        self.block1_conv1 = Conv2d(3, 16, 3, stride=1, padding=1)
        self.block1_conv2 = Conv2d(16, 16, 3, stride=1, padding=1)
        self.block2_conv1 = Conv2d(16, 32, 3, stride=1, padding=1)
        self.block2_conv2 = Conv2d(32, 32, 3, stride=1, padding=1)
        self.block3_conv1 = Conv2d(32, 64, 3, stride=1, padding=1)
        self.block3_conv2 = Conv2d(64, 64, 3, stride=1, padding=1)
        self.block4_conv1 = Conv2d(64, 128, 3, stride=1, padding=1)
        self.block4_conv2 = Conv2d(128, 128, 3, stride=1, padding=1)

        self.block5_tconv1 = ConvTranspose2d(128, 128, 3, stride=1, padding=1)
        self.block5_tconv2 = ConvTranspose2d(128, 128, 3, stride=1, padding=1)
        self.block6_tconv1 = ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.block6_tconv2 = ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.block7_tconv1 = ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1)
        self.block7_tconv2 = ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        self.block8_tconv1 = ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1)
        self.block8_tconv2 = ConvTranspose2d(16, 3, 3, stride=1, padding=1)

        self.block1_bn1 = BatchNorm2d(16)
        self.block1_bn2 = BatchNorm2d(16)
        self.block2_bn1 = BatchNorm2d(32)
        self.block2_bn2 = BatchNorm2d(32)
        self.block3_bn1 = BatchNorm2d(64)
        self.block3_bn2 = BatchNorm2d(64)
        self.block4_bn1 = BatchNorm2d(128)
        self.block4_bn2 = BatchNorm2d(128)

        self.block5_bn1 = BatchNorm2d(128)
        self.block5_bn2 = BatchNorm2d(128)
        self.block6_bn1 = BatchNorm2d(64)
        self.block6_bn2 = BatchNorm2d(64)
        self.block7_bn1 = BatchNorm2d(32)
        self.block7_bn2 = BatchNorm2d(32)
        self.block8_bn1 = BatchNorm2d(16)
        self.block8_bn2 = BatchNorm2d(3)

        self.featureSize = (imagSize >> 3)

        featureFlattenSize = 128 * self.featureSize ** 2

        self.fc_mean = Linear(featureFlattenSize, codeSize)
        self.fc_var = Linear(featureFlattenSize, codeSize)
        self.fc_decode = Linear(codeSize, featureFlattenSize)

    def encode(self, x):
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = F.relu(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = F.relu(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = F.relu(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = F.relu(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = torch.sigmoid(x)

        x = x.view(x.size(0), -1)

        mean = self.fc_mean(x)
        var = self.fc_var(x)

        return mean, var

    def decode(self, x):
        x = self.fc_decode(x)
        x = x.view(x.size(0), -1, self.featureSize, self.featureSize)

        x = self.block5_tconv1(x)
        x = self.block5_bn1(x)
        x = F.relu(x)
        x = self.block5_tconv2(x)
        x = self.block5_bn2(x)
        x = F.relu(x)

        x = self.block6_tconv1(x)
        x = self.block6_bn1(x)
        x = F.relu(x)
        x = self.block6_tconv2(x)
        x = self.block6_bn2(x)
        x = F.relu(x)

        x = self.block7_tconv1(x)
        x = self.block7_bn1(x)
        x = F.relu(x)
        x = self.block7_tconv2(x)
        x = self.block7_bn2(x)
        x = F.relu(x)

        x = self.block8_tconv1(x)
        x = self.block8_bn1(x)
        x = F.relu(x)
        x = self.block8_tconv2(x)
        x = self.block8_bn2(x)
        x = torch.sigmoid(x)

        return x

    def add_noise(self, x, var):
        e = torch.randn(x.size(), device=x.device)
        return x + torch.exp(var) * 0.5 * e

    def loss(self, mean, var):
        return torch.sum(torch.exp(var) - (1 + var) + torch.square(mean))

    def forward(self, x):
        mean, var = self.encode(x)
        x = self.add_noise(mean, var)
        x = self.decode(x)
        return x, self.loss(mean, var)