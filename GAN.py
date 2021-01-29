
import torch
from torch.nn import *
import torch.nn.functional as F


class Generator(torch.nn.Module):
    def __init__(self, imageSize, codeSize):
        super(Generator, self).__init__()
        self.featureSize = (imageSize >> 4)
        featureFlattenSize = 512 * self.featureSize ** 2
        self.fc_decode = Linear(codeSize, featureFlattenSize)

        self.block1_tconv1 = ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=1)
        self.block2_tconv1 = ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1)
        self.block3_tconv1 = ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.block4_tconv1 = ConvTranspose2d(64, 3, 5, stride=2, padding=2, output_padding=1)

        self.block1_bn1 = BatchNorm2d(256)
        self.block2_bn1 = BatchNorm2d(128)
        self.block3_bn1 = BatchNorm2d(64)

    def forward(self, x):
        x = self.fc_decode(x)
        x = x.view(x.size(0), -1, self.featureSize, self.featureSize)

        x = self.block1_tconv1(x)
        x = self.block1_bn1(x)
        x = F.relu(x)

        x = self.block2_tconv1(x)
        x = self.block2_bn1(x)
        x = F.relu(x)

        x = self.block3_tconv1(x)
        x = self.block3_bn1(x)
        x = F.relu(x)

        x = self.block4_tconv1(x)
        x = torch.tanh(x)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self, imageSize):
        super(Discriminator, self).__init__()
        self.block1_dropout1 = Dropout(0.5)

        self.block1_conv1 = Conv2d(3, 64, 5, stride=2, padding=2)
        self.block2_conv1 = Conv2d(64, 128, 5, stride=2, padding=2)
        self.block3_conv1 = Conv2d(128, 256, 5, stride=2, padding=2)
        self.block4_conv1 = Conv2d(256, 512, 5, stride=2, padding=2)

        self.block2_bn1 = BatchNorm2d(128)
        self.block3_bn1 = BatchNorm2d(256)
        self.block4_bn1 = BatchNorm2d(512)

        self.block1_leakyRelu1 = LeakyReLU(0.2)
        self.block2_leakyRelu1 = LeakyReLU(0.2)
        self.block3_leakyRelu1 = LeakyReLU(0.2)
        self.block4_leakyRelu1 = LeakyReLU(0.2)

        self.featureSize = (imageSize >> 4)
        featureFlattenSize = 512 * self.featureSize ** 2

        self.fc1 = Linear(featureFlattenSize, 1)

    def forward(self, x):
        x = self.block1_dropout1(x)
        x = self.block1_conv1(x)
        x = self.block1_leakyRelu1(x)

        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_leakyRelu1(x)

        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_leakyRelu1(x)

        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_leakyRelu1(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.sigmoid(x)

        return x


class GAN(torch.nn.Module):
    def __init__(self, imageSize, codeSize):
        super(GAN, self).__init__()
        self.codeSize = codeSize
        self.generator = Generator(imageSize, codeSize)
        self.discriminator = Discriminator(imageSize)

    def generate(self, x):
        x = self.generator.forward(x)
        return x

    def discriminate(self, x):
        x = self.discriminator.forward(x)
        return x

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x