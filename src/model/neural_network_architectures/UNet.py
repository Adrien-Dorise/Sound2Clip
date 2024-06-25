"""
This is DragonflAI example on a Unet neural network architecture
Last Update by Edouard Villain - June 2024

Author: Edouard Villain, Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
Created: May 2024
"""


from dragonflai.model.neuralNetwork import NeuralNetwork
from dragonflai.utils.utils_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvBlock(nn.Module):
    """Class to performs a double convolution, a batch normalization and a ReLu activation"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """Initialize the DoubleConvBlock class

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels after the convolution
            mid_channels (int, optional): Number of channels produced by the intermediate conv. If None, set to 'out_channels'. Defaults to None.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Do the forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output Tensor of shape (N, out_channels, H, W)
        """
        return self.double_conv(x)


class DownSample(nn.Module):
    """A class that compute a downsampling and a double convolution"""

    def __init__(self, in_channels, out_channels):
        """Initialize the DownSample class

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels in the output image
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        """Do the forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    """A class that compute an upscaling and a double convolution"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        """Initialize UpSample class

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels in the output image
            bilinear (bool, optional): If True, apply bilinear upsampling, if False, use a transposed conv. Defaults to True.
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        """Compute the forward pass

        Args:
            x1 (torch.Tensor): Input tensor to be upscaled
            x2 (torch.Tensor): Tensor to be concatenated with the upscaled x1 tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Class to create output layer"""
    def __init__(self, in_channels, out_channels):
        """Initialize OutConv class

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels in the output of the network
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Compute the forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor (1D)
        """
        return self.conv(x)

class UNetModel(nn.Module):
    """Class for UNet model creation"""
    def __init__(self, n_channels, n_classes, bilinear=False, ):
        """Initialize the UNet Model

        Args:
            n_channels (int): Number of channels in the input image
            n_classes (int): Number of classes
            bilinear (bool, optional): If True, use bilinear upsampling, if False, use a transposed conv . Defaults to False.
        """
        super(UNetModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConvBlock(n_channels, 64))
        self.down1 = (DownSample(64, 128))
        self.down2 = (DownSample(128, 256))
        self.down3 = (DownSample(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (DownSample(512, 1024 // factor))
        self.up1 = (UpSample(1024, 512 // factor, bilinear))
        self.up2 = (UpSample(512, 256 // factor, bilinear))
        self.up3 = (UpSample(256, 128 // factor, bilinear))
        self.up4 = (UpSample(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        """Compute the forward pass

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor of shape (N, n_classes, H, W)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_PET(NeuralNetwork):
    """UNet Class used in exemple inheriting from the NeuralNetwork class"""
    def __init__(self, n_channels, n_classes, save_path="./results/tmp/"):
        """Initialize UNet_PET class

        Args:
            n_channels (int): Number of channels in the input image
            n_classes (int): Number of classes
        """
        super().__init__(modelType=modelType.NEURAL_NETWORK, taskType=taskType.REGRESSION, save_path=save_path)

        self.architecture = UNetModel(n_channels, n_classes).to(self.device)