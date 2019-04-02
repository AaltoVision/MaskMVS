import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch import Tensor
import cv2
import math
import numpy as np
import time
from numpy.linalg import inv


def down_conv_layer(input_channels, output_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=1,
            bias=False),
   nn.BatchNorm2d(output_channels),
   nn.ReLU(),
        nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=2,
            bias=False),
   nn.BatchNorm2d(output_channels),
   nn.ReLU())

def conv_layer(input_channels, output_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False),
  nn.BatchNorm2d(output_channels),
        nn.ReLU())

def up_conv_layer(input_channels, output_channels, kernel_size):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False),
  nn.BatchNorm2d(output_channels),
        nn.ReLU())


def mask_layer(input_channels,num_d):
    return nn.Sequential(
        nn.Conv2d(input_channels, num_d, 3, padding=1), nn.Sigmoid())


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]






class MaskNet(nn.Module):


    def __init__(self, num_d, batchNorm=True):
        super(MaskNet, self).__init__()



        self.conv1 = down_conv_layer(3*(1+num_d), 128, 7)
        self.conv2 = down_conv_layer(128, 256, 5)
        self.conv3 = down_conv_layer(256, 512, 3)
        self.conv4 = down_conv_layer(512, 512, 3)
        self.conv5 = down_conv_layer(512, 512, 3)

        self.upconv5 = up_conv_layer(512, 512, 3)
        self.iconv5 = conv_layer(1024, 512, 3)  #input upconv5 + conv4

        self.upconv4 = up_conv_layer(512, 512, 3)
        self.iconv4 = conv_layer(1024, 512, 3)  #input upconv4 + conv3
        self.mask4 = mask_layer(512,num_d)

        self.upconv3 = up_conv_layer(512, 256, 3)
        self.iconv3 = conv_layer(
            512+num_d, 256, 3)  
        self.mask3 = mask_layer(256,num_d)

        self.upconv2 = up_conv_layer(256, 128, 3)
        self.iconv2 = conv_layer(
            256+num_d, 128, 3)  
        self.mask2 = mask_layer(128,num_d)

        self.upconv1 = up_conv_layer(128, 64, 3)
        self.iconv1 = conv_layer(64+num_d, 64,
                                 3)
        self.mask1 = mask_layer(64,num_d)


    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        upconv5 = crop_like(self.upconv5(conv5),conv4)

        iconv5 = self.iconv5(torch.cat((upconv5, conv4), 1))

        upconv4 = crop_like(self.upconv4(iconv5),conv3)
        iconv4 = self.iconv4(torch.cat((upconv4, conv3), 1))
        mask4 = self.mask4(iconv4)
        umask4 = F.upsample(mask4, scale_factor=2)

        upconv3 = crop_like(self.upconv3(iconv4),conv2)
        iconv3 = self.iconv3(torch.cat((upconv3, conv2, umask4), 1))
        mask3 = self.mask3(iconv3)
        umask3 = F.upsample(mask3, scale_factor=2)

        upconv2 = crop_like(self.upconv2(iconv3),conv1)
        iconv2 = self.iconv2(torch.cat((upconv2, conv1, umask3), 1))
        mask2 = self.mask2(iconv2)
        umask2 = F.upsample(mask2, scale_factor=2)

        upconv1 = crop_like(self.upconv1(iconv2),umask2)
        iconv1 = self.iconv1(torch.cat((upconv1, umask2), 1))
        mask1 = self.mask1(iconv1)
        return mask1

