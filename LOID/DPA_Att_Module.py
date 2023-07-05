import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision import models
from patchify import patchify, unpatchify

#image = np.random.rand(512,512,3)
#patches = patchify(image, (2,2), step = 1)
#assert patches.shape == (2,3,2,2)
#reconstructed_image = unpatchify(patches, image.shape)
#assert (reconstructed_image == image).all()

 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),  #  
            nn.ReLU(inplace=True),   #  
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.right = shortcut

    def forward(self, x):
       # print('1.. x.shape=',x.shape)
        out = self.left(x)
       # print('out.shape=',out.shape)
        residual = x if self.right is None else self.right(x)
       # print(self.right,' residual.shape=',residual.shape)
        out += residual
        return F.relu(out)


class DPA_Att_Module(nn.Module):
    # input image size is 14*14
    def __init__(self, in_channels, out_channels, size1=(576,576), size2=(1152,1152), size3=(2304,2304), size4=(4608,4608), num_classes=1, num_channels=3,pretrained=True):
        super( DPA_Att_Module, self).__init__()
 #       resnet = models.resnet34(pretrained=pretrained)
#        filters = [1024, 2048, 4096, 8192]
        self.first_conv2d = nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1)
 #       self.MA = cbam_block3(in_channels, ratio=8, kernel_size=7)  ###
        self.trunk_branches = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1)   #,
#            nn.Conv2d(in_channels, out_channels,kernel_size=1)
         )
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax1_blocks = nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1)        
        self.skip1_connection_conv2d = nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax2_blocks = nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1)
        self.skip2_connection_conv2d = nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1)
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.softmax3_blocks = nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1)
        self.skip3_connection_conv2d = nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1)
#        self.interpolation4 = nn.UpsamplingBilinear2d(size=size4)
#        self.softmax4_blocks = nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1)
        
        self.softmax5_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1),
            nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1)
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(1, 32, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer2 = self._make_layer(32, 64, 4, stride=1) # stride=2 
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer3 = self._make_layer(64, 128, 6, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer4 = self._make_layer(128, 256, 3, stride=1)
        
 #       self.fc = nn.Linear(32, num_classes)


        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

  #      self.last_blocks = nn.Conv2d(2, 256, kernel_size=3, padding=1)

    #  residual block
    def _make_layer(self, in_channels, out_channels, block_num, stride=1):

        #  
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, 1, shortcut))
        #  
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x0 = x
#        print('x0.shape=',x.shape)
        x = self.first_conv2d(x)
#        print('x1.shape=',x.shape)
 #       out_MA = self.MA(x0)   ###
        out_trunk = self.trunk_branches(x0)
#        print('out_trunk.shape=',out_trunk.shape)
        out_interp1 = self.interpolation1(x)
#        print('out_interp1.shape=',out_interp1.shape)
        out_softmax1 = self.softmax1_blocks(out_interp1)
#        print('out_softmax1.shape=',out_softmax1.shape)
        out_skip1_connection = self.skip1_connection_conv2d(out_softmax1)
        out_interp2 = self.interpolation2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_interp2)
        out_skip2_connection = self.skip2_connection_conv2d(out_softmax2)
        out_interp3 = self.interpolation3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_interp3)
 #       out_skip3_connection = self.skip3_connection_conv2d(out_softmax3)
#        out_interp4 = self.interpolation4(out_softmax3)
#        out_softmax4 = self.softmax4_blocks(out_interp4)
        out_softmax5 = self.softmax5_blocks(out_softmax3)
#        print('out_softmax5.shape=',out_softmax5.shape)
        p1 = self.pool1(out_softmax5)
        r1 = p1 + out_skip2_connection
        r1 = self.layer1(r1)
        p2 = self.pool2(r1)
        r2 = p2 + out_skip1_connection
        r2 = self.layer2(r2)
        p3 = self.pool3(r2)
#        r3 = p3 + out_skip1_connection
        r3 = self.layer3(p3)
        #p4 = self.pool4(r3)
 #       r3 = p3 + out_skip1_connection
        #r4 = self.layer4(p4)
#        print('r4.shape=',r4.shape)
        out_softmax6 = self.softmax6_blocks(r3)
 #       print('out_softmax6.shape=',out_softmax6.shape)
 #       print('out_trunk.shape=',out_trunk.shape)
        #print('before : out_softmax1.shape=',out_softmax1.shape)
        out = out_softmax6 + out_trunk
#        print('out.shape=',out.shape)
        out = (1 + out) * out_trunk
 #       out0 = out + out_MA     ###
 #       out_last = self.last_blocks(out)
        #print('out_last.shape=',out_last.shape)

        return out