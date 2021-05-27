import torch
from torch import nn
from collections import OrderedDict

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        padding = kernel_size//2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        
        x = self.block(x)
        
        return x
        
class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        
        x = self.block(x)
        
        return x
        

class EncoderBlock(nn.Module):
    def __init__(self, num_layers, kernel_size=3):
        super(EncoderBlock, self).__init__()
        padding = kernel_size//2
        
        self.init_layer1 = ConvBlock(1, 16, kernel_size)
        
        self.block1 = nn.Sequential(
            OrderedDict(
                [(f'conv1_{i}', ConvBlock(16, 16, kernel_size)) for i in range(num_layers-1)]
            )
        )
        self.mp1 = nn.MaxPool2d(2)
        
        self.init_layer2 = ConvBlock(16, 32, kernel_size)
        self.block2 = nn.Sequential(
            OrderedDict(
                [(f'conv2_{i}',ConvBlock(32, 32, kernel_size)) for i in range(num_layers-1)]
            )
        )
        self.mp2 = nn.MaxPool2d(2)
        
        self.init_layer3 = ConvBlock(32, 64, kernel_size)
        self.block3 = nn.Sequential(
            OrderedDict(
                [(f'conv3_{i}', ConvBlock(64, 64, kernel_size)) for i in range(num_layers-1)]
            )
        )
        self.last_layer = nn.Conv2d(64, 8, 7)
        
    def forward(self, x):
        
        x = self.init_layer1(x)
        x = self.block1(x) + x
        x = self.mp1(x)
        
        x = self.init_layer2(x)
        x = self.block2(x) + x
        x = self.mp2(x)
        
        x = self.init_layer3(x)
        x = self.block3(x) + x
        x = self.last_layer(x)
        
        return x
        
        
class DecoderBlock(nn.Module):
    def __init__(self, num_layers, kernel_size=3):
        super(DecoderBlock, self).__init__()
        padding = kernel_size//2
        
        self.init_layer1 = ConvTBlock(8, 64, kernel_size=7, stride=7)
        self.block1 = nn.Sequential(
            OrderedDict(
                [(f'conv1_{i}',ConvBlock(64, 64, kernel_size)) for i in range(num_layers-1)]
            )
        )
        
        self.init_layer2 = ConvTBlock(64, 32, kernel_size=2, stride=2)
        self.block2 = nn.Sequential(
            OrderedDict(
                [(f'conv2_{i}',ConvBlock(32, 32, kernel_size)) for i in range(num_layers-1)]
            )
        )
        self.init_layer3 = ConvTBlock(32, 16, kernel_size=2, stride=2)
        self.block3 = nn.Sequential(
            OrderedDict(
                [(f'conv3_{i}',ConvBlock(16, 16, kernel_size)) for i in range(num_layers-1)]
            )
        )
        
        self.last_layer = nn.Conv2d(16, 1, 1)
        
        
    def forward(self, x):
        
        x = self.init_layer1(x)
        x = self.block1(x)
        
        x = self.init_layer2(x)
        x = self.block2(x)
        
        x = self.init_layer3(x)
        x = self.block3(x)
        x = self.last_layer(x)
        
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, enc_layers, dec_layers):
        super(AutoEncoder, self).__init__()
        self.encoder = EncoderBlock(3)
        self.decoder = DecoderBlock(3)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x