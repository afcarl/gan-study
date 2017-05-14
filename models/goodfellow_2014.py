'''
goodfellow_2014.py

Implements GAN models described in https://arxiv.org/abs/1406.2661.
See https://github.com/goodfeli/adversarial.
'''

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from common import Maxout

__all__ = ['DnetGF2014', 'GnetGF2014'] # For import

class BaseCovT(nn.Module):
    '''
    Base 2D transpose convolution layer.
    '''

    def __init__(in_chn, out_chn, kernel=4, stride=2, pad=1, bias=False):
        '''
        Initializes 2D transpose convolution layer.

        @param in_chn input channels.
        @param out_chn output channels.
        @param kernel_sz kernel size.
        @param stride stride of the convolution.
        @param pad input paddding.
        @param bias If True, adds a learnable bias to the output.
        '''

        # Initializing
        super(BaseCovT, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.bias = bias

        # Set layers
        self.covt = nn.ConvTranspose2d(in_chn, out_chn, kernel, stride,\
        bias=bias)
        self.batch = nn.BatchNorm2d(out_chn)
        self.actf = nn.ReLU(True)

    def forward(self, x):
        '''
        Computes layer forward pass.

        @param x input data.
        '''

        # Computing pad
        x = F.pad(x, self.pad, mode='reflect')

        # Computing pass
        x = self.covt(x)
        x = self.batch(x)
        x = self.actf(x)
        return x

class BaseCov(nn.Module):
    '''
    Base 2D convolution layer.
    '''

    def __init__(in_chn, out_chn, kernel=4, stride=2, pad=1, bias=False):
        '''
        Initializes 2D convolution layer.

        @param in_chn input channels.
        @param out_chn output channels.
        @param kernel_sz kernel size.
        @param stride stride of the convolution.
        @param pad input paddding.
        @param bias If True, adds a learnable bias to the output.
        '''

        # Initializing
        super(BaseCovT, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.bias = bias

        # Set layers
        self.cov = nn.Conv2d(in_chn, out_chn, kernel, stride, bias=bias)
        self.batch = nn.BatchNorm2d(out_chn)
        self.actf = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        '''
        Computes layer forward pass.

        @param x input data.
        '''

        # Computing pad
        x = F.pad(x, self.pad, mode='reflect')

        # Computing pass
        x = self.cov(x)
        x = self.batch(x)
        x = self.actf(x)
        return x

class GnetGF2014(nn.Module):
    '''
    Implements Goodfellow2014 generator.
    '''

    def __init__(self, d_in, d_out):
        '''
        Initializes generative model.

        @param d_in input data dimension.
        @param d_out output data dimension. Must be a power of two.
        '''

        # Initializing
        super(GnetGF2014, self).__init__()
        self.d_in, self.d_out, self.n_chn = d_in, d_out, n_chn

        # Finding number of layers
        self.nlayers = int(math.log(d_out, 2))

        # Setting middle layers
        layers = [BaseCovT(self.d_in, self.d_out, 4, 1, 1)]
        for i in range(self.nlayers-2):
            layers.append(BaseCovT(self.d_out, d_out, 4, 2, 1))

        # Middle layers
        self.middle = nn.Sequential(layers)

        # Last layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(self.d_out, self.d_out, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        '''
        Computes layer forward pass.

        @param x input data.
        '''
        x = self.middle(x)
        return self.final(x)

class DnetGF2014(nn.Module):
    '''
    Implements Goodfellow2014 discriminator.
    '''

    def __init__(self, d_in, d_out):
        '''
        Initializes discriminative model.

        @param d_in input data dimension.
        @param d_out output data dimension. Must be a power of two.
        '''

        # Initializing
        super(DnetGF2014, self).__init__()
        self.d_in, self.d_out = d_in, d_out

        # Finding number of layers
        self.nlayers = int(math.log(d_in, 2))

        # Setting middle layers
        layers = [BaseCov(d_in, d_out, 4, 2, 1)]
        for i in range(self.nlayers-2):
            layers.append(BaseCov(self.d_out, d_out, 4, 2, 1))

        # Middle layers
        self.middle = nn.Sequential(layers)

        # Last layer
        self.final = nn.Sequential(
            nn.Conv2d(self.d_out, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Computes layer forward pass.

        @param x input data.
        '''
        x = self.middle(x)
        x = self.final(x)
        return x.view(-1, 1)
