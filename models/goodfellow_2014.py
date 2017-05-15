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
from common import Maxout1d

__all__ = ['DnetGF2014', 'GnetGF2014'] # For import

class GnetGF2014(nn.Module):
    '''
    Implements Goodfellow2014 generator.
    '''

    def __init__(self, layer_sz, image_sz):
        '''
        Initializes generative model.

        @param layer_sz Layers size list.
        @param image_sz Output image dimension (C,H,W).
        '''

        # Initializing
        super(GnetGF2014, self).__init__()
        self.layer_sz, self.image_sz = layer_sz, image_sz

        # Finding number of layers
        self.nlayers = len(layer_sz)-1

        # Setting feature layers
        layers = []
        for i in range(self.nlayers-1):
            layers.extend(self.new_layer(layer_sz[i], layer_sz[i+1]))

        # Middle layers
        self.middle = nn.Sequential(*layers)

        # Last layer
        self.final = nn.Sequential(
            nn.Linear(layer_sz[-2], layer_sz[-1]),
            nn.Tanh()
        )

    def new_layer(self, n_in, n_out):
        '''
        Creates a new layer for the generator.

        @param n_in number of inputs.
        @param number of outputs.
        @return new layer
        '''
        layer = []
        layer.append(nn.Linear(n_in, n_out))
        layer.append(nn.LeakyReLU(0.2, inplace=True))
        return layer

    def forward(self, x):
        '''
        Computes layer forward pass.

        @param x input data.
        '''
        x = self.middle(x)
        x = self.final(x)
        x = x.view(x.size(0), *self.image_sz)
        return x

class DnetGF2014(nn.Module):
    '''
    Implements Goodfellow2014 discriminator.
    '''

    def __init__(self, layer_sz, image_sz):
        '''
        Initializes discriminative model.

        @param layer_sz Layers size list.
        @param image_sz Output image dimension (C,H,W).
        '''

        # Initializing
        super(DnetGF2014, self).__init__()
        self.layer_sz, self.image_sz = layer_sz, image_sz

        # Finding number of layers
        self.nlayers = len(layer_sz)-1

        # Setting feature layers
        layers = []
        for i in range(self.nlayers-1):
            layers.extend(self.new_layer(layer_sz[i], layer_sz[i+1]))

        # Middle layers
        self.middle = nn.Sequential(*layers)

        # Last layer
        self.final = nn.Sequential(
            nn.Linear(layer_sz[-2], layer_sz[-1]),
            nn.Sigmoid()
        )

    def new_layer(self, n_in, n_out):
        '''
        Creates a new layer for the generator.

        @param n_in number of inputs.
        @param number of outputs.
        @return new layer
        '''
        layer = []
        layer.append(Maxout1d(n_in, n_out))
        return layer

    def forward(self, x):
        '''
        Computes layer forward pass.

        @param x input data.
        '''
        x = x.view(x.size(0), -1)
        x = self.middle(x)
        x = self.final(x)
        return x
