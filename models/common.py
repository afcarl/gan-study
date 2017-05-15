'''
common.py

Common layers, activation units, and modules.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Maxout1d(nn.Module):
    '''
    Maxout 1D layer. See https://github.com/pytorch/pytorch/issues/805
    and https://arxiv.org/abs/1302.4389.
    '''

    def __init__(self, d_in, d_out, pool_sz=5):
        '''
        Initializes maxout layer.

        @param d_in input data dimension.
        @param d_out output data dimension.
        @param pool_sz pooling size.
        '''

        super(Maxout1d, self).__init__()
        self.d_in, self.d_out, self.pool_sz = d_in, d_out, pool_sz
        self.lin = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_in, d_out * pool_sz)
        )

    def forward(self, x):
        '''
        Computes layer forward pass.

        @param x input data.
        @return processed output data.
        '''

        # Finding output data shape
        shape = list(x.size())
        shape[-1] = self.d_out
        shape.append(self.pool_sz)
        last_dim = len(shape)-1

        # Maxout pass
        out = self.lin(x)
        out, i = out.view(*shape).max(last_dim)
        return out.squeeze(last_dim)
