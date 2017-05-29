'''
mnist_gan.py

Trains a GAN model on the MNIST database.
'''

import os # path manipulation and OS resources
import time # Time measurement
import yaml # Open configuration file
import math # math operations
import shutil # To copy/move files
import argparse # command line argumments parser
import numpy as np # algebric operations
import matplotlib.pyplot as plt # plots
from scipy.linalg import hadamard # Hadamard matrix

import torch # Torch variables handler
import torch.nn as nn # Networks support
import torch.nn.functional as F # functional
import torch.nn.parallel # parallel support
import torch.backends.cudnn as cudnn # Cuda support
import torch.optim as optim # Optimizer
import torch.utils.data as data # Data loaders

import torchvision.transforms as tf # Data transforms
import torchvision.utils as vutils # Image utils
from torchvision.datasets import MNIST, CIFAR10 # Datasets

import models as models # Custom GAN models
from utils.meter import AverageMeter # measurement
from utils.data import DataCSV, ConcDataset, ToOneHot # Dataset auxiliary

def load_data(dataset, path):
    '''
    Loads dataset images.

    @param dataset dataset type
    @param path dataset/images path.

    @return dataset wrapper for dataloader.
    '''

    # Setting preprocessing methods
    scl_factor = [0.5, 0.5, 0.5]
    norms = tf.Normalize(mean=scl_factor, std=scl_factor) # Normalize
    totns = tf.ToTensor() # Converts to tensor

    # Testing dataset
    if dataset == 'MNIST':

        # Setting labels tranform
        onehot = ToOneHot(range(11))

        # Loading MNIST dataset
        trset = MNIST(path, True, tf.Compose([totns, norms]), onehot)
        tsset = MNIST(path, False, tf.Compose([totns, norms]), onehot)

        # Concatenating training and test sets to wrapper
        dwrpr = ConcDataset([trset, tsset])

    elif dataset == 'CIFAR10':

        # Setting labels tranform
        onehot = ToOneHot(range(11))

        # Loading CIFAR10 dataset
        trset = CIFAR10(path, True, tf.Compose([totns, norms]), onehot)
        tsset = CIFAR10(path, False, tf.Compose([totns, norms]), onehot)

        # Concatenating training and test sets to wrapper
        dwrpr = ConcDataset([trset, tsset])

    elif dataset == 'CELEBA':

        # Breaking csv file path from base path
        root, csv = os.path.split(path)

        # Setting images path
        imgs_path = os.path.join(root, 'imgs')

        # Loading dataseet
        scale = tf.Scale(tuple(conf['scales']))
        dwrpr = DataCSV(imgs_path, path, tf.Compose([scale, totns, norms]))

    # Return dataset wrapper
    return dwrpr

def checkpoint(state, is_best, curpath, bstpath):
    '''
    Saves current model.

    @param state Current model state.
    @param is_best Boleean flag testing if current models is the best model.
    @param curpath Current model file output path.
    @param bstpath Best model file output path.
    '''

    # Save current and best model
    torch.save(state, curpath)
    if is_best:
        shutil.copy(curpath, bstpath)


def compute_embedded(btsize, zdim, trgs, is_cond=False):
    '''
    Computes the embedded vector for the generator.

    @param btsize batch size.
    @param zdim embedded vector dimension.
    @param trgs target inputs for conditional GAN.
    @param is_cond conditional flag.
    '''

    # Computing random noise
    zvar = torch.randn(btsize, zdim)

    # Testing for conditional info
    if is_cond:

        # Computing Hadamard matrix
        tdim = trgs.size(1)
        had_mat = hadamard(zdim)[:tdim, :]/128.0
        had_mat = torch.from_numpy(had_mat).float()

        # Converting targets to embedded dimension
        zvar = zvar + torch.mm(trgs, had_mat)

    # Return variables
    return torch.autograd.Variable(zvar.cuda())

def train(dload, dmdl, gmdl, dopt, gopt, zt, path, cond=False):
    '''
    Trains the models.

    @param dload Data loader.
    @param dmdl Discriminator.
    @parma gmdl Generator.
    @param dopt Discriminator optimizer.
    @param gopt Generator optimizer.
    @param zt test figure noise
    @param path Resulting images path.
    @param cond Conditional flag

    @return discriminator and generator losses.
    '''

    # Average meters
    tbatch = AverageMeter() # Batch time
    tdload = AverageMeter() # Loading time
    avgdlss = AverageMeter() # Average discriminator loss
    avgglss = AverageMeter() # Average generator loss

    # Switch models to training mode
    dmdl.train()
    gmdl.train()

    # Embedded vector dimension
    zindim = zt.size(1)

    # Compute batch evaluation
    end_time = time.time()
    for i, (imgs, trgs) in enumerate(dload):

        # Loading time
        tdload.update(time.time() - end_time)

        # Finding batch size
        btsize = imgs.size(0)

        # Setting true labels and images
        rlbl = torch.autograd.Variable(trgs.cuda())
        ivar = torch.autograd.Variable(imgs.cuda())

        # Setting fake labels
        flbl = trgs.clone()
        flbl[:, -1] = 1.0
        flbl = torch.autograd.Variable(flbl.cuda())

        # Set embedded vector
        zvar = compute_embedded(btsize, zindim, trgs[:, :-1], cond)

        # Computing generator images
        fake = gmdl(zvar)

        # Computing discriminator error
        dlss = F.binary_cross_entropy(dmdl(ivar), rlbl)
        dlss += F.binary_cross_entropy(dmdl(fake.detach()), flbl)

        # Update the discriminator
        dopt.zero_grad()
        dlss.backward()
        dopt.step()

        # Generate new images
        zvar = compute_embedded(btsize, zindim, trgs[:, :-1], cond)
        fake = gmdl(zvar)

        # Update the generator error
        glss = F.binary_cross_entropy(dmdl(fake), rlbl)
        gopt.zero_grad()
        glss.backward()
        gopt.step()

        # Measure losses
        avgdlss.update(dlss.data[0], btsize)
        avgglss.update(glss.data[0], btsize)

        # Measure batch time
        tbatch.update(time.time() - end_time)

        # Print images and info
        if (i % 100) == 0:

            # Print info
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {tbatch.val:.3f} ({tbatch.avg:.3f})\t'
                'Data {tdload.val:.3f} ({tdload.avg:.3f})\t'
                'DLoss {avgdlss.val:.4f} ({avgdlss.avg:.4f})\t'
                'GLoss {avgglss.val:.4f} ({avgglss.avg:.4f})\t'.format(
                    epoch, i, len(dload), tbatch=tbatch, tdload=tdload,
                    avgdlss=avgdlss, avgglss=avgglss
                )
            )

            # Image
            vutils.save_image(imgs, path+'real.png',\
            normalize=True, nrow=16)
            out = gmdl(zt)
            vutils.save_image(out.data, path+'fake.png',\
            normalize=True, nrow=16)

        # Reset time
        end_time = time.time()

    # Return losses
    return avgdlss.avg, avgglss.avg

if __name__ == '__main__':

    # Setting datasets
    dset_names = ['MNIST', 'CIFAR10', 'CELEBA']

    # Arguments parser
    parser = argparse.ArgumentParser(description='GAN study parser')

    # Dataset
    parser.add_argument('--dataset', metavar='DSET', default='MNIST',
        choices=dset_names,
        help='Valid datasets: ' + ' | '.join(dset_names) + ' default: MNIST')
    parser.add_argument('--data-path', metavar='DIR', help='path to dataset')

    # Config file
    parser.add_argument('--conf-file', metavar='CONF', help='config file')

    # Output path
    parser.add_argument('--out-path', metavar='OUT', help='output path')

    # Conditional option
    parser.add_argument('--is-cond', dest='is_cond', action='store_true',
    help='use conditional info')

    # Parsing arguments
    args = parser.parse_args()

    # Config path
    data_path = args.data_path
    conf_path = args.conf_file
    rslt_path = args.out_path

    # Setting if data is conditional
    is_cond = args.is_cond

    # Open config file
    conf = None
    with open(conf_path, 'r') as cf:
        conf = yaml.load(cf)

    # Setting outputs
    cur_mdl_pth = rslt_path+'cgan_curr.pth.tar'
    bst_mdl_pth = rslt_path+'cgan_best.pth.tar'
    imgs_path = rslt_path+'performance.pdf'

    # Setting initial losses
    conf['cepoch'] = 0
    conf['dmloss'], conf['gmloss'], conf['smloss'] = [], [], []
    min_lss = float('inf')

    # Load dataset
    dwrpr = load_data(args.dataset, data_path)

    # Setting loader
    btsize, nworks = conf['btsize'], conf['nworks']
    dload = data.DataLoader(dwrpr, btsize, True, None, nworks)

    # Setting models
    imsz, dlyrs, glyrs = tuple(conf['imsize']), conf['dlayer'], conf['glayer']
    dmdl = models.__dict__[conf['dmodel']](dlyrs, imsz)
    gmdl = models.__dict__[conf['gmodel']](glyrs, imsz)

    # Setting to parallel
    dmdl = torch.nn.DataParallel(dmdl).cuda()
    gmdl = torch.nn.DataParallel(gmdl).cuda()
    cudnn.benchmark = True # Inbuilt cudnn auto-tuner (fastest)

    # Loading models
    if os.path.isfile(cur_mdl_pth):
        check = torch.load(cur_mdl_pth)
        dmdl.load_state_dict(check['dmdl'])
        gmdl.load_state_dict(check['gmdl'])
        conf = check['conf']

    # Setting optimizers
    lr, mm = conf['learnr'], conf['momntm']
    dopt = optim.Adam(dmdl.parameters(), lr=lr)
    gopt = optim.Adam(gmdl.parameters(), lr=lr)

    # Auxiliary random fixed noise
    onehot = ToOneHot(range(11))
    zt = onehot(np.arange(btsize) % 10)
    zt = compute_embedded(btsize, conf['zindim'], zt, is_cond)

    # Training loop
    for epoch in range(conf['cepoch'], conf['nepoch']):

        # Train model
        dmloss, gmloss = train(dload,dmdl,gmdl,dopt,gopt,zt,rslt_path, is_cond)

        # Update losses
        conf['dmloss'].append(dmloss)
        conf['gmloss'].append(gmloss)
        conf['smloss'].append(dmloss+gmloss)
        conf['cepoch'] = epoch+1

        # Set checkpoint
        check = {'dmdl': dmdl.state_dict(), 'gmdl': gmdl.state_dict()}
        check['conf'] = conf

        # Save model
        is_best = conf['smloss'][-1] < min_lss
        min_lss = min(conf['smloss'][-1], min_lss)
        checkpoint(check, is_best, cur_mdl_pth, bst_mdl_pth)

        # Plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.figure(figsize=(11.69,8.27))

        # Plot
        plt.plot(conf['smloss'], 'k.-', label='Sum')
        plt.plot(conf['dmloss'], 'ro-.', label='D')
        plt.plot(conf['gmloss'], 'g--^', label='G')
        plt.legend(loc='best')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')

        # Save
        plt.savefig(imgs_path)
        plt.close('all')
