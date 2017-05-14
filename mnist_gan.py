'''
mnist_gan.py

Trains a GAN model on the MNIST database.
'''

import os # path manipulation and OS resources
import time # Time measurement
import yaml # Open configuration file
import shutil # To copy/move files
import models as models # Custom GAN models
import torch # Torch variables handler
import torch.nn as nn # Networks support
import torch.nn.functional as F # functional
import torch.nn.parallel # parallel support
import torch.backends.cudnn as cudnn # Cuda support
import torch.optim as optim # Optimizer
import torchvision.transforms as tf # Data transforms
import torch.utils.data as data # Data loaders
import torchvision.utils as vutils # Image utils
from utils.meter import AverageMeter # measurement
from utils.data import ConcDataset # Data set concatenation
from torchvision.datasets import MNIST # Datasets

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

def train(dload, dmodel, gmodel, dopt, gopt, rslt_path):
    '''
    Trains the models.

    @param dload Data loader.
    @param dmodel Discriminator.
    @parma gmodel Generator.
    @param dopt Discriminator optimizer.
    @param gopt Generator optimizer.
    @param rslt_path Resulting images path.

    @return discriminator and generator losses.
    '''

    # Average meters
    tbatch = AverageMeter() # Batch time
    tdload = AverageMeter() # Loading time
    avgdlss = AverageMeter() # Average discriminator loss
    avgglss = AverageMeter() # Average generator loss

    # Switch models to training mode
    dmodel.train()
    gmodel.train()

    # Compute batch evaluation
    end_time = time.time()
    for i, (imgs, _) in enumerate(dload):

        # Loading time
        tdload.update(time.time() - end_time)

        # Setting labels
        real_label = Variable(torch.ones(imgs.shape[0]))
        fake_label = Variable(-torch.ones(imgs.shape[0]))

        # Set variables
        zdim = [imgs.shape[0], gmodel.d_in, 1, 1]
        imgs_var = torch.autograd.Variable(imgs.cuda())
        zvec_var = torch.randn(zdim).type(torch.FloatTensor)
        zvec_var = torch.autograd.Variable(zvec_var.cuda())

        # Computing generator images
        fake = gmodel(zvec_var)

        # Computing discriminator error
        dlss = nn.f.binary_cross_entropy(dmodel(imgs_var), real_label)
        dlss += nn.f.binary_cross_entropy(dmodel(fake.detach()), fake_label)

        # Update the discriminator
        dopt.zero_grad()
        dlss.backward()
        dopt.step()

        # Generate new images
        zvec_var = torch.randn(zdim).type(torch.FloatTensor)
        zvec_var = torch.autograd.Variable(zvec_var.cuda())
        fake = gmodel(zvec_var)

        # Update the generator error
        glss = nn.f.binary_cross_entropy(dmodel(fake), real_label)
        gopt.zero_grad()
        glss.backward()
        gopt.step()

        # Measure losses
        avgdlss.update(dlss.data[0], images.size(0))
        avgglss.update(glss.data[0], images.size(0))

        # Measure batch time
        tbatch.update(time.time() - end_time)

        # Print info
        print('Epoch: [{0}][{1}/{2}]\t'
            'Time {tbatch.val:.3f} ({tbatch.avg:.3f})\t'
            'Data {tdload.val:.3f} ({tdload.avg:.3f})\t'
            'DLoss {avgdlss.val:.4f} ({avgdlss.avg:.4f})\t'
            'GLoss {avgglss.val:.4f} ({avgglss.avg:.4f})\t'.format(
                epoch, i, len(loader), tbatch=tbatch, tdload=tdload,
                avgdlss=avgdlss, avgglss=avgglss
            )
        )

if __name__ == '__main__':

    # Config path
    data_path = '../data/mnist/'
    conf_path = '../conf/mnist_gf2014.yaml'
    rslt_path = '../rslt/mnist/'

    # Open config file
    global conf
    conf = None
    with open(conf_path, 'r') as cf:
        conf = yaml.load(cf)

    # Setting outputs
    cur_mdl_pth = mdls_path+'cgan_curr.pth.tar'
    bst_mdl_pth = mdls_path+'cgan_best.pth.tar'
    imgs_path = rslt_path+conf['dmodel']+'_'+conf['gmodel']+'.pdf'

    # Setting initial losses
    conf['cepoch'] = 0
    conf['dmloss'], conf['gmloss'], conf['smloss'] = [], [], []
    min_lss = float('inf')

    # Load MNIST dataset
    norms = tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize
    totns = tf.ToTensor() # Converts to tensor
    mnist_trset = MNIST(data_path, True, tf.Compose([norms, totns]))
    mnist_tsset = MNIST(data_path, False, tf.Compose([norms, totns]))
    mnist_set = ConcDataset([mnist_trset, mnist_tsset])

    # Setting loader
    btsize, nworks = conf['btsize'], conf['nworks']
    dload = data.DataLoader(reader, btsize, True, None, nworks)

    # Setting models
    imsz, zdim = conf['imsize'], conf['zindim']
    dmodel = models.__dict__[conf['dmodel']](imsz, 1)
    gmodel = models.__dict__[conf['gmodel']](zdim, imsz)

    # Setting to parallel
    dmodel = torch.nn.DataParallel(dmodel).cuda()
    gmodel = torch.nn.DataParallel(gmodel).cuda()
    cudnn.benchmark = True # Inbuilt cudnn auto-tuner (fastest)

    # Loading models
    if os.path.isfile(cur_mdl_pth):
        check = torch.load(cur_mdl_pth)
        dmodel.load_state_dict(check['dmodel'])
        gmodel.load_state_dict(check['gmodel'])

    # Setting optimizers
    dopt = optim.Adam(dmodel.parameters(), lr=conf['learnr'])
    gopt = optim.Adam(dmodel.parameters(), lr=conf['learnr'])

    # Training loop
    for epoch in range(conf['nepoch']):

        # Train model
        dmloss, gmloss = train(dload, dmodel, gmodel, dopt, gopt, rslt_path)

        # Update losses
        conf['dmloss'].append(dmloss)
        conf['gmloss'].append(gmloss)
        conf['smloss'].append(dmloss+gmloss)

        # Set checkpoint
        check = {'dmodel': dmodel.state_dict(), 'gmodel': gmodel.state_dict()}
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
        plt.plot(conf['smloss'], 'k-', label='Sum')
        plt.plot(conf['dmloss'], 'ro-.', label='D')
        plt.plot(conf['gmloss'], 'g--^', label='G')
        plt.legend(loc='best')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')

        # Save
        plt.savefig(imgs_path)
        plt.close('all')
