#!/usr/bin/env python 
# coding:utf-8

import time, argparse, gc

import numpy as np
import cupy as cp
import pandas as pd

from sklearn import metrics

from rdkit import Chem

from feature import *
import SCFPfunctions as Mf
import SCFPmodel as Mm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Reporter, report, report_scope
from chainer import Link, Chain, ChainList, training
from chainer.datasets import tuple_dataset
from chainer.dataset import concat_examples
from chainer.training import extensions

import GPy
import GPyOpt

#-------------------------------------------------------------
 # featurevectorのサイズ
atomInfo = 21
structInfo = 21
lensize = atomInfo + structInfo
#-------------------------------------------------------------
    
START = time.time()

#-------------------------------
    #hypterparameter
parser = argparse.ArgumentParser(description='CNN fingerprint')    
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of moleculars in each mini-batch')
parser.add_argument('--validation', '-v', type=int, default= 5, help='N-fold cross validation')
parser.add_argument('--epoch', '-e', type=int, default=300, help='Number of sweeps over the dataset to train')
parser.add_argument('--frequency', '-f', type=int, default=1, help='Frequency of taking a snapshot')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--output', '-o', required=True, help='Directory to output the result')
parser.add_argument('--input', '-i', required=True, help='Input SDFs Dataset')
parser.add_argument('--atomsize', '-a', type=int, default=400, help='max length of smiles')
parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
parser.add_argument('--protein', '-p', required=True, help='Name of protein (subdataset)')
parser.add_argument('--boost', type=int, default=1, help='Positive sample boost')
parser.add_argument('--k1', type=int, default=5, help='window-size of first convolution layer')
parser.add_argument('--s1', type=int, default=1, help='stride-step of first convolution layer')
parser.add_argument('--f1', type=int, default=960, help='No. of filters of first convolution layer')
parser.add_argument('--k2', type=int, default=19, help='window-size of first pooling layer')
parser.add_argument('--s2', type=int, default=1, help='stride-step of first max-pooling layer')
parser.add_argument('--k3', type=int, default=49, help='window-size of second convolution layer')
parser.add_argument('--s3', type=int, default=1, help='stride-step of second convolution layer')
parser.add_argument('--f3', type=int, default=480, help='No. of filters of second convolution layer')
parser.add_argument('--k4', type=int, default=33, help='window-size of second pooling layer')
parser.add_argument('--s4', type=int, default=1, help='stride-step of second pooling layer')
parser.add_argument('--n_hid', type=int, default=160, help='No. of hidden perceptron')
parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class)')
args = parser.parse_args()

f = open(args.output+'/'+args.protein+'/CV_log.txt', 'w')
print(args.protein)
f.write('{0}\n'.format(args.protein))
    
  #-------------------------------
    # GPU check
xp = np
if args.gpu >= 0:
    print('GPU mode...')
    xp = cp
    chainer.cuda.get_device_from_id(args.gpu).use()

#-------------------------------
    # Loading SMILEs
print('Data loading...')
file = args.input + '/'+ args.protein + '_all.smiles'
f.write('Loading TOX21smiles: {0}\n'.format(file))
smi = Chem.SmilesMolSupplier(file,delimiter='\t',titleLine=False)
mols = [mol for mol in smi if mol is not None]
    
    # Make Feature Matrix
f.write('Make FeatureMatrix...\n')
F_list, T_list = [],[]
for mol in mols:
    if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > args.atomsize: f.write("too long mol was ignored\n")
    else:
        F_list.append(mol_to_feature(mol,-1,args.atomsize))
        T_list.append(mol.GetProp('_Name') )
                
    #-------------------------------    
    # Setting Dataset to model
f.write("Reshape the Dataset...\n")
Mf.random_list(F_list)
Mf.random_list(T_list)
data_t = cp.asarray(T_list, dtype=cp.int32).reshape(-1,1)
data_f = cp.asarray(F_list, dtype=cp.float32).reshape(-1,1,args.atomsize,lensize)
f.write('{0}\t{1}\n'.format(data_t.shape, data_f.shape))

f.write('Validate the Dataset...k ={0}\n'.format(args.validation))
dataset = datasets.TupleDataset(data_f, data_t)
if args.validation > 1:
    dataset = datasets.get_cross_validation_datasets(dataset, args.validation)
    #dataset = datasets.get_cross_validation_datasets_random(dataset, args.validation)
    
#-------------------------------
# reset memory
del mol, mols, data_f, data_t, F_list, T_list
gc.collect()
#-------------------------------      
# 5-fold
print('Training...')
f.write('Convolutional neural network is  running...\n')
v = 1
while v <= args.validation:
    print('...{0}'.format(v))
    f.write('Cross-Validation : {0}\n'.format(v))

    # Set up a neural network to train
    model = Mm.CNN(args.atomsize, lensize, args.k1, args.s1, args.f1, args.k2, args.s2, args.k3, args.s3, args.f3,args.k4, args.s4,args.n_hid,args.n_out)

#-------------------------------
    # Make a specified GPU current
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

#-------------------------------
    # Setup an optimizer
    f.write('Optimizer is setting up...\n')
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

#-------------------------------      
    # Set up a trainer
    f.write('Trainer is setting up...\n')
        
    train_iter = chainer.iterators.SerialIterator(dataset[v-1][0], batch_size= args.batchsize, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(dataset[v-1][1], batch_size= args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.output+'/'+args.protein+'/')    

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot_object(model, 'model_'+str(v)+'_snapshot_{.updater.epoch}'), trigger=(frequency,'epoch'))
    trainer.extend(extensions.snapshot_object(optimizer, 'optimizer_'+str(v)+'_snapshot_{.updater.epoch}'), trigger=(args.epoch,'epoch'))
    #trainer.extend(extensions.snapshot_object(model, 'model_'+str(v)+'_snapshot_{.updater.iteration}', trigger=(frequency,'iteration')))
    #trainer.extend(extensions.snapshot_object(optimizer, 'optimizer_'+str(v)+'_snapshot_{.updater.iteration}', trigger=(frequency,'iteration')))

    # Write a log of evaluation statistics for each epoch    
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name='log_'+str(v)+'_epoch'))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), log_name='log_'+str(v)+'_iteration'))

    # Print selected entries of the log to stdout
    trainer.extend(extensions.PrintReport( ['epoch', 'elapsed_time',  
                                            'main/loss', 'validation/main/loss',
                                          'main/accuracy','validation/main/accuracy',
                                           ]))

# Run the training
    trainer.run()
    v = v +1
    
    



END = time.time()
f.write('Nice, your Learning Job is done.\n')
f.write("Total time is {} sec.\n".format(END-START))
f.close()