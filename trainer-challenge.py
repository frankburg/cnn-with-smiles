#!/usr/bin/env python 
# coding:utf-8

import time, argparse, gc, os

import numpy as np
import cupy as cp
import pandas as pd

from rdkit import Chem

from feature import *
import SCFPfunctions as Mf
import SCFPmodel as Mm

# chainer v2
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Reporter, report, report_scope
from chainer import Link, Chain, ChainList, training
from chainer.datasets import tuple_dataset
from chainer.training import extensions

#-------------------------------------------------------------
 # featurevector size
atomInfo = 21
structInfo = 21
lensize= atomInfo + structInfo

#------------------------------------------------------------- 
def main():
    
    START = time.time()
    
    #--------------------------
    parser = argparse.ArgumentParser(description='SMILES CNN fingerprint')    
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of moleculars in each mini-batch. Default = 32')
    parser.add_argument('--epoch', '-e', type=int, default= 500, help='Number of sweeps over the dataset to train. Default = 500')
    parser.add_argument('--frequency', '-f', type=int, default=1, help='Frequency of taking a snapshot. Defalt = 1')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (-1 indicates CPU). Default = -1')
    parser.add_argument('--output', '-o',  required=True, help='PATH to output')
    parser.add_argument('--input', '-i',  required=True, help='PATH to TOX21 data')
    parser.add_argument('--atomsize', '-a', type=int, default=400, help='Max length of smiles, SMILES which length is larger than this value will be skipped. Default = 400')
    parser.add_argument('--protein', '-p', required=True, help='Name of protein (subdataset)')
    parser.add_argument('--boost', type=int, default=-1, help='Augmentation rate (-1 indicates OFF). Default = -1')
    parser.add_argument('--k1', type=int, default=11, help='window-size of first convolution layer. Default = 11')
    parser.add_argument('--s1', type=int, default=1, help='stride-step of first convolution layer. Default = 1')
    parser.add_argument('--f1', type=int, default=128, help='No. of filters of first convolution layer. Default = 128')
    parser.add_argument('--k2', type=int, default=5, help='window-size of first pooling layer. Default = 5')
    parser.add_argument('--s2', type=int, default=1, help='stride-step of first max-pooling layer. Default = 1')
    parser.add_argument('--k3', type=int, default=11, help='window-size of second convolution layer. Default = 11')
    parser.add_argument('--s3', type=int, default=1, help='stride-step of second convolution layer. Default = 1')
    parser.add_argument('--f3', type=int, default=64, help='No. of filters of second convolution layer. Default = 64')
    parser.add_argument('--k4', type=int, default=5, help='window-size of second pooling layer. Default = 5')
    parser.add_argument('--s4', type=int, default=1, help='stride-step of second pooling layer. Default = 1')
    parser.add_argument('--n_hid', type=int, default=96, help='No. of hidden perceptron. Default = 96')
    parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class). Default = 1')

    args = parser.parse_args()
    
    print('GPU: ', args.gpu)
    print('# Minibatch-size: ', args.batchsize)
    print('# epoch: {}'.format(args.epoch))
    print('# 1st convolution: ',args.k1, args.s1, args.f1)
    print('# max-pooling: ',args.k2, args.s2)
    print('# 2nd convolution: ',args.k3, args.s3, args.f3)
    print('# max-pooling: ',args.k4, args.s4)
    print('')
    
  #-------------------------------
    # GPU check
    xp = np
    if args.gpu >= 0:
        print('GPU mode')
        xp = cp

#-------------------------------
    # Loading SMILEs
    print('Making Training  Dataset...')
    file=args.input + '/' + args.protein + '_wholetraining.smiles'
    print('Loading smiles: ', file)
    smi = Chem.SmilesMolSupplier(file,delimiter=' ',titleLine=False)
    mols = [mol for mol in smi if mol is not None]
    
    F_list, T_list = [],[]
    for mol in mols:
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > args.atomsize: print("too long mol was ignored")
        else:
            F_list.append(mol_to_feature(mol,-1,args.atomsize))
            T_list.append(mol.GetProp('_Name'))
    Mf.random_list(F_list)
    Mf.random_list(T_list)
    
    data_t = xp.asarray(T_list, dtype=cp.int32).reshape(-1,args.n_out)
    data_f = xp.asarray(F_list, dtype=cp.float32).reshape(-1,args.n_out,args.atomsize,lensize)
    print(data_t.shape, data_f.shape)
    train_dataset = datasets.TupleDataset(data_f, data_t)
    
    print('Making Scoring Dataset...')
    ile=args.input + '/' + args.protein + '_score.smiles'
    print('Loading smiles: ', file)
    smi = Chem.SmilesMolSupplier(file,delimiter='\t',titleLine=False)
    mols = [mol for mol in smi if mol is not None]
    
    F_list, T_list = [],[]
    for mol in mols:
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > args.atomsize: print("SMIELS is too long. This mol will be ignored.")
        else:
            F_list.append(mol_to_feature(mol,-1,args.atomsize))
            T_list.append(mol.GetProp('_Name'))            
    Mf.random_list(F_list)
    Mf.random_list(T_list)
    data_t = xp.asarray(T_list, dtype=cp.int32).reshape(-1,1)
    data_f = xp.asarray(F_list, dtype=cp.float32).reshape(-1,1,args.atomsize,lensize)
    print(data_t.shape, data_f.shape)
    test_dataset = datasets.TupleDataset(data_f, data_t)
    
    #-------------------------------
    # reset memory
    del mol, mols, data_f, data_t, F_list, T_list
    gc.collect()
    
#-------------------------------      
    # Set up a neural network to train
    model = Mm.CNN(args.atomsize, lensize, args.k1, args.s1, args.f1, args.k2, args.s2, args.k3, args.s3, args.f3,args.k4, args.s4,args.n_hid,args.n_out)

    #-------------------------------
        # Make a specified GPU current
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    
    #-------------------------------
        # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
        
    #-------------------------------      
        # Set up a trainer
    print('Trainer is setting up...')
    
    output_dir = args.input+'/'+args.protein
    os.makedirs(output_dir)
    
    train_iter = chainer.iterators.SerialIterator(train_dataset, batch_size= args.batchsize, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test_dataset, batch_size= args.batchsize, repeat=False, shuffle=True)
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=output_dir)
    
       # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
        # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot_object(model, 'model_snapshot_{.updater.epoch}'), trigger=(args.frequency,'epoch'))
        # Write a log of evaluation statistics for each epoch    
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name='log_epoch'))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), log_name='log_iteration'))
        # Print selected entries of the log to stdout
    trainer.extend(extensions.PrintReport( ['epoch', 'elapsed_time','main/loss', 'validation/main/loss','main/accuracy','validation/main/accuracy']))
         # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())
    
    # Run the training
    trainer.run()
    
    END = time.time()
    print('Nice, your Learning Job is done.　Total time is {} sec．'.format(END-START))
    
#-------------------------------      
    # Model Fegure    

#------------------------------- 
if __name__ == '__main__':
    main()
