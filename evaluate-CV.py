#!/usr/bin/env python 
# coding:utf-8

import argparse
import gc

import numpy as np
import cupy as cp
import pandas as pd
from sklearn import metrics

from rdkit import Chem
from feature import *
import SCFPfunctions as Mf
import SCFPmodel as Mm

from sklearn import metrics

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, serializers
from chainer import Link, Chain, ChainList
from chainer.datasets import tuple_dataset
from chainer.training import extensions

#-------------------------------------------------------------
# featurevectorのサイズ
atomInfo = 21
structInfo = 21
lensize= atomInfo + structInfo

#------------------------------------------------------------- 
def main():
    
    #引数管理
    parser = argparse.ArgumentParser(description='CNN fingerprint')    
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of moleculars in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=300, help='Number of max iteration to evaluate')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--frequency', '-f', type=int, default=1, help='Frequency of taking a snapshot')
    parser.add_argument('--validation', '-v', type=int, default= 5, help='Cross validation No.')
    parser.add_argument('--model', '-m', required=True, help='Directory to Model to evaluate')
    parser.add_argument('--data', '-s', required=True, help='Input Smiles Dataset')
    parser.add_argument('--protein', '-p', default='NR-AR', help='Name of protain what you are choose')
    parser.add_argument('--atomsize', '-a', type=int, default=400, help='max length of smiles')
    parser.add_argument('--k1', type=int, default=1, help='window-size of first convolution layer')
    parser.add_argument('--s1', type=int, default=1, help='stride-step of first convolution layer')
    parser.add_argument('--f1', type=int, default=1, help='No. of filters of first convolution layer')
    parser.add_argument('--k2', type=int, default=1, help='window-size of first pooling layer')
    parser.add_argument('--s2', type=int, default=1, help='stride-step of first max-pooling layer')
    parser.add_argument('--k3', type=int, default=1, help='window-size of second convolution layer')
    parser.add_argument('--s3', type=int, default=1, help='stride-step of second convolution layer')
    parser.add_argument('--f3', type=int, default=1, help='No. of filters of second convolution layer')
    parser.add_argument('--k4', type=int, default=1, help='window-size of second pooling layer')
    parser.add_argument('--s4', type=int, default=1, help='stride-step of second pooling layer')
    parser.add_argument('--n_hid', type=int, default=1, help='No. of hidden perceptron')
    parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class)')
    
    args = parser.parse_args()
    
    #-------------------------------
    # Loading SMILEs
    file=args.data + '/'+ args.protein + '_all.smiles'
    print('Loading TOX21smiles: ', file)
    smi = Chem.SmilesMolSupplier(file,delimiter='\t',titleLine=False)
    mols = [mol for mol in smi if mol is not None]
    
    # Make Feature Matrix
    print('Make FeatureMatrix...')
    F_list, T_list = [],[]
    for mol in mols:
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > args.atomsize: print("too long mol was ignored")
        else:
            F_list.append(mol_to_feature(mol,-1,args.atomsize))
            T_list.append(mol.GetProp('_Name') )
                
    #-------------------------------    
    # Setting Dataset to model
    print("Reshape the Dataset...")
    Mf.random_list(F_list)
    Mf.random_list(T_list)
    data_t = cp.asarray(T_list, dtype=cp.int32).reshape(-1,1)
    data_f = cp.asarray(F_list, dtype=cp.float32).reshape(-1,1,args.atomsize,lensize)
    print(data_t.shape, data_f.shape)
    borders = [len(data_t) * i // args.validation for i in range(args.validation+1)]
    borders.reverse()
    with cp.cuda.Device(args.gpu):
        data_t_gpu = cp.array(data_t)
        data_f_gpu = cp.array(data_f)
    print('')
    
#-------------------------------
    # reset memory
    del mol, mols, data_f, data_t, F_list, T_list
    gc.collect()
    
#-------------------------------      
    print('Evaluater is  running...')

#-------------------------------
    # Set up a neural network to evaluate
    model = Mm.CNN(args.atomsize, lensize, args.k1, args.s1, args.f1, args.k2, args.s2, args.k3, args.s3, args.f3, args.k4, args.s4, args.n_hid, args.n_out)
    model.compute_accuracy = False
    model.to_gpu(args.gpu)
    f = open(args.model+'/'+args.protein+'/evaluation_epoch_each.csv', 'w') 

#-------------------------------
    #list_TP, list_FP, list_FN, list_TN = [], [], [], []
    
    print("epoch","validation","TP","FN","FP","TN","Loss","Accuracy","B_accuracy","Sepecificity","Precision","Recall","F-measure","AUC", sep="\t")
    f.write("epoch,validation,TP,FN,FP,TN,Loss,Accuracy,B_accuracy,Sepecificity,Precision,Recall,F-measure,AUC\n")

    for v in range(args.validation):  
        for epoch in range(args.frequency, args.epoch+1 ,args.frequency):
            
            with cp.cuda.Device(args.gpu):
                
                serializers.load_npz(args.model+'/'+args.protein+'/model_'+str(v+1)+'_snapshot_' + str(epoch), model)
                x_gpu = data_f_gpu[borders[v+1]:borders[v]]
                y_gpu = data_t_gpu[borders[v+1]:borders[v]]
                batch = [len(y_gpu) * i // 200 for i in range(200+1)]
                pred_score_tmp,loss_tmp =[],[]
                
                for i in range(200):
                    x_tmp = x_gpu[batch[i]:batch[i+1]]
                    y_tmp = y_gpu[batch[i]:batch[i+1]]
                    pred_tmp_gpu, sr = model.predict(Variable(x_tmp))
                    pred_tmp_gpu = F.sigmoid(pred_tmp_gpu.data)
                    loss_tmp_gpu = model(Variable(x_tmp),Variable(y_tmp)).data
                    pred_score_tmp.extend(pred_tmp_gpu.data.reshape(-1).get().tolist())
                    loss_tmp.append(loss_tmp_gpu.get().tolist())
                    
                loss = np.mean(loss_tmp)
                pred_score_gpu = cp.array(pred_score_tmp).reshape(-1,1)
                pred_gpu = 1*(pred_score_gpu>=0)
                pred_score_gpu = F.sigmoid(pred_score_gpu).data
                
                count_TP_gpu= cp.sum(cp.logical_and(y_gpu == pred_gpu, pred_gpu == 1)*1)
                count_FP_gpu = cp.sum(cp.logical_and(y_gpu != pred_gpu, pred_gpu == 1)*1)
                count_FN_gpu = cp.sum(cp.logical_and(y_gpu != pred_gpu, pred_gpu == 0)*1)
                count_TN_gpu = cp.sum(cp.logical_and(y_gpu ==pred_gpu, pred_gpu == 0)*1)
            
                count_TP = count_TP_gpu.get()
                count_FP = count_FP_gpu.get()
                count_FN = count_FN_gpu.get()
                count_TN = count_TN_gpu.get()
                y = y_gpu.get()
                pred_score = pred_score_gpu.get()
            
            Accuracy = (count_TP + count_TN)/(count_TP+count_FP+count_FN+count_TN)
            Sepecificity = count_TN/(count_TN + count_FP)
            Precision = count_TP/(count_TP+count_FP)
            Recall = count_TP/(count_TP+count_FN)
            Fmeasure = 2*Recall*Precision/(Recall+Precision)
            B_accuracy = (Sepecificity+Recall)/2
            AUC = metrics.roc_auc_score(y, pred_score, average = 'weighted')
        
            print(epoch,v+1,count_TP,count_FN,count_FP,count_TN,loss,Accuracy,B_accuracy,Sepecificity,Precision,Recall,Fmeasure,AUC, sep="\t")
            text = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}\n'.format(
                epoch,v+1,count_TP,count_FN,count_FP,count_TN,loss,Accuracy,B_accuracy,Sepecificity,Precision,Recall,Fmeasure,AUC)
            f.write(text)
    
    f.close()

#------------------------------- 
if __name__ == '__main__':
    main()
