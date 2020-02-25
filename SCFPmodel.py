import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Reporter, report, report_scope
from chainer import Link, Chain, ChainList, training
#from chainer.training import extensions

import numpy as np
import cupy as cp

import SCFPfunctions as Mf

#-------------------------------------------------------------
    #Network definition
class CNN(chainer.Chain):

    def __init__(self, atomsize, lensize, k1, s1, f1, k2, s2, k3, s3, f3, k4, s4, n_hid, n_out):
        
        # atomsize, lenseize = size of feature matrix
        # k1, s1, f1 = window-size, stride-step, No. of filters of first convolution layer
        # k2, s2 = window-size, stride-step of first max-pooling layer
        # k3, s3, f3 = window-size, stride-step, No. of filters of second convolution layer
        # k4, s4 = window-size, stride-step of second max-pooling layer
        
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, f1, (k1, lensize), stride=s1, pad = (k1//2,0)),
            bn1=L.BatchNormalization(f1),
            conv2=L.Convolution2D(f1, f3, (k3, 1), stride=s3, pad = (k3//2,0)),
            bn2=L.BatchNormalization(f3),
            fc3=L.Linear(None, n_hid),
            bn3=L.BatchNormalization(n_hid),
            fc4=L.Linear(None, n_out)
        )
        self.atomsize, self.lensize, self.n_out = atomsize, lensize, n_out
        self.k1, self.s1, self.f1, self.k2, self.s2, self.k3, self.s3, self.f3, self.k4, self.s4 = k1, s1, f1, k2, s2, k3, s3, f3, k4, s4
        self.l1 = (self.atomsize+(self.k1//2*2)-self.k1)//self.s1+1
        self.l2 = (self.l1+(self.k2//2*2)-self.k2)//self.s2+1
        self.l3 = (self.l2+(self.k3//2*2)-self.k3)//self.s3+1
        self.l4 = (self.l3+(self.k4//2*2)-self.k4)//self.s4+1
        
    def __call__(self, x, t):
        y, sr = self.predict(x)
        loss = F.sigmoid_cross_entropy(y, t) + sr
        accuracy = F.binary_accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss    
    
    def predict(self,x):
        h = F.leaky_relu(self.bn1(self.conv1(x))) # 1st conv
        h = F.average_pooling_2d(h, (self.k2,1), stride=self.s2, pad=(self.k2//2,0)) # 1st pooling
        h = F.leaky_relu(self.bn2(self.conv2(h))) # 2nd conv
        h = F.average_pooling_2d(h, (self.k4,1), stride=self.s4, pad=(self.k4//2,0)) # 2nd pooling
        h = F.max_pooling_2d(h, (self.l4,1)) # grobal max pooling, fingerprint
        h = self.fc3(h) # fully connected
        sr = 0.00001* cp.mean(cp.log(1 + h.data * h.data)) # sparse regularization
        h = F.leaky_relu(self.bn3(h))
        return self.fc4(h), sr

    def fingerprint(self,x):
        x = Variable(x.astype(cp.float32).reshape(-1,1, self.atomsize, self.lensize))
        h = F.leaky_relu(self.bn1(self.conv1(x))) # 1st conv
        h = F.average_pooling_2d(h, (self.k2,1), stride=self.s2, pad=(self.k2//2,0)) # 1st pooling
        h = F.leaky_relu(self.bn2(self.conv2(h))) # 2nd conv
        h = F.average_pooling_2d(h, (self.k3,1), stride=self.s3, pad=(self.k3//2,0)) # 2nd pooling
        h = F.max_pooling_2d(h, (self.l4,1)) # grobal max pooling, fingerprint
        return h.data
    
    def layer1(self,x):
        x = Variable(x.astype(cp.float32).reshape(-1,1, self.atomsize, self.lensize))
        h = self.bn1(self.conv1(x)) # 1st conv
        return h.data
    
    def pool1(self,x):
        x = Variable(x.astype(cp.float32).reshape(-1,1, self.atomsize, self.lensize))
        h = F.leaky_relu(self.bn1(self.conv1(x))) # 1st conv
        h = F.average_pooling_2d(h, (self.k2,1), stride=self.s2, pad=(self.k2//2,0)) # 1st pooling
        return h.data
    
    def layer2(self,x):
        x = Variable(x.astype(cp.float32).reshape(-1,1, self.atomsize, self.lensize))
        h = F.leaky_relu(self.bn1(self.conv1(x))) # 1st conv
        h = F.average_pooling_2d(h, (self.k2,1), stride=self.s2, pad=(self.k2//2,0)) # 1st pooling
        h = self.bn2(self.conv2(h)) # 2nd conv
        return h.data
    
    def pool2(self,x):
        x = Variable(x.astype(cp.float32).reshape(-1,1, self.atomsize, self.lensize))
        h = F.leaky_relu(self.bn1(self.conv1(x))) # 1st conv
        h = F.average_pooling_2d(h, (self.k2,1), stride=self.s2, pad=(self.k2//2,0)) # 1st pooling
        h = F.leaky_relu(self.bn2(self.conv2(h))) # 2nd conv
        h = F.average_pooling_2d(h, (self.k3,1), stride=self.s3, pad=(self.k3//2,0)) # 2nd pooling
        return h.data

#------------------------------------------------------------- 

#def strong_sigmoid(x):
#    return 1*(x >=0)
