= Convolutional neural network based on SMILES representation of molecules for detecting chemical motif =
SMILES Convolution Fingerprint (SCFP)


== Requirements ==
* Python 3 (>= 3.5.2)
* Chainer 2.1.0 (only v2)
* numpy (>= 1.11.1)
* cupy (>= 1.0.3) for GPU
* pandas (>= 0.18.1)
* scikit-learn (>= 0.17.1)
* RDkit (>= 2016.09.4)


== Usage ==
[1] Dataset

[1.1] TOX21 dataset
The TOX21 dataset already processed is contained in "./TOX21".
The original raw dataset can be downloaded at the TOX21 challenge 2014 website (https://tripod.nih.gov/tox21/challenge/data.jsp).

SUBDATASETNAME*_train.smiles    ... "Train" data for training.
SUBDATASETNAME*_test.smiles     ... "Test" data for optimization.
SUBDATASETNAME*_score.smiles    ... "Score" (test) data for final evaluation.

[1.2] Dataset processing
If you want to use your original data, you must process it using the style below:
[SMILES \t  active_or_inactive]

for example:
O=C1NC(=O)c2cc(Nc3ccccc3)c(Nc3ccccc3)cc21	0
CCn1c(=O)[nH]c2cc(Cl)c(Cl)cc21	1
O=C1Nc2ccccc2/C1=C1\Nc2ccccc2\C1=N\O	1
...


[2] Training CNN

Usage : python trainer-challenge.py -i <TOX21 folder> -o <output> -p <protein> [Other options]

This program will output model snapshots (.npz) of specified epoch and training log.

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE, -b BATCHSIZE
                        Number of molecules in each mini-batch. Default = 32 
  --epoch EPOCH, -e EPOCH
                        Number of sweeps over the dataset to train. Default = 500
  --frequency FREQUENCY, -f FREQUENCY
                        Frequency of taking a snapshot. Defalt = 1
  --gpu GPU, -g GPU     GPU ID (-1 indicates CPU). Default = -1
  --atomsize ATOMSIZE, -a ATOMSIZE
                        Max length of SMILES, SMILES whose lengths are larger than this value will be skipped. Default = 400
  --boost BOOST         Augmentation rate (-1 indicates OFF). Default = -1
  --k1 K1               Window-size of first convolution layer. Default = 11
  --s1 S1               Stride-step of first convolution layer. Default = 1
  --f1 F1               No. of filters of first convolution layer. Default = 128
  --k2 K2               Window-size of first pooling layer. Default = 5
  --s2 S2               Stride-step of first max-pooling layer. Default = 1
  --k3 K3               Window-size of second convolution layer. Default = 11
  --s3 S3               Stride-step of second convolution layer. Default = 1
  --f3 F3               No. of filters of second convolution layer. Default = 64
  --k4 K4               Window-size of second pooling layer. Default = 5
  --s4 S4               Stride-step of second pooling layer. Default = 1
  --n_hid N_HID         No. of hidden perceptrons. Default = 96
  --n_out N_OUT         No. of output perceptrons (class). Default = 1


[3] Evaluation

Usage : python trainer-challenge.py -i <TOX21 folder> -o <output> [Other options]

This program will load model snapshots and output evaluation log.

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE, -b BATCHSIZE
                        Number of molecules in each mini-batch. Default = 32
  --epoch EPOCH, -e EPOCH
                        Number of max iterations to evaluate. Default = 500
  --gpu GPU, -g GPU     GPU ID (negative value indicates CPU). Default = -1
  --frequency FREQUENCY, -f FREQUENCY
                        Epoch frequency for evaluation. Default = 1
  --model MODEL, -m MODEL
                        Directory to Model to evaluate 
  --data DATA, -d DATA  Input SMILES Dataset
  --protein PROTEIN, -p PROTEIN
                        Name of protein (subdataset)
  --k1 K1               Window-size of first convolution layer. Default = 11
  --s1 S1               Stride-step of first convolution layer. Default = 1
  --f1 F1               No. of filters of first convolution layer. Default = 128
  --k2 K2               Window-size of first pooling layer. Default = 5
  --s2 S2               Stride-step of first max-pooling layer. Default = 1
  --k3 K3               Window-size of second convolution layer. Default = 11
  --s3 S3               Stride-step of second convolution layer. Default = 1
  --f3 F3               No. of filters of second convolution layer. Default = 64
  --k4 K4               Window-size of second pooling layer. Default = 5
  --s4 S4               Stride-step of second pooling layer. Default = 1
  --n_out N_OUT         No. of output perceptrons (class). Default = 1
  --n_hid N_HID         No. of hidden perceptrons. Default = 96


== Example ==
You can demonstrate our program with test data.

$ cd /path/to/smiles/
$ mkdir /path/to/outdir/

% Cross Validation
$ python trainer-CV.py --gpu=0 -i ./TOX21 -o /path/to/outdir/ -p NR-AR --k1 1 --f1 320 --k2 51 --k3 15 --f3 880 --k4 45 --n_hid 264 
$ python evaluate-CV.py --gpu=0 -m /path/to/outdir/ -d ./TOX21 -p NR-AR --k1 1 --f1 320 --k2 51 --k3 15 --f3 880 --k4 45 --n_hid 264

- TOX21 challenge 2014
$ python trainer-challenge.py --gpu=0 -i ./TOX21 -o /path/to/outdir/ -p NR-AR --k1 1 --f1 320 --k2 51 --k3 15 --f3 880 --k4 45 --n_hid 264 
$ python evaluate-challenge.py --gpu=0 -m /path/to/outdir/ -d ./TOX21 -p NR-AR --k1 1 --f1 320 --k2 51 --k3 15 --f3 880 --k4 45 --n_hid 264


== Reference ==
* Hirohara, M., Sakakibara, Y. 
: Convolutional neural network based on SMILES representation of molecules for detecting chemical motif