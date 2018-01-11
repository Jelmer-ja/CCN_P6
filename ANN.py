import chainer.functions as F
import chainer.links as L
from chainer import Link, Chain, ChainList, report, iterators, optimizers
import matplotlib.pyplot as plt
import math
import random
from utils import *

class Generator(Chain):
    def __init__(self, n_units):
        super(Generator, self).__init__()
        self.mnist_dim = 28
        self.n_units = n_units
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(n_units * self.mnist_dim **2)  # n_in -> n_units        INPUT LAYER
            self.l2 = L.BatchNormalization(n_units * self.mnist_dim ** 2)    # n_units -> n_out       BATCH NORMALIZATION
            self.l3 = L.Deconvolution2D(in_channels=n_units, out_channels=1, ksize=3, stride=1,pad=1,outsize=(28,28))                   #  DECONVOLUTION

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = F.relu(self.l2(h1))
        h = F.reshape(h2, [-1,self.n_units,self.mnist_dim,self.mnist_dim])
        y = F.sigmoid(self.l3(h))
        return y

class Discriminator(Chain):
    def __init__(self, n_units):
        super(Discriminator, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l2 = L.Convolution2D(in_channels=None, out_channels=n_units,ksize=3,stride=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l3 = L.Linear(None, 1)    # n_units -> n_out

    def __call__(self, x):
        h2 = F.relu(self.l2(x))
        y = F.squeeze(self.l3(h2))
        return y

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01
