import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from ANN import *
from utils import *

def main():
    epoch = 20
    train_data, test_data = get_mnist(n_train=1000,n_test=100,with_label=False,classes=[0])
    gen = Generator(784)
    dis = Discriminator(784)
    gen = GeneratorDistribution(784)
    iterator = RandomIterator()
    optimizer = optimizers.SGD()

def run_network():
    pass

if(__name__ == "__main__"):
    main()