import chainer.functions as F
import chainer
import chainer.links as L
from chainer import Link, Chain, ChainList, report

class Generator(Chain):
    def __init__(self, n_units, n_out):
        super(Generator, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units        INPUT LAYER
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units     DENSELY CONNECTED RELU LAYER
            self.l3 = L.BatchNormalization(None, n_out)    # n_units -> n_out       BATCH NORMALIZATION
            self.l4 = L.Deconvolution2D(None, 784)                   #  DECONVOLUTION

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        y = F.sigmoid(self.l4(h3))
        return y

class Discriminator(Chain):
    def __init__(self, n_units, n_out):
        super(Discriminator, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units            INPUT LAYER
            self.l2 = L.Convolution2D(None, n_units)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l3 = L.Linear(None, 1)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.max_pooling_2d(h2)
        y = F.relu(self.l3(h3))
        return y