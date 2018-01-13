import chainer.functions as F
import chainer.links as L
from chainer import Link, Chain, ChainList, report, iterators, optimizers
import matplotlib.pyplot as plt
import math
import random
from utils import *
from ANN import *

def main():
    epoch = 2000
    train_data, test_data = get_mnist(n_train=1000, n_test=100, with_label=False, classes=[0])
    batch_size = 5
    n_units = 10
    gen = Generator(n_units)
    dis = Discriminator(n_units)
    iterator = iterators.SerialIterator(train_data, batch_size)
    g_optimizer = optimizers.Adam()
    g_optimizer.setup(gen)
    d_optimizer = optimizers.Adam()
    d_optimizer.setup(dis)
    showtrain(train_data)
    #loss = run_network(epoch, batch_size, gen, dis, iterator, n_units, g_optimizer, d_optimizer)
    #showImages(gen)
    #plot_loss(loss, epoch)

def run_network(epoch, batch_size, gen, dis, iterator, n_units, g_optimizer, d_optimizer):
    losses = [[], []]
    for i in range(0, epoch):
        # for j in range (0,batch_size) THEY USED K=1 IN THE PAPER SO SO DO WE

        batch = iterator.next()
        dis.cleargrads();
        gen.cleargrads()
        noise = randomsample(20, batch_size)
        g_sample = gen(noise)
        disc_gen = dis(g_sample)
        disc_data = dis(np.reshape(batch, (batch_size, 1, 28, 28), order='F'))
        softmax1 = F.sigmoid_cross_entropy(disc_gen, np.zeros(batch_size).astype('int32'))
        softmax2 = F.sigmoid_cross_entropy(disc_data, np.ones(batch_size).astype('int32'))
        loss = softmax1 + softmax2
        loss.backward()
        d_optimizer.update()
        losses[0].append(loss.data)

        gloss = F.sigmoid_cross_entropy(disc_gen, np.ones(batch_size).astype('int32'))
        gloss.backward()
        g_optimizer.update()
        losses[1].append(gloss.data)
    return losses

def randomsample(size,batch_size):
    return np.random.uniform(-1.0,1.0,[batch_size,size]).astype('float32')

def plot_loss(loss,epoch):
    plt.plot(np.array(range(1, epoch + 1)), np.array(loss[0]), label='Discriminator Loss')
    plt.plot(np.array(range(1, epoch + 1)), np.array(loss[1]), label='Generator Loss')
    plt.legend()
    plt.show()

def showImages(gen,batch_size):
    f,axes = plt.subplots(2,5)
    noise = randomsample(20, 10)
    with chainer.using_config('train', False):
        images = gen(noise)
    for i in range(0,10):
        if(i % 2 == 0):
            x = 0
        else:
            x = 1
        y = int(round(i/2,0))
        axes[x][y].imshow(np.reshape(images[i].data[:,], (28, 28), order='F'))
    plt.show()

def showtrain(train):
    f,axes = plt.subplots(2,5)
    for i in range(0,10):
        if (i % 2 == 0):
            x = 0
        else:
            x = 1
        y = int(round(i / 2, 0))
        image = train[i]
        axes[x][y].imshow(np.reshape(image, (28, 28), order='F'))
    plt.show()

if(__name__ == "__main__"):
    main()