# CCN_P6


A couple of hints for the 5th assignment to get you going:

- Make sure you understand which parts are necessary for an GAN. There are a lot of options for the different parts,
but the following architecture should work.

- You should realize that you have to make 2 separate networks. A generative and an discriminative one.
It is best to give the generative network a fully connected relu layer, followed by batch normalization and a
deconvolutional layer with an sigmoid activation function. For the discirminative network use a convolutional
network with a relu activation function and a linear readout layer.

- Make sure that the output size of the generative network is the size of the images, so 28x28. The output size of
the discriminative network is just 1 (whether it's a real zero or not).

- Think about the loss functions you use. You want to classify whether an image is a real zero or a fake one. You
can use  the log of the discriminator as loss or the sigmoid_cross_entropy loss.
See https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/ for a comparison between them.

- The order of updating the networks is important. First generate a sample with the generator network. Classify it
with the discriminator network, calculate the loss such that you enhance samples that the discriminator thinks are real
and update the networks. Next calculate the loss of the generated sample enhancing those samples that the discriminator
correctly recognizes as fake. Combine this with the loss that the discriminator gets on real images and update the
networks based on this combined loss.

For some extra hints on implementing you can check out the following tensorflow implementation:

http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/

Good luck!