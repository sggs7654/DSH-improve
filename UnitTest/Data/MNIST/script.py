from Data.MNIST import MNIST

mnist = MNIST()
mnist.load_data()
print(mnist.point_set.shape)