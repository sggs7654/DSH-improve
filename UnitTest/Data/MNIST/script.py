from Data.MNIST import MNIST

mnist = MNIST()
mnist.load_data()
print(mnist.point_set.shape)
mnist.build_test_set()
print(mnist.query_indices)
print(mnist.result_indices)