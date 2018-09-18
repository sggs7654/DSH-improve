from Screening.EDA import EDA
import numpy as np
import Screening.DSH_method
import Cluster.kmeans
import Data.GeneratedData
import Screening.preprocessor
import General.draw
from Data.MNIST import MNIST


def init_test():
    n = 5
    dim = 2
    w = np.zeros((dim,n))
    t = np.zeros(n)
    eda = EDA(w=w, t=t, centroids=1, weight=1, m=3)
    print(eda.population)


def select_test():
    gd = Data.GeneratedData.GeneratedData()
    cluster = Cluster.kmeans.Cluster(gd.point_set)
    p = Screening.preprocessor.Preprocessor(gd, cluster)
    eda = EDA(w=p.w, t=p.t, centroids=p.centroids, weight=p.weight, m=2)
    # eda.population = np.mat([[0,1],[1,2]])
    eda.select()

def search_test():
    gd = Data.GeneratedData.GeneratedData()
    cluster = Cluster.kmeans.Cluster(gd.point_set)
    p = Screening.preprocessor.Preprocessor(gd, cluster)
    eda = EDA(w=p.w, t=p.t, centroids=p.centroids, weight=p.weight, m=4)
    eda.search()
    draw = General.draw.draw()
    draw.new_screening(data=gd, w=p.w, t=p.t,
                       index_screening=eda.optimum_solution[len(eda.optimum_solution)-1])


def load_mnist_test():
    mnist = MNIST()
    mnist.load_data()
    cluster = Cluster.kmeans.Cluster(mnist.point_set)
    p = Screening.preprocessor.Preprocessor(mnist, cluster)
    eda = EDA(w=p.w, t=p.t, centroids=p.centroids, weight=p.weight, m=4)
    eda.search()
