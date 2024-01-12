__authors__ = ['1633753', '1633822', '1633937']
__group__ = 'DL.10'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        train_data = np.array(train_data)
        
        if train_data.dtype != np.float: #convertim a float si no ho és
            train_data = train_data.astype(float)
        
        #fem reshape
        P, M, N = train_data.shape
        D = M*N
        train_data = np.reshape(train_data, (P, D))
        
        self.train_data = train_data #actualitzem self.data amb les dimensions i tipus correctes
        

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        test_data = np.array(test_data)
        
        if test_data.dtype != np.float: #convertim a float si no ho és
            test_data = test_data.astype(float)
        
        #fem el reshape
        N, M, P = test_data.shape
        D = M*P
        test_data = np.reshape(test_data, (N, D))
        
       
        distances = cdist(test_data, self.train_data, metric='euclidean') #calculo la distancia
        
        indices = np.argsort(distances, axis=1)[:, :k] #trobo els indexs dels k veins més propers de cada fila
        
        self.neighbors = self.labels[indices] #guardem els labels dels k veins mes propers a self.neighbours

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        guardar_visitats = {}
        llista = []
        # lista_percent = []
        for i in self.neighbors:
           for j in range(len(i)):
               if i[j] not in guardar_visitats:
                   guardar_visitats[i[j]] = 1 #si està als visitats no 
               else:
                   guardar_visitats[i[j]] += 1
           llista.append(max(guardar_visitats, key=guardar_visitats.get))
           guardar_visitats = {} #reiniciem els visitats per la següent iteració
        return np.array(llista)  

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k) #aconseguim els k veins mes propers
        return self.get_class() #retornem les clases
