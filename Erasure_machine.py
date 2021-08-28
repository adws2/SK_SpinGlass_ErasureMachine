import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
import time
import torch


#=========================================================================================
torch.set_default_tensor_type(torch.DoubleTensor)

class fit():
    def __init__(self,epsilon = 0.7, learning_rate = 0.1, max_iter = 151,regularizer = 0.1):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularizer = regularizer
        
        
    def operators(self,configuration):
        #O = <S_i S_j>
        system_size,sample_size = configuration.shape
        Op = np.zeros((int(system_size*(system_size-1)/2), sample_size))

        jindex = 0
        for index in range(system_size-1):
            temp = configuration[index,:] * configuration[index+1:,:]
            Op[jindex: jindex + system_size - index -1 , :] = temp
            jindex += system_size - index - 1
        return Op
        
        
    def em_cpu(self, ops):
        n_ops = ops.shape[0]
        np.random.seed(13)
        w = np.random.rand(n_ops)-0.5
        for i in range(self.max_iter):

            energies_w = w@ops

            energies_max = energies_w.max()  
            probs_w = np.exp((energies_w-energies_max)*(self.epsilon-1)) 
            z_data = np.sum(probs_w)
            probs_w /= z_data
            ops_expect_w = np.sum(probs_w[np.newaxis,:]*ops,axis=1)


            w += self.learning_rate*(ops_expect_w - w*self.epsilon - self.regularizer*w)

        return w
    
    def em_gpu(self, ops):
        n_ops = ops.shape[0]
        np.random.seed(13)
        w = np.random.rand(n_ops)-0.5
        w = torch.from_numpy(w).cuda()
        ops = torch.from_numpy(ops).cuda()
        
        for i in range(self.max_iter):

            energies_w = w@ops

            probs_w = torch.exp((energies_w)*(self.epsilon-1)) 
            z_data = torch.sum(probs_w)
            probs_w /= z_data
            ops_expect_w = torch.sum(probs_w[np.newaxis,:]*ops,axis=1)


            w += self.learning_rate*(ops_expect_w - w*self.epsilon - self.regularizer*w)

        return w

    def J_to_w(self,J):
        # adjacency matrix -> parameters 
        N = len(J)
        iu1 = np.triu_indices(N,1)
        w = J[iu1]
        return w

