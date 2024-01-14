"""
reprodued by Min Jae Jung
(original paper: Erasure machine: Inverse Ising inference from reweighting of observation frequencies)
"""

import numpy as np
from numpy.random import rand
import time
import torch


#=========================================================================================
torch.set_default_tensor_type(torch.DoubleTensor)

class ErasureMachine:
    def __init__(self, learning_rate = 0.1, max_iter = 151, device = 'cuda'):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.device = 'cpu'
        if torch.cuda.is_available() and device == 'cuda':
            print('Use cuda')
            self.device = 'cuda'
        
    def operators(self, configuration):
        #O = <S_i S_j>
        system_size,sample_size = configuration.shape
        Op = np.zeros((int(system_size*(system_size-1)/2), sample_size))

        jindex = 0
        for index in range(system_size-1):
            temp = configuration[index,:] * configuration[index+1:,:]
            Op[jindex: jindex + system_size - index -1 , :] = temp
            jindex += system_size - index - 1
        return Op
        
           
    def forward(self, configuration, epsilon = 0.7,  L1 = 0. , L2 = 0.):
        """
        ops : observed spin configuration
        L1 : L1 regularization
        L2 : L2 regularizarion

        Return : spin adjacenct matrix (model parameter)
        """
        ops = self.operators(configuration)
        n_ops = ops.shape[0]
        np.random.seed(13)
        w = np.random.rand(n_ops)-0.5 # random initialization

        if self.device == 'cuda':
            w = torch.from_numpy(w).cuda()
            ops = torch.from_numpy(ops).cuda()
        
        for i in range(self.max_iter):

            energies_w = w@ops

            probs_w = torch.exp((energies_w)*(epsilon-1)) 
            z_data = torch.sum(probs_w)
            probs_w /= z_data
            ops_expect_w = torch.sum(probs_w[np.newaxis,:]*ops,axis=1)


            w += self.learning_rate*(ops_expect_w - w*epsilon - L2*w - L1*w /abs(w) )

        return w

    def J_to_w(self,J):
        # adjacency matrix -> parameters 
        N = len(J)
        iu1 = np.triu_indices(N,1)
        w = J[iu1]
        return w

