### Markov chain Monte Carlo simulation
### Ising model matrix calcation


import numpy as np
from numpy.random import rand
import torch

torch.set_default_tensor_type(torch.DoubleTensor)


class Simulater:
    """
    2차원 Ising 모델, 또는 SK 모델의 configuration을 MCMC로 생성합니다.

    system_size : 시스템 사이즈
    sample_size : 독립된 샘플 개수
    Temperature : 시스템의 온도
    equalibrium : 안정된 시스템 (평형상태)으로 도달하기 위한 MCMC-step 입력 => 경험적으로 선택
    sampling_interval : Markov chain은 t 번째 샘플과 t + delta t 샘플의 상관관계가 존재함 -> 긴 MCMC 간격으로 샘플링 해야함
    """
    def __init__(self, system_size, sample_size, Temperature, equalibrium, sampling_interval):
        self.N = system_size
        self.M = sample_size
        self.T = Temperature
        self.eq_T = equalibrium
        self.K = sampling_interval

        self.config = np.random.choice([-1.,1.] , size=self.N, replace = True)
        self.config = torch.from_numpy(self.config)
        self.samples = torch.zeros(self.N ,self.M,len(self.T))

    def adjacency_matrix(self, type):
        if type =='2D-Ising':
            L = int(np.sqrt(self.N))
            J = np.zeros((self.N,self.N))
            k = 0
            for a in range(self.N):
                for b in range(self.N):
                    i = a%L + ( b * L )%L**2 
                    j = (a+1)%L + ( b * L )%L**2 
                    J[i,j] = 1
                    J[j,i] = 1
                    k+=1
                    j = (a)%L + ( (b+1)*L )%L**2
                    J[i,j] = 1
                    J[j,i] = 1
                    k+=1
            
        elif type == 'SK-model':
            var = 1./np.sqrt(self.N/2) 
            H = np.random.normal(0., var, size = (self.N,self.N))
            for k in range(self.N):  H[k,k] = 0
            J = (H + H.T)/2
            
        else:
            print('Input 2D-Ising or SK-model')
            J = 0.
        self.J = torch.from_numpy(J)
        
        
    def move(self,t):

        for i in range(self.N):
            dE =  2.*self.J[i,:]@self.config* self.config[i]

            if rand()<torch.exp(-dE/t):
                self.config[i] *= -1
                
                
    def set_cuda(self):
        if torch.cuda.is_available():
            self.config = self.config.to('cuda')
            self.J = self.J.to('cuda')
            self.samples = self.samples.to('cuda')
        else:
            print('cuda is not avilable')

                
                
    def MC_sampling(self):
        """
        main processor
        """

        for t_idx,t in enumerate(self.T):
            for _ in range(self.eq_T): 
                self.move(self,t)

            for k_sample in range(self.M):                            ### cal T
                for _ in range(self.K):
                    self.move(self,t)
                self.samples[:,k_sample,t_idx] = self.config
        
        return self.samples


    def spin_autocorrelation(self,t):
        ## A(T) = < [ (s(t) - <s> ) ( s(t+T) - <s>) ] >
        A = torch.zeros(self.M -1)
        samples = torch.zeros(self.N , self.M)

        for _ in range(self.eq_T):
            self.move(self,t)

        for k in range(self.M):
            samples[:,k] = self.config
            self.move(self,t)

        ave_s = torch.mean(samples, 1)
        
        for Time_inter in range(self.M -1):
            
            for step in range(self.M - Time_inter):
                
                s_t = samples[:,step]

                s_t_T = samples[:,step + Time_inter]

                A[Time_inter] = torch.mean((s_t - ave_s) * (s_t_T - ave_s))


        return A/A[0]

    def mag_autocorrelation(self,t):
        ## A(T) = < [ (m(t) - <m> ) ( m(t+T) - <m>) ] >
        A = torch.zeros(self.M -1)
        samples = torch.zeros(self.N , self.M)

        for _ in range(self.eq_T):
            self.move(self,t)

        for k in range(self.M):
            samples[:,k] = self.config
            self.move(self,t)
        mag_array = abs(torch.mean(samples, 0))
        
        ave_m = torch.mean(mag_array)
        
        for Time_inter in range(self.M -1):
            
            for step in range(self.M - Time_inter):
                
                m_t = mag_array[step]

                m_t_T = mag_array[step + Time_inter]

                A[Time_inter] = (m_t - ave_m) * (m_t_T - ave_m)


        return A/A[0]

    def w_to_J(self,w):
        #A : parameter vector
        #L: number of node
        A = np.triu_indices(L,1)
        self.J = np.zeros((L,L))
        for i,(a,b) in enumerate(zip(A[0],A[1])):
            self.J[a,b] = w[i]
            self.J[b,a] = w[i]
