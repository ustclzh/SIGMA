import numpy as np
import scipy.io as scio
import scipy
import torch 
import random
from scipy.stats import qmc
import math
import time

#import mkl

# Set the number of threads to 1
#mkl.set_num_threads(1)
from . import GP_processing
torch.set_default_dtype(torch.double)

class SDE_Module(object):
    def __init__(
        self, 
        para_theta=None, 
        sde_operator=1, 
        sigma_e=0.001, 
        n_I = 20,
        n_obs = 20,
        noisy=True, 
        noisy_known = False, 
        optimize_alg = True,
        band_width = 1.
        ):
        self.sde_operator=sde_operator
        self.para_theta=para_theta # identify initial gauss for parameter
        self.noisy_known=noisy_known
        self.sigma_e=torch.tensor(sigma_e)
        self.sigma_e = self.sigma_e.reshape(-1,1)
        self.optimize_alg= optimize_alg
        self.T_terminal = 1
        self._Load_Data(n_I, n_obs)
        self._GP_Preprocess(noisy=noisy, noisy_known=noisy_known,band_width = band_width)

    def _Load_Data(self, n_I, n_obs):
        # the operator is \partial/\partial x_1 +\partial/\partial x_2   
        if self.sde_operator == -4 :
            self.T_terminal = 10
        elif self.sde_operator == 5:
            self.T_terminal = 10
        self.x_I = torch.linspace(0, self.T_terminal , n_I + 1)
        self.x_I = self.x_I.reshape(-1,1)   
        self.n_I = self.x_I.shape[0]
        self.d = 1
        self.y_I=self.True_Solution(self.x_I)
        self.p = self.y_I.shape[1]
        self.u_true =self.y_I 
        gap = int(n_I/n_obs)
        ind_obs = list(range(0,n_I+1,gap))
        self.ind_non_obs = filter(lambda x: x%gap !=0, list(range(0,n_I+1)))
        self.ind_obs = ind_obs
        self.x_obs = self.x_I[ind_obs,:]
        self.y_obs = self.y_I[ind_obs,:]
        self.y_obs = self.y_obs + self.sigma_e * torch.randn(self.y_obs.shape)
        self.sigma_e = self.sigma_e * torch.ones(1,self.p)
        self.n_obs = self.y_obs.shape[0]
        if self.para_theta is None :
            self.para_theta=self.theta_true *(1 + torch.rand(self.theta_true.shape)/10)
    def Drift(self, t_input, u, para_theta, delta_t = None):
        n = t_input.shape[0]
        if delta_t is None : delta_t = t_input[1:n]-t_input[0:-1]
        if (self.sde_operator==0): 
            f =  para_theta[0] * delta_t
            sigma1 = para_theta[1] * torch.ones(f.shape)*torch.sqrt(delta_t)
        elif (self.sde_operator==2):
            f =  para_theta[0] * ( para_theta[1] - u[0:-1]) * delta_t
            sigma1 = para_theta[2] * torch.ones(f.shape)*torch.sqrt(delta_t)
        elif (self.sde_operator==3):
            f =  para_theta[0]  * delta_t / u[0:-1]
            sigma1 = para_theta[1] * torch.ones(f.shape)*torch.sqrt(delta_t)
        elif (self.sde_operator==4):
            f =  (para_theta[0] +  para_theta[1] * u[0:-1] + para_theta[2] / u[0:-1] + para_theta[3] * u[0:-1] ** 2 ) *delta_t
            sigma1 = para_theta[4] *  u[0:-1] ** para_theta[5] * torch.sqrt(delta_t)
        elif (self.sde_operator==5):
            f =  4 * delta_t * u[0:-1] * (para_theta[0] -  u[0:-1]**2)
            sigma1 = para_theta[1] * torch.ones(f.shape) * torch.sqrt(delta_t)
        elif (self.sde_operator==-1):
            f =  para_theta[0] * u[0:-1] * delta_t
            sigma1 = para_theta[1]*u[0:-1]*torch.sqrt(delta_t)
        elif (self.sde_operator==-2):
            f =  para_theta[0] * ( para_theta[1] - u[0:-1] ) * delta_t
            sigma1 = para_theta[2]*u[0:-1]*torch.sqrt(delta_t)
        elif (self.sde_operator==-3):
            f2 =  para_theta[0] * (para_theta[1] - u[0:-1,1] ) * delta_t[0]
            f1 = torch.zeros(f2.shape[0])
            f = torch.cat((f1.reshape(-1,1),f2.reshape(-1,1)),1)
            sigma1 = torch.sqrt(u[0:-1,1] * delta_t[0])
            sigma2 = para_theta[2] * u[0:-1,1] * torch.sqrt(delta_t[0])
            sigma1 = torch.cat((sigma1.reshape(-1,1),sigma2.reshape(-1,1)),1)
        elif (self.sde_operator==-4):
            f1 = ( para_theta[0] * u[0:-1,0] - para_theta[1] * u[0:-1,0] * u[0:-1,1]) * delta_t[0]
            f2 = (-para_theta[2] * u[0:-1,1] + para_theta[1] * u[0:-1,0] * u[0:-1,1]) * delta_t[0]
            f = torch.cat((f1.reshape(-1,1),f2.reshape(-1,1)),1)
            sigma1 = torch.empty((u.shape[0]-1,2))
            sigma1[:,0] = para_theta[3]*u[0:-1,0]*torch.sqrt(delta_t[0])
            sigma1[:,1] = para_theta[4]*u[0:-1,1]*torch.sqrt(delta_t[0])

        return(f,sigma1)
     
    def Source(self, t_input, u, para_theta):
        n = t_input.shape[0]
        delta_t = (t_input[1:n]-t_input[0:-1])
        f, sigma = self.Drift(t_input, u, para_theta, delta_t)
        f = u[0:-1] + f
        return(f,sigma)
    
    def Source_c(self, t_input, u, para_theta):
        n = t_input.shape[0]
        delta_t = (t_input[1:n]-t_input[0:-1]) / 2
        f, sigma = self.Drift(t_input, u, para_theta, delta_t)
        f = u[0:-1] + f
        return(f,sigma/np.sqrt(2))
    
    def True_Solution(self, t_input): # for toy examples, where the true function/PDE solution is known analytically. 
        if (self.sde_operator==0): # Brownian Motion
            self.theta_true=torch.tensor([2.,0.4])
            self.y_ini = torch.tensor([0])
            
            x = torch.linspace(0,self.T_terminal,1001)
            n = x.shape[0]
            y = torch.zeros(n)
            y[0] = self.y_ini[0]
            for i in range (n-1):
                y[i+1] = y[i] + self.theta_true[0] * (x[i+1]-x[i]) + torch.randn(1)*torch.sqrt(x[i+1]-x[i])*self.theta_true[1]
            self.y_all = y
            ind = t_input * 1000 / self.T_terminal
            ind = ind.squeeze()
            ind = ind.int().numpy()
            y = y[ind]
            y = y.reshape(-1,1)
            

        elif (self.sde_operator==2): # OU process
            self.theta_true=torch.tensor([3.,2.,0.2])
            self.y_ini = torch.tensor([1])

            x = torch.linspace(0,self.T_terminal,1001)
            n = x.shape[0]
            y = torch.zeros(n)
            y[0] = self.y_ini[0]
            for i in range (n-1):
                y[i+1] = y[i] + self.theta_true[0] *(self.theta_true[1] - y[i]) * (x[i+1]-x[i])  + torch.randn(1)*torch.sqrt(x[i+1]-x[i])*self.theta_true[2]
            self.y_all = y
            ind = t_input * 1000 / self.T_terminal
            ind = ind.squeeze()
            ind = ind.int().numpy()
            y = y[ind]
            y = y.reshape(-1,1)
        elif (self.sde_operator==3): # Bessel process
            self.theta_true=torch.tensor([5.,0.5])
            self.y_ini = torch.tensor([1])

            x = torch.linspace(0,self.T_terminal,1001)
            n = x.shape[0]
            y = torch.zeros(n)
            y[0] = self.y_ini[0]
            for i in range (n-1):
                y[i+1] = y[i] + self.theta_true[0] *  (x[i+1]-x[i]) / y[i]   + torch.randn(1)*torch.sqrt(x[i+1]-x[i])*self.theta_true[1]
            self.y_all = y
            ind = t_input * 1000 / self.T_terminal
            ind = ind.squeeze()
            ind = ind.int().numpy()
            y = y[ind]
            y = y.reshape(-1,1)
        elif (self.sde_operator==4):
            self.theta_true=torch.tensor([0.,1.,0.,0.,0.1,1.])
            self.y_ini = torch.tensor([1.])

            x = torch.linspace(0,self.T_terminal,1001)
            n = x.shape[0]
            y = torch.zeros(n)
            y[0] = self.y_ini[0]
            for i in range (n-1):
                y[i+1] = y[i] + (self.theta_true[0] +  self.theta_true[1] * y[i] + self.theta_true[2] / y[i] + self.theta_true[3] * y[i] ** 2 ) * (x[i+1]-x[i])   + torch.randn(1)* self.theta_true[4] *  y[i] ** self.theta_true[5] * torch.sqrt(x[i+1]-x[i])
            self.y_all = y
            ind = t_input * 1000 / self.T_terminal
            ind = ind.squeeze()
            ind = ind.int().numpy()
            y = y[ind]
            y = y.reshape(-1,1)
        elif (self.sde_operator==5):
            self.theta_true=torch.tensor([2., 1.])
            self.y_ini = torch.tensor([1.])
            x = torch.linspace(0,self.T_terminal,1001)
            n = x.shape[0]
            y = torch.zeros(n)
            y[0] = self.y_ini[0]
            for i in range (n-1):
                y[i+1] = y[i] + 4 * y[i] * (self.theta_true[0] -  y[i]**2) * (x[i+1]-x[i])   + torch.randn(1)* self.theta_true[1] * torch.sqrt(x[i+1]-x[i])
            self.y_all = y
            ind = t_input * 1000 / self.T_terminal
            ind = ind.squeeze()
            ind = ind.int().numpy()
            y = y[ind]
            y = y.reshape(-1,1)


        elif (self.sde_operator==-1): # Geometric Bronian motion
            self.theta_true=torch.tensor([2.,0.2])
            self.y_ini = torch.tensor([1.])

            t = torch.linspace(0,self.T_terminal,1001)
            n = t.shape[0]
            y = torch.zeros(n)
            y[0] = 0.
            y_simu = torch.zeros(n)
            y_simu[0] = self.y_ini[0]
            for i in range (n-1):
                y[i+1] = y[i] + torch.randn(1) * torch.sqrt(t[i+1]-t[i])
                y_simu[i+1] = y_simu[i] + self.theta_true[0] * y_simu[i] * (t[i+1]-t[i])  + torch.randn(1) * torch.sqrt(t[i+1]-t[i]) * self.theta_true[1] * y_simu[i]
            y = self.y_ini[0] *  torch.exp((self.theta_true[0]-self.theta_true[1]**2/2)*t + self.theta_true[1] * y)
            self.y_simu = y_simu
            self.y_all = y
            ind = t_input * 1000 / self.T_terminal
            ind = ind.squeeze()
            ind = ind.int().numpy()
            y = y[ind]
            y = y.reshape(-1,1)

        elif (self.sde_operator==-2): # Garch
            self.theta_true=torch.tensor([2.,2.,0.2])
            self.y_ini = torch.tensor([1.])

            x = torch.linspace(0,self.T_terminal,1001)
            n = x.shape[0]
            y = torch.zeros(n)
            y[0] = self.y_ini[0]
            for i in range (n-1):
                y[i+1] = y[i] + self.theta_true[0] *(self.theta_true[1] - y[i]) * (x[i+1]-x[i])  + torch.randn(1)*torch.sqrt(x[i+1]-x[i])*self.theta_true[2] * y[i]
            self.y_all = y
            ind = t_input * 1000 / self.T_terminal
            ind = ind.squeeze()
            ind = ind.int().numpy()
            y = y[ind]
            y = y.reshape(-1,1)

        elif (self.sde_operator==-3): # multi-d Grarch
            self.theta_true=torch.tensor([2.,2.,0.2])
            self.y_ini = torch.tensor([0.,1.])

            x = torch.linspace(0,self.T_terminal,1001)
            n = x.shape[0]
            y = torch.zeros(n,2)
            y[0,0] = self.y_ini[0]
            y[0,1] = self.y_ini[1]
            for i in range (n-1):
                y[i+1,0] = y[i,0] + torch.randn(1)*torch.sqrt(y[i,1]*(x[i+1]-x[i]))
                y[i+1,1] = y[i,1] + self.theta_true[0] *(self.theta_true[1] - y[i,1]) * (x[i+1]-x[i])  + torch.randn(1)*y[i,1]*torch.sqrt(x[i+1]-x[i])*self.theta_true[2]
            self.y_all = y
            ind = t_input * 1000 / self.T_terminal
            ind = ind.squeeze()
            ind = ind.int().numpy()
            y = y[ind,:] 
        elif (self.sde_operator==-4): # F-V model
            self.theta_true=torch.tensor([0.5,0.0025,0.3,0.1,0.1])
            self.y_ini = torch.tensor([41.,9.])
            self.T_terminal = 10

            x = torch.linspace(0,self.T_terminal,10001)
            n = x.shape[0]
            y = torch.zeros(n,2)
            y[0,0] = self.y_ini[0]
            y[0,1] = self.y_ini[1]
            for i in range (n-1):
                b = torch.randn(2)
                Sig = torch.empty((2))
                Sig[0] = self.theta_true[3]*y[i,0]
                Sig[1] = self.theta_true[4]*y[i,1]
                drift = Sig * b
                #print(drift)
                y[i+1,0] = y[i,0] + (self.theta_true[0] * y[i,0] - self.theta_true[1] * y[i,0] * y[i,1])* (x[i+1]-x[i])+ drift[0]*torch.sqrt((x[i+1]-x[i]))
                y[i+1,1] = y[i,1] + ( - self.theta_true[2] * y[i,1] + self.theta_true[1] * y[i,0] * y[i,1])* (x[i+1]-x[i]) + drift[1]*torch.sqrt((x[i+1]-x[i]))
            self.y_all = y
            ind = t_input * 10000 / self.T_terminal
            ind = ind.squeeze()
            ind = ind.int().numpy()
            y = y[ind,:] 
        return (y)



    # def Jacobian(self):
    #     j = torch.gradient

    def True_Solution_no_err(self, t_input, theta): # for toy examples, where the true function/PDE solution is known analytically. 
        if (self.sde_operator==4):
            n = t_input.shape[0]
            y = torch.zeros(n)
            y[0] = 1
            for i in range (n-1):
                y[i+1] = y[i] + (theta[0] +  theta[1] * y[i] + theta[2] / y[i] + theta[3] * y[i] ** 2 ) * (t_input[i+1]-t_input[i])
            #y = torch.min(y, torch.tensor(10.))
        return (y)


    def _GP_Preprocess(self, noisy, noisy_known, band_width = 1.):
        '''
        GP preprocessing of the Input Data
        '''
        self.GP_Components = []
        self.GP_Models=[]
        for i in range(self.p):
            # available observation index
            self.nu = 0.5
            GP_model=GP_processing.GP_modeling(self, noisy = noisy, nu=self.nu, noisy_known=noisy_known)
            if self.n_obs> 100 * self.x_obs.shape[1]:
                subsample_ind = random.sample(range(self.n_obs),100*self.x_obs.shape[1])
            else:
                subsample_ind = random.sample(range(self.n_obs),self.n_obs)
            GP_model.Train_GP(self, self.x_obs[subsample_ind], self.y_obs[subsample_ind,i],ind_aug = 0, ind_y = i, optimize_alg = self.optimize_alg, band_width = band_width)
            self.GP_Components.append({
                #'aIdx':aIdx, # non-missing data index
                'mean':GP_model.mean,
                'kernel':GP_model.kernel,
                'outputscale':GP_model.outputscale,
                'noisescale':GP_model.noisescale
            })
            self.GP_Models.append(GP_model)
        