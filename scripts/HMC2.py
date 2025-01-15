
from numpy.core.numeric import Inf
import torch
import time
import numpy as np
torch.set_default_dtype(torch.double)

class Posterior_Density_Inference:
    def __init__(self, SDE_Model, negllik, u, theta_SDE, logsigmasq, u_KL_shape, theta_SDE_shape, sigma_shape, ind_non_obs, noisy_known=False, lsteps=50, epsilon=1e-5, n_samples=4000, upper_bound = None, lower_bound = None, burn_in_ratio = 0.1):
        all_theta=self.vectorize(u, theta_SDE, logsigmasq)
        print(all_theta.shape)
        self.ind_non_obs = ind_non_obs
        self.SDE_Model = SDE_Model
        self.all_theta = all_theta
        self.theta_SDE_shape = theta_SDE_shape

        self.u_KL_shape = u_KL_shape
        self.sigma_shape = sigma_shape
        self.lsteps = lsteps
        self.epsilon = epsilon * torch.ones(all_theta.shape)
        self.burn_in_ratio = burn_in_ratio
        self.n_samples = n_samples
        self.total_samples = int(n_samples / (1 - burn_in_ratio))
        self.Minus_Log_Posterior = negllik
        self.ub = upper_bound
        if upper_bound is not None:
            if upper_bound.shape[0] != all_theta.shape[0]:
                raise ValueError
        self.lb = lower_bound
        if lower_bound is not None:
            if lower_bound.shape[0] != all_theta.shape[0]:
                raise ValueError
        self.noisy_known=noisy_known

    def vectorize(self, u, theta_SDE, logsigmasq=None):
        t1 = torch.reshape(u.detach(), (-1,))
        t2 = torch.reshape(theta_SDE.detach(), (-1,))

        if logsigmasq is not None:
            t5 = torch.reshape(logsigmasq.detach(), (-1,))
        else:
            t5=torch.tensor(torch.ones(0))
        long_vec = torch.cat((t1, t2))
        return long_vec

    def get_dim(self, tensor_shape):
        if len(tensor_shape) == 0:
            return 1
        if len(tensor_shape) == 1:
            return tensor_shape[0]
        dim = 1
        for i in range(len(tensor_shape)):
            dim *= tensor_shape[i]
        return dim

    def devectorize(self, long_tensor, u_shape, theta_SDE_shape, sigma_shape = None):
        u_dim = self.get_dim(u_shape)
        theta_SDE_dim = self.get_dim(theta_SDE_shape)
        if sigma_shape is not None: sigma_dim = self.get_dim(sigma_shape)
        xlatent = torch.reshape(long_tensor[:u_dim],u_shape)
        theta_SDE = torch.reshape(long_tensor[u_dim : u_dim + theta_SDE_dim],theta_SDE_shape)
        return xlatent, theta_SDE

    def Minus_Log_Posterior_vec(self, all_theta):
        xlatent_0, theta_SDE = self.devectorize(all_theta, self.u_KL_shape, self.theta_SDE_shape, self.sigma_shape)
        return self.Minus_Log_Posterior(theta_SDE, xlatent_0, self.SDE_Model.x_I)

    def Nabla(self, theta_torch):
        theta_torch = theta_torch.detach()
        xlatent, theta_SDE= self.devectorize(theta_torch, self.u_KL_shape, self.theta_SDE_shape, self.sigma_shape)
        xlatent.requires_grad = True  
        theta_SDE.requires_grad = True
        llik = self.Minus_Log_Posterior(theta_SDE, xlatent, self.SDE_Model.x_I)
        llik.backward() 
        v = self.vectorize(xlatent.grad, theta_SDE.grad)
        v = torch.max(torch.min(v,torch.tensor(1e10)),torch.tensor(-1e10))
        #print(v)
        return v

    def Sampling(self):
        def bounce(m, lb, ub):
            if lb is None and ub is None:
                return m
            if lb is None:
                max_tensor = torch.clamp(m - ub, min=0)
                return m - 2 * max_tensor
            if ub is None:
                min_tensor = torch.clamp(lb - m, min=0)
                return m + 2 * min_tensor
            if torch.sum(lb < ub) < m.shape[0]:
                raise ValueError
            if torch.sum(m >= lb) == m.shape[0] and torch.sum(m <= ub) == m.shape[0]:
                return m
            if torch.sum(m >= lb) < m.shape[0]:
                min_tensor = torch.clamp(lb - m, min=0)
                return bounce(m + 2 * min_tensor, lb, ub)
            if torch.sum(m <= ub) < m.shape[0]:
                max_tensor = torch.clamp(m - ub, min=0)
                return bounce(m - 2 * max_tensor, lb, ub)

        log_post_density_trace = np.zeros(self.total_samples)
        samples = np.zeros((self.total_samples, self.all_theta.shape[0]))
        samples_u = torch.zeros((self.total_samples, self.u_KL_shape[0],self.u_KL_shape[1]))
        samples_theta_sde = torch.zeros((self.total_samples, self.theta_SDE_shape[0]))
        random_ls = np.random.uniform(0, 1, self.total_samples)
        acceptance_ls = np.zeros(self.total_samples)
        nan_ls = np.zeros(self.total_samples)
        theta_ini = self.all_theta.clone().detach()
        cur_theta = self.all_theta.clone().detach()
        a0=time.time()
        for EachIter in range(self.total_samples):
            #cur_nllik_1 = self.Minus_Log_Posterior_vec(cur_theta).detach()
            rstep = torch.rand(self.epsilon.shape) * self.epsilon + self.epsilon
            p = torch.normal(mean=0., std=torch.ones(self.all_theta.shape))
            cur_p = p.clone()
            theta = cur_theta.clone()         
            p = p - rstep * self.Nabla(theta).clone() / 2
            for i in range(self.lsteps):
                p[self.SDE_Model.ind_obs] = torch.zeros(len(self.SDE_Model.ind_obs))
                theta[self.SDE_Model.ind_obs] = theta_ini[self.SDE_Model.ind_obs]

                theta = theta + rstep * p
                p = p - rstep * self.Nabla(theta).clone()
                theta = bounce(theta, self.lb, self.ub)

            p = p - rstep * self.Nabla(theta).clone() / 2

            p[self.SDE_Model.ind_obs] = torch.zeros(len(self.SDE_Model.ind_obs))
            theta[self.SDE_Model.ind_obs] = theta_ini[self.SDE_Model.ind_obs]

            new_nllik = self.Minus_Log_Posterior_vec(theta)
            new_p = 0.5 * torch.sum(torch.square(p))
            new_H = new_nllik + new_p
            cur_nllik = self.Minus_Log_Posterior_vec(cur_theta).detach()
            cur_H = cur_nllik + 0.5 * torch.sum(torch.square(cur_p))

            if torch.isnan(theta[0]) or torch.isnan(new_H):
                samples[EachIter] = cur_theta.clone()
                nan_ls[EachIter] = 1
                self.epsilon *= 0.9
                print('NaN!')
            else:
                # accept
                tmp = float(torch.exp(cur_H - new_H))
                if  tmp > random_ls[EachIter]:
                    samples[EachIter] = theta.clone()
                    cur_theta = theta.clone()
                    acceptance_ls[EachIter] = 1
                    # rejected
                else:
                    samples[EachIter] = cur_theta.clone()
                    # accepted

            log_post_density_trace[EachIter] = - self.Minus_Log_Posterior_vec(cur_theta).item()        
            samples_u[EachIter], samples_theta_sde[EachIter]= self.devectorize(cur_theta.clone(), self.u_KL_shape, self.theta_SDE_shape, self.sigma_shape)
            print(samples_theta_sde[EachIter])

            if EachIter > 200 and EachIter < self.total_samples - self.n_samples:
                if np.sum(acceptance_ls[EachIter - 100 : EachIter]) < 60:
                    # decrease epsilon
                    self.epsilon *= 0.995
                if np.sum(acceptance_ls[EachIter - 100 : EachIter]) > 90:
                    # increase epsilon
                    self.epsilon *= 1.005
            if EachIter % 100 == 0 and EachIter > 100:
                #print(EachIter)
                #print(cur_nllik)
                b0=time.time()
                print('ite:', EachIter, 'accept: ', np.sum(acceptance_ls[EachIter - 100 : EachIter]) / 100, 'time:', (b0-a0) / 60, 'mins')
                if EachIter < self.total_samples - self.n_samples:
                    standard_deviation = torch.tensor(np.std(samples[EachIter - 100:EachIter, :], axis = 0))
                    if torch.mean(standard_deviation) > 1e-6:
                        self.epsilon = 0.05 * standard_deviation * torch.mean(self.epsilon) / torch.mean(standard_deviation) + 0.95 * self.epsilon
        post_mean_all=np.mean(samples,0)
        u_mean, theta_SDE = self.devectorize(torch.tensor(post_mean_all), self.u_KL_shape, self.theta_SDE_shape, self.sigma_shape)
        HMC_sample = {
            'samples': samples,
            'acceptance_rate': acceptance_ls,
            'posterior_mean' : post_mean_all,
            'posterior_mean_u_KL' : u_mean,
            'posterior_mean_theta_SDE' : theta_SDE,
            'log_post_density_trace' : log_post_density_trace,
            'nan_ls' : nan_ls,
            'time': b0-a0
        }
        return (HMC_sample)