from numpy import inf
import torch
import time
import numpy as np
import random
import scipy.stats
from scipy.stats import qmc
from scipy.optimize import minimize
from . import GP_processing
from . import HMC
from . import HMC2
torch.set_default_dtype(torch.double)

#import mkl

# Set the number of threads to 1
#mkl.set_num_threads(1)
class SIGMA(object):
    def __init__(self, SDE_Module, para = False, rho = 1., num_sem = 1):
        self.ind_non_obs = SDE_Module.ind_non_obs
        self.SDE_Module = SDE_Module
        self.sde_operator=SDE_Module.sde_operator
        self.noisy_known=SDE_Module.noisy_known
        self.x_I=SDE_Module.x_I
        self.n_I, self.d = self.x_I.size()
        self.x_I_c = self.x_I[1:self.n_I] - (self.x_I[1]-self.x_I[0])/2
        self.y_obs = SDE_Module.y_obs
        self.n_obs, self.p = self.y_obs.size()
        self.x_obs = SDE_Module.x_obs
        self.ind_obs = SDE_Module.ind_obs
        self.GP_Components=SDE_Module.GP_Components
        self.GP_Models=SDE_Module.GP_Models
        self.optimize_alg = SDE_Module.optimize_alg
        self.rho = rho
        self.para = para
        self.num_sem = num_sem 

    def mle(self):
        # mle 
        d_theta = self.SDE_Module.para_theta.shape[0]
        self.d_theta = d_theta
        current_opt = np.Inf
        current_opt_a = np.Inf
        theta_MLE = self.SDE_Module.para_theta.clone()
        theta_MLE_a = self.SDE_Module.para_theta.clone()
        sampler = qmc.Halton(d=d_theta, scramble=False)
        theta_cand = sampler.random(n=d_theta*5)
        #theta_cand=lhs(d_theta, samples=d_theta*5, criterion='maximin')
        for ini in range(d_theta*5):
            bnds=((0,10),) * d_theta             
            res = minimize(self.minus_log_likelihood_p,(torch.tensor(theta_cand[ini,:])), args=(self.y_obs,self.x_obs), method='Nelder-Mead', bounds=bnds)
            res_a = minimize(self.minus_log_likelihood_a,(torch.tensor(theta_cand[ini,:])), args=(self.y_obs,self.x_obs), method='Nelder-Mead', bounds=bnds)
            if res['fun']<current_opt: 
                theta_MLE=res['x']
                current_opt=res['fun']
            if res_a['fun']<current_opt_a: 
                theta_MLE_a=res_a['x']
                current_opt_a=res_a['fun']
        self.theta_MLE = theta_MLE
        self.theta_MLE_a = theta_MLE_a
        print('MLE:',theta_MLE, 'MLE_analytical',theta_MLE_a,'True:', self.SDE_Module.theta_true)

    def map(self, nEpoch = 2500, opt_algorithm = 2, center_modify = False):
        self.mle()
        #setup initial values
        self.opt_algorithm = opt_algorithm
        u, theta_GP_mu= self._Pre_Process()
        u=u.requires_grad_()
        theta_SDE = torch.tensor(self.theta_MLE).requires_grad_()
        theta_GP_mu =  theta_GP_mu.requires_grad_()
        theta_GP_sigma = torch.tensor([1.]).requires_grad_()
        if self.sde_operator == -4: theta_GP_sigma = torch.tensor([1.,1.]).requires_grad_()
        lognoisescale = torch.zeros(self.p)
        for i in range(self.p):
            lognoisescale[i]=torch.log(self.GP_Components[i]['noisescale'].double())

        #setup optimization
        u_lr = 1e-2 * (self.sde_operator < 0 ) + 1e-1 * (self.sde_operator >= 0 )
        if self.sde_operator == 4 : theta_GP_sigma = torch.tensor([1.,0.01]).requires_grad_()
        time0 = time.time()
        obj_loss = self.Minus_Log_Posterior
        if center_modify : 
            obj_loss = self.Minus_Log_Posterior_c
            print('do center modification')

        if self.opt_algorithm == 1:
            u_lr = u_lr * 10
            if self.noisy_known is False  :
                lognoisescale=lognoisescale.requires_grad_()
                self.optimizer_u_theta = torch.optim.LBFGS([u,theta_SDE, theta_GP_mu, theta_GP_sigma,lognoisescale], lr = u_lr)
            else :
                self.optimizer_u_theta = torch.optim.LBFGS([u,theta_SDE, theta_GP_mu, theta_GP_sigma], lr = u_lr)
                lognoisescale_opt=lognoisescale
            pointwise_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_u_theta, lr_lambda = lambda epoch: 1/((epoch+1)**0.5), last_epoch=-1)
            print('start optimiza theta and u:')
            def closure():
                self.optimizer_u_theta.zero_grad()
                loss = obj_loss(u, theta_SDE, theta_GP_mu, theta_GP_sigma, lognoisescale / 2)
                loss.backward()
                return loss
            for epoch in range(nEpoch):
                self.optimizer_u_theta.zero_grad()
                loss_u_theta = obj_loss(u, theta_SDE, theta_GP_mu, theta_GP_sigma, lognoisescale / 2)
                if epoch==0:
                    loss_u_theta_opt=loss_u_theta.clone().detach()
                    u_opt=u.clone().detach()
                    theta_opt=theta_SDE.clone().detach()
                    if self.noisy_known is False : lognoisescale_opt=lognoisescale.clone().detach()
                else:
                    #if para_theta[0]<0: para_theta[0] = torch.abs(para_theta[0])
                    if loss_u_theta<loss_u_theta_opt:
                        loss_u_theta_opt=loss_u_theta.clone().detach()
                        u_opt=u.clone().detach()
                        theta_opt=theta_SDE.clone().detach()
                        if self.noisy_known is False : lognoisescale_opt=lognoisescale.clone().detach()
                loss_u_theta.backward()
                self.optimizer_u_theta.step(closure)
                pointwise_lr_scheduler.step()
                if (np.isnan(obj_loss(u, theta_SDE, theta_GP_mu, theta_GP_sigma,lognoisescale / 2).detach().numpy())):
                    u = u_opt
                    theta_SDE = theta_opt
                    if self.noisy_known is False : lognoisescale=lognoisescale_opt
                if (epoch+1) % 500 == 0 :
                    print(epoch+1, '/', nEpoch, 'current opt: theta:', theta_opt.numpy(),'error/out_scale', torch.exp(lognoisescale_opt).clone().detach().numpy()/self.GP_Components[0]['outputscale'])
                    #print('gradient', torch.mean(torch.abs(u.grad.squeeze())).numpy(), theta_SDE.grad.numpy())
            u.requires_grad_(False)
            theta_SDE.requires_grad_(False)
            theta_GP_mu.requires_grad_(False)
            theta_GP_sigma.requires_grad_(False)            
            lognoisescale.requires_grad_(False)
        
        elif self.opt_algorithm == 2:
            u_lr = u_lr/10
            torch.autograd.set_detect_anomaly(True)
            if self.noisy_known is False  :
                lognoisescale=lognoisescale.requires_grad_()
                optimizer_u_theta = torch.optim.Adam([u,theta_SDE, theta_GP_mu, theta_GP_sigma,lognoisescale], lr = u_lr)
            else :
                optimizer_u_theta = torch.optim.Adam([u,theta_SDE, theta_GP_mu, theta_GP_sigma,], lr = u_lr)
                lognoisescale_opt=lognoisescale
            pointwise_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_u_theta, lr_lambda = lambda epoch: 1/((epoch+1)**0.2), last_epoch=-1)
            print('start optimiza theta and u:')
            for epoch in range(nEpoch):
                optimizer_u_theta.zero_grad()
                loss_u_theta = obj_loss(u, theta_SDE, theta_GP_mu, theta_GP_sigma, lognoisescale / 2)
                if epoch == 0:
                    loss_u_theta_opt = loss_u_theta.clone().detach()
                    u_opt = u.clone().detach()
                    theta_opt = theta_SDE.clone().detach()
                    if self.noisy_known is False : lognoisescale_opt = lognoisescale.clone().detach()
                else:
                    #if para_theta[0]<0: para_theta[0] = torch.abs(para_theta[0])
                    if loss_u_theta<loss_u_theta_opt:
                        loss_u_theta_opt=loss_u_theta.clone().detach()
                        u_opt=u.clone().detach()
                        theta_opt=theta_SDE.clone().detach()
                        if self.noisy_known is False : lognoisescale_opt = lognoisescale.clone().detach()
                loss_u_theta.backward()
                optimizer_u_theta.step()
                pointwise_lr_scheduler.step()
                if (np.isnan(obj_loss(u, theta_SDE, theta_GP_mu, theta_GP_sigma, lognoisescale / 2).detach().numpy())):
                    u = u_opt
                    theta_SDE = theta_opt
                    if self.noisy_known is False : lognoisescale = lognoisescale_opt
                if (epoch+1) % 500 == 0 :
                    print(epoch+1, '/', nEpoch, 'current opt: theta:', theta_opt.numpy(),theta_GP_sigma.clone().detach().numpy(),'error/out_scale', torch.exp(lognoisescale_opt).clone().detach().numpy()/self.GP_Components[0]['outputscale'])
                    #print('gradient', torch.mean(torch.abs(u.grad.squeeze())).numpy(), theta_SDE.grad.numpy())
            u.requires_grad_(False)
            theta_SDE.requires_grad_(False)
            theta_GP_mu.requires_grad_(False)
            theta_GP_sigma.requires_grad_(False)
            lognoisescale.requires_grad_(False)

        sigma_e_sq_MAP = torch.zeros(self.p)
        for i in range(self.p):
            lognoisescale[i] = torch.log(self.GP_Components[i]['noisescale'].double())
            self.GP_Components[0]['noisescale'] = torch.max (torch.exp(lognoisescale_opt[i]), 1e-6 * self.GP_Components[i]['outputscale'])
            sigma_e_sq_MAP[i] = self.GP_Components[i]['noisescale']

        u_est_err = torch.sqrt(torch.mean(torch.square(u_opt[:,0]-self.SDE_Module.u_true[:,0])))
        theta_err = (torch.mean((theta_opt[0:self.SDE_Module.theta_true.shape[0]]-self.SDE_Module.theta_true).square())).sqrt().clone()
        theta_opt = theta_opt[0:self.SDE_Module.theta_true.shape[0]]
        theta_err_relative = (torch.mean(((theta_opt[0:self.SDE_Module.theta_true.shape[0]]-self.SDE_Module.theta_true)/self.SDE_Module.theta_true).square())).sqrt().clone()
        print('Estimated parameter:', (theta_opt.clone().detach()).numpy(), 'True parameter:',self.SDE_Module.theta_true.numpy(), 'Error of theta:', theta_err,'relative err',theta_err_relative)
        time1 = time.time() - time0
        Var = self.Normal_Approximation(u_opt, theta_SDE, theta_GP_mu, theta_GP_sigma, lognoisescale/2)

        map_est={
            'theta_err': theta_err, 
            'theta_MAP' : theta_opt.clone().detach(), 
            'theta_SDE':theta_SDE.clone(),
            'theta_GP_mu':theta_GP_mu.clone(),
            'theta_GP_sigma':theta_GP_sigma.clone(),
            'sigma_e_sq_MAP':lognoisescale,
            'u_MAP' : u_opt, 
            'u_err': u_est_err,
            'time': time1,
            'var':Var
        }
        return (map_est)

    def Minus_Log_Posterior(self, u, theta_SDE, theta_GP_mu, theta_GP_sigma, logsigma = torch.log(torch.tensor([1e-6,1e-6]))/2) :
        #for i in range(self.p):
        #    u[self.SDE_Module.ind_obs,i] = self.y_obs[:,i]
        lkh = torch.zeros((self.p, 3))
        mu = self.GP_predict(self.x_I,theta_GP_mu)
        sigma_gp, sigma_gp_c = self.GP_std(mu, theta_GP_sigma)
        mu_sde, sigma_sde = self.SDE_Module.Source(self.x_I, u, theta_SDE)
        for i in range(self.p):
            outputscale = self.GP_Components[i]['outputscale']
            # p(X[I] = x[I]) = P(U[I] = u[I])
            sigma_sq_hat = ((u[:,i]-mu[:,i]) / sigma_gp[:,i]) @ self.GP_Models[i].Cinv @ ((u[:,i]-mu[:,i]) / sigma_gp[:,i])
            lkh[i,0] = - 0.5 * sigma_sq_hat - torch.sum(torch.log(sigma_gp[:,i])) - 1e6 * torch.square(mu[0,i]-self.SDE_Module.y_ini[i])
            
            # p(Y[I] = y[I] | X[I] = x[I])
            noisescale = self.noisy_known* (self.GP_Components[i]['noisescale'].clone()) + (1-self.noisy_known) * torch.max(torch.exp(2 * logsigma[i]), 1e-6 * outputscale)
            #if i == 0:
            lkh[i,1] = - 0.5 / noisescale * torch.sum (torch.square(u[self.SDE_Module.ind_obs,i]-self.y_obs[:,i])) - 0.5 * self.n_obs * torch.log(noisescale)
            # p(X'[I]=f(x[I],theta)|X(I)=x(I))
            mu_gp  = mu[1:self.n_I,i]  +  (sigma_gp[1:self.n_I,i] * self.GP_Models[i].Cxx[0,1]) / (sigma_gp[0:-1,i] * self.GP_Models[i].Cxx[0,0]) * (u[0:-1,i]-mu[0:-1,i])           
            sigma_gp_c = torch.max(sigma_gp_c, 1e-9 * outputscale)
            sigma_sde = torch.max(sigma_sde, 1e-9 * outputscale)
            lkh[i,2] = - 0.5 * torch.sum(torch.log(sigma_sde[:,i]) - torch.log(sigma_gp_c[:,i]) 
                                         - 0.5 + 0.5 * torch.square(sigma_gp_c[:,i]/sigma_sde[:,i]) 
                                         + 0.5 * torch.square((mu_sde[:,i]-mu_gp)/sigma_sde[:,i])) 
            lkh[i,2] = lkh[i,2] * 1000000
        return (-torch.sum(lkh))

    def GP_predict(self, t_grid, theta) :
        if self.para is False:
            r = self.RBF_kernel(t_grid, self.x_I)
            pred = r @ theta
        else:
            t_grid = t_grid.reshape(-1,1)
            if self.sde_operator == 0 :
                pred = theta[0] * t_grid + self.SDE_Module.y_ini[0]
            elif self.sde_operator == 2 :
                pred = theta[1] - torch.exp(-theta[0] * t_grid) * (theta[1] - self.SDE_Module.y_ini[0])
            elif self.sde_operator == 3 :
                pred = torch.sqrt(theta[0] * 2 * t_grid + self.SDE_Module.y_ini[0] ** 2)
            elif self.sde_operator == 4 :
                pred = self.SDE_Module.True_Solution_no_err(t_grid, theta[0:4])
            elif self.sde_operator == 5 :
                pred = 2*theta[0] * t_grid**2 + t_grid**4 + self.SDE_Module.y_ini[0]

            elif self.sde_operator == -1 :
                pred = - self.SDE_Module.y_ini[0] * torch.exp(theta[0] * t_grid)
            elif self.sde_operator == -2 :
                pred = theta[1] - torch.exp(-theta[0] * t_grid) * (theta[1] - self.SDE_Module.y_ini[0])
            elif self.sde_operator == -3 :
                u2 = theta[1] - torch.exp(-theta[0] * t_grid) * (theta[1] - self.SDE_Module.y_ini[1])
                u1 = torch.zeros(u2.shape)
                pred = torch.cat((u1,u2),1)
            elif self.sde_operator == -4 :
                u2 = theta[1] - torch.exp(-theta[0] * t_grid) * (theta[1] - self.SDE_Module.y_ini[1])
                u1 = torch.zeros(u2.shape)
                pred = torch.cat((u1,u2),1)
        return (pred)

    def GP_std(self, mu, theta) :
        if self.sde_operator == 0 :
            sigma_gp = torch.abs(theta[0]) * torch.ones(self.n_I,1)

        elif self.sde_operator == 2 :
            sigma_gp = torch.abs(theta[0]) * torch.ones(self.n_I,1)
        elif self.sde_operator == 3 :
            sigma_gp = torch.abs(theta[0]) * torch.ones(self.n_I,1) 
        elif self.sde_operator == 4 :
            sigma_gp = torch.abs(theta[0]) * torch.abs(mu) ** theta[1]
        elif self.sde_operator == 5 :
            sigma_gp = torch.abs(theta[0]) * torch.ones(self.n_I,1) 

        elif self.sde_operator == -1 :
            sigma_gp = torch.abs(theta[0]) * torch.abs(mu)
        elif self.sde_operator == -2 :
            sigma_gp = torch.abs(theta[0]) * torch.abs(mu)
        elif self.sde_operator == -3 :
            sigma_gp = torch.empty(mu.shape)
            sigma_gp[:,1] = torch.abs(theta[0]) * torch.abs(mu[:,1])
            sigma_gp[:,0] = torch.sqrt(torch.abs(mu[:,1]))
        elif self.sde_operator == -4 :
            sigma_gp = torch.empty(mu.shape)
            sigma_gp[:,1] = torch.abs(theta[1]) * torch.abs(mu[:,1])
            sigma_gp[:,0] = torch.abs(theta[0]) * torch.abs(mu[:,0])
        sigma_gp_c = sigma_gp[1:self.n_I,:] * torch.sqrt((self.GP_Models[0].Cxx[1,1] - self.GP_Models[0].Cxx[0,1]**2/self.GP_Models[0].Cxx[0,0]))
        return (sigma_gp, sigma_gp_c)
    
    def RBF_kernel(self, x1, x2):
        rho = self.rho
        r_ = torch.exp( - torch.square(x1.reshape(-1,1) - x2.reshape(1,-1)) / rho)
        # handle limit at 0, allows more efficient backprop
        r_ = r_.clamp_(1e-15) 
        C_ = r_
        return (C_)
    
    def _Pre_Process(self):
        # obtain features from GP_Components
        u = torch.empty(self.n_I, self.p).double()
        theta_GP_mu = torch.empty(self.n_I, self.p).double()
        self.u_ini = torch.empty(self.n_I, self.p).double()
        self.corr_cond_mid = torch.empty(self.p).double()
        for i in range(self.p):
            kernel = self.GP_Components[i]['kernel']
            mean = torch.tensor(self.GP_Components[i]['mean'])
            # Compute GP prior covariance matrix
            self.GP_Models[i].Cxx = kernel.K(self.x_I) + 1e-6 * torch.eye(self.n_I)
            self.GP_Models[i].Cinv = torch.linalg.inv(self.GP_Models[i].Cxx )
            Cxx = self.GP_Models[i].Cxx
            self.GP_Models[i].Cinv_obs = torch.linalg.inv(Cxx[self.ind_obs][:,self.ind_obs])
            # obtain initial values
            C_Ia = kernel.K(self.x_I, self.x_obs)
            S = C_Ia @ torch.linalg.inv(kernel.K(self.x_obs) + 1e-6 * torch.eye(self.x_obs.shape[0]))
            u[:,i] = mean + S @ (self.y_obs[:,i] - mean)
            self.parameter_GP = torch.tensor([1.,1.])
            if self.sde_operator == 4 : self.parameter_GP = torch.tensor([0.1,0.,1.,1.,0.,0.])
            if self.sde_operator == -2 : self.parameter_GP = torch.tensor([1.,1.,1.])

            K = self.RBF_kernel(self.x_I, self.x_I)
            sig = 0.1*torch.eye(self.n_I)
            sig[0,0] = 1e-6
            y_smooth = K @  torch.linalg.inv(K + sig) @ u[:,i]
            theta_GP_mu[:,i] = torch.linalg.inv(K + 1e-6*torch.eye(self.n_I)) @ y_smooth
            self.u_ini[:,i] = y_smooth
            x_local = torch.tensor([self.x_I[0],self.x_I[1]])
            self.GP_Models[i].Sigma_local = kernel.K(torch.reshape(x_local,(1,-1)))
            r = kernel.K(torch.reshape(x_local,(1,-1)),torch.reshape(self.x_I_c[0],(1,-1)))
            self.GP_Models[i].r_local = r.squeeze()
            self.corr_cond_mid[i] = (1 - self.GP_Models[i].r_local @ torch.linalg.inv(self.GP_Models[i].Sigma_local) @ self.GP_Models[i].r_local)
            #self.corr_cond_mid[i] = (1 - self.GP_Models[i].r_local[0]**2)
            self.GP_Models[i].r_pred = self.GP_Models[i].r_local @ torch.linalg.inv(self.GP_Models[i].Sigma_local)
            #self.GP_Models[i].r_pred = self.GP_Models[i].r_local[0] / (self.GP_Models[i].Sigma_local[0,0])

        if self.para is True:
            theta_GP_mu = self.SDE_Module.para_theta.clone()
            theta_GP_mu = theta_GP_mu[0:-1]
        return (u, theta_GP_mu)
    
    def minus_log_likelihood_p(self,theta_para, u, t_list):
        delta_u = u[1:self.n_I,:]-u[0:-1,:]
        mu_sde, sigma_sde = self.SDE_Module.Drift(t_list, u, theta_para)
        lik = 0.5 * torch.sum(torch.square((delta_u - mu_sde)/sigma_sde) ) +  torch.sum(torch.log(sigma_sde))
        return (lik)

    def minus_log_likelihood_a(self,theta_para, u, t_list):
        if self.sde_operator == 0: # Brownian motion
            lkh = self.minus_log_likelihood_p(theta_para, u, t_list)
        elif self.sde_operator == 2: # OU process
            n = u.shape[0]
            theta_para[0] = np.maximum(theta_para[0], 1e-6)
            theta_para[2] = np.maximum(theta_para[2], 1e-6)
            u = u-theta_para[1]
            delta_u = u[1:self.n_obs,:]
            delta_t = t_list[1:n]-t_list[0:-1]
            mu_sde = u[0:-1] * torch.exp( - delta_t * theta_para[0])
            sigma_sde =theta_para[2]/ np.sqrt(2 * theta_para[0]) *torch.sqrt(1 - torch.exp( - 2 * delta_t * theta_para[0]))
            lkh = 0.5 * torch.sum( torch.square(delta_u - mu_sde)/sigma_sde**2 ) +  torch.sum(torch.log(sigma_sde))
        elif self.sde_operator == -1: # GBM
            n = u.shape[0]
            u = torch.log(u)
            u = u - torch.log(self.SDE_Module.y_ini[0])
            delta_u = u[1:self.n_obs,:]-u[0:-1,:]
            delta_t = t_list[1:n]-t_list[0:-1]
            mu_sde = (theta_para[0]-theta_para[1]**2/2) * delta_t
            sigma_sde = theta_para[1] * torch.sqrt(delta_t)
            lkh = 0.5 * torch.sum( torch.square(delta_u - mu_sde)/sigma_sde**2 ) +  torch.sum(torch.log(sigma_sde))
        else: 
            lkh = self.minus_log_likelihood_p(theta_para, u, t_list)
        return (lkh)

    def Normal_Approximation(self, u, theta_SDE, theta_GP_mu, theta_GP_sigma, logsigma = None):
        parameter = (u, theta_SDE, theta_GP_mu, theta_GP_sigma, logsigma)
        precision = torch.autograd.functional.hessian(self.Minus_Log_Posterior, parameter)
        Posterior_para_NA = torch.linalg.pinv(precision[1][1])
        return (Posterior_para_NA)

    def Sample_Using_HMC(
        self, 
        n_epoch = 5000, 
        lsteps=100, 
        epsilon=1e-5, 
        n_samples=20000, 
        Map_Estimation = None, 
        opt_algorithm = 2,
        center_modify = True
        ):
        if Map_Estimation is None : Map_Estimation=self.map(nEpoch = n_epoch, opt_algorithm = opt_algorithm)
        self.Map_Estimation = Map_Estimation
        log_sigma=torch.log(Map_Estimation['sigma_e_sq_MAP']).double() / 2
        print('log_sigma',log_sigma)
        u_KL=Map_Estimation['u_MAP']
        theta_SDE=Map_Estimation['theta_SDE']
        theta_GP_mu=Map_Estimation['theta_GP_mu']
        theta_GP_sigma=Map_Estimation['theta_GP_sigma']
        print(theta_SDE,theta_GP_mu,theta_GP_sigma)
        lkh = self.Minus_Log_Posterior
        if center_modify : lkh = self.Minus_Log_Posterior_c
        self.sampler = HMC.Posterior_Density_Inference(self.SDE_Module, lkh, u_KL, theta_SDE, theta_GP_mu, theta_GP_sigma, log_sigma, u_KL.shape, theta_SDE.shape, theta_GP_mu.shape, theta_GP_sigma.shape, log_sigma.shape, self.ind_non_obs, noisy_known=self.noisy_known, lsteps=lsteps, epsilon=epsilon, n_samples=n_samples)
        print(u_KL.shape)
        self.HMC_sample = self.sampler.Sampling()
        return (self.HMC_sample, self.Map_Estimation)

    def map_margin(self, nEpoch = 2500, opt_algorithm = 2, center_modify = False):
        self.mle()
        #setup initial values
        self.opt_algorithm = opt_algorithm
        u, theta_GP_mu= self._Pre_Process()
        u=u.requires_grad_()

        self.u_plot = []
        self.opt_algorithm = opt_algorithm
        u, theta_GP_mu= self._Pre_Process()
        
        theta_SDE = torch.tensor(self.theta_MLE).requires_grad_()
        theta_GP_mu =  theta_GP_mu.requires_grad_()
        theta_GP_sigma = torch.tensor([0.1]).requires_grad_()
        if self.sde_operator == -4: theta_GP_sigma = torch.tensor([1.,1.]).requires_grad_()


        u_lr = 1e-2 * (self.sde_operator < 0 ) + 1e-1 * (self.sde_operator >= 0 )

        if self.sde_operator == 4 : theta_GP_sigma = torch.tensor([1.,0.01]).requires_grad_()
        time0 = time.time()
        p = self.y_obs.shape[1]
        lognoisescale = torch.zeros(p)
        for i in range(p):
            lognoisescale[i]=torch.log(self.GP_Components[i]['noisescale'].double())
        
        obj_loss = self.Margin_Log_Posterior
        if center_modify : obj_loss = self.Margin_Log_Posterior_c

        if self.opt_algorithm == 1:
            u_lr = u_lr * 10
            if self.noisy_known is False  :
                lognoisescale=lognoisescale.requires_grad_()
                self.optimizer_u_theta = torch.optim.LBFGS([theta_SDE, theta_GP_mu, theta_GP_sigma,lognoisescale], lr = u_lr)
            else :
                self.optimizer_u_theta = torch.optim.LBFGS([theta_SDE, theta_GP_mu, theta_GP_sigma], lr = u_lr)
                lognoisescale_opt=lognoisescale
            pointwise_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_u_theta, lr_lambda = lambda epoch: 1/((epoch+1)**0.5), last_epoch=-1)
            print('start optimiza theta and u:')
            def closure():
                self.optimizer_u_theta.zero_grad()
                loss = obj_loss( theta_SDE, theta_GP_mu, theta_GP_sigma, lognoisescale / 2)
                loss.backward()
                return loss
            for epoch in range(nEpoch):
                self.optimizer_u_theta.zero_grad()
                loss_u_theta = obj_loss( theta_SDE, theta_GP_mu, theta_GP_sigma, lognoisescale / 2)
                if epoch==0:
                    loss_u_theta_opt=loss_u_theta.clone().detach()
                    theta_opt=theta_SDE.clone().detach()
                    if self.noisy_known is False : lognoisescale_opt=lognoisescale.clone().detach()
                else:
                    #if para_theta[0]<0: para_theta[0] = torch.abs(para_theta[0])
                    if loss_u_theta<loss_u_theta_opt:
                        loss_u_theta_opt=loss_u_theta.clone().detach()
                        theta_opt=theta_SDE.clone().detach()
                        if self.noisy_known is False : lognoisescale_opt=lognoisescale.clone().detach()
                loss_u_theta.backward()
                self.optimizer_u_theta.step(closure)
                pointwise_lr_scheduler.step()
                if (np.isnan(obj_loss( theta_SDE, theta_GP_mu, theta_GP_sigma,lognoisescale / 2).detach().numpy())):
                    theta_SDE = theta_opt
                    if self.noisy_known is False : lognoisescale=lognoisescale_opt
                if (epoch+1) % 500 == 0 :
                    print(epoch+1, '/', nEpoch, 'current opt: theta:', theta_opt.numpy(),'error/out_scale', torch.exp(lognoisescale_opt).clone().detach().numpy()/self.GP_Components[0]['outputscale'])
                    #print('gradient', theta_SDE.grad.numpy())
            u.requires_grad_(False)
            theta_SDE.requires_grad_(False)
            theta_GP_mu.requires_grad_(False)
            theta_GP_sigma.requires_grad_(False)            
            lognoisescale.requires_grad_(False)
            
        elif self.opt_algorithm == 2:
            u_lr = u_lr/10
            torch.autograd.set_detect_anomaly(True)
            if self.noisy_known is False  :
                lognoisescale=lognoisescale.requires_grad_()
                optimizer_u_theta = torch.optim.Adam([theta_SDE, theta_GP_mu, theta_GP_sigma,lognoisescale], lr = u_lr)
            else :
                optimizer_u_theta = torch.optim.Adam([theta_SDE, theta_GP_mu, theta_GP_sigma,], lr = u_lr)
                lognoisescale_opt=lognoisescale
            pointwise_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_u_theta, lr_lambda = lambda epoch: 1/((epoch+1)**0.2), last_epoch=-1)
            print('start optimiza theta and u:')
            for epoch in range(nEpoch):
                optimizer_u_theta.zero_grad()
                loss_u_theta = obj_loss( theta_SDE, theta_GP_mu, theta_GP_sigma, lognoisescale / 2)
                if epoch == 0:
                    loss_u_theta_opt = loss_u_theta.clone().detach()
                    theta_opt = theta_SDE.clone().detach()
                    if self.noisy_known is False : lognoisescale_opt = lognoisescale.clone().detach()
                else:
                    #if para_theta[0]<0: para_theta[0] = torch.abs(para_theta[0])
                    if loss_u_theta<loss_u_theta_opt:
                        loss_u_theta_opt=loss_u_theta.clone().detach()
                        theta_opt=theta_SDE.clone().detach()
                        if self.noisy_known is False : lognoisescale_opt = lognoisescale.clone().detach()
                loss_u_theta.backward()
                optimizer_u_theta.step()
                pointwise_lr_scheduler.step()
                if (np.isnan(obj_loss( theta_SDE, theta_GP_mu, theta_GP_sigma, lognoisescale / 2).detach().numpy())):
                    theta_SDE = theta_opt
                    if self.noisy_known is False : lognoisescale = lognoisescale_opt
                if (epoch+1) % 500 == 0 :
                    print(epoch+1, '/', nEpoch, 'current opt: theta:', theta_opt.numpy(),'error/out_scale', torch.exp(lognoisescale_opt).clone().detach().numpy()/self.GP_Components[0]['outputscale'])
                    #print('gradient', theta_SDE.grad.numpy())
            u.requires_grad_(False)
            theta_SDE.requires_grad_(False)
            theta_GP_mu.requires_grad_(False)
            theta_GP_sigma.requires_grad_(False)
            lognoisescale.requires_grad_(False)

        sigma_e_sq_MAP = torch.zeros(self.p)
        for i in range(self.p):
            lognoisescale[i] = torch.log(self.GP_Components[i]['noisescale'].double())
            self.GP_Components[0]['noisescale'] = torch.max (torch.exp(lognoisescale_opt[i]), 1e-6 * self.GP_Components[i]['outputscale'])
            sigma_e_sq_MAP[i] = self.GP_Components[i]['noisescale']

        theta_err = (torch.mean((theta_opt[0:self.SDE_Module.theta_true.shape[0]]-self.SDE_Module.theta_true).square())).sqrt().clone()
        theta_opt = theta_opt[0:self.SDE_Module.theta_true.shape[0]]
        theta_err_relative = (torch.mean(((theta_opt[0:self.SDE_Module.theta_true.shape[0]]-self.SDE_Module.theta_true)/self.SDE_Module.theta_true).square())).sqrt().clone()
        print('Estimated parameter:', (theta_opt.clone().detach()).numpy(), 'True parameter:',self.SDE_Module.theta_true.numpy(), 'Error of theta:', theta_err,'relative err',theta_err_relative)
        time1 = time.time() - time0

        map_est={
            'theta_err': theta_err, 
            'theta_MAP' : theta_opt.clone().detach(), 
            'theta_SDE':theta_SDE.clone(),
            'theta_GP_mu':theta_GP_mu.clone(),
            'theta_GP_sigma':theta_GP_sigma.clone(),
            'sigma_e_sq_MAP':lognoisescale,
            'time': time1,
        }
        return (map_est)

    def Margin_Log_Posterior(self, theta_SDE, theta_GP_mu, theta_GP_sigma, logsigma = torch.log(torch.tensor([1e-6]))/2) :
        lkh = torch.zeros((self.p, 3))
        mu = self.GP_predict(self.x_I,theta_GP_mu)
        sigma_gp, sigma_gp_c = self.GP_std(mu, theta_GP_sigma)
        #print(sigma_gp.shape,mu.shape,mu_sde.shape,u.shape)
        for num_simu in range (self.num_sem):
            u = self.GP_simu(sigma_gp, mu, self.y_obs, self.x_obs, self.x_I)
            for i in range(self.p):
                outputscale = self.GP_Components[i]['outputscale']
                # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                
                sigma_sq_hat = ((self.y_obs[:,i]-mu[self.ind_obs,i]) / sigma_gp[self.ind_obs,i]) @ self.GP_Models[i].Cinv_obs @ ((self.y_obs[:,i]-mu[self.ind_obs,i]) / sigma_gp[self.ind_obs,i])
                lkh[i,0] += - 0.5 * sigma_sq_hat - torch.sum(torch.log(sigma_gp[self.ind_obs,i]))
                #sigma_sq_hat = ((u[:,i]-mu[:,i]) / sigma_gp[:,i]) @ self.GP_Models[i].Cinv @ ((u[:,i]-mu[:,i]) / sigma_gp[:,i])
                #lkh[i,0] = - 0.5 * (sigma_sq_hat + torch.sum(torch.log(sigma_gp[:,i])) ) - 1e6 * (mu[0,i]-self.SDE_Module.y_ini[i])**2
                mu_sde, sigma_sde = self.SDE_Module.Source(self.x_I, u, theta_SDE)
                mu_gp  = mu[1:self.n_I,i]  +  (sigma_gp[1:self.n_I,i] * self.GP_Models[i].Cxx[0,1]) / (sigma_gp[0:-1,i] * self.GP_Models[i].Cxx[0,0]) * (u[0:-1,i]-mu[0:-1,i])           
                sigma_gp_c = torch.max(sigma_gp_c, 1e-9 * outputscale)
                sigma_sde = torch.max(sigma_sde, 1e-9 * outputscale)
                lkh[i,2] += - 0.5 * torch.sum(2*torch.log(sigma_sde[:,i])-2*torch.log(sigma_gp_c[:,i])-1+(sigma_gp_c[:,i]/sigma_sde[:,i])**2 + (mu_sde[:,i]-mu_gp)**2/(sigma_sde[:,i]**2)) 
        return (-torch.sum(lkh))

    def GP_simu(self, sigma_gp, mu_gp, u_obs, t_obs, t_I):
        u = torch.empty(t_I.shape[0],self.p)
        for i in range (self.p):
            kernel = self.GP_Components[i]['kernel']
            K = kernel.K(t_I)
            sigma_gp_obs = torch.diag(sigma_gp[self.ind_obs,i])
            mu_gp_obs = mu_gp[self.ind_obs,i]
            K_obs = kernel.K(t_obs) + 1e-6 * torch.eye(t_obs.shape[0])
            K_12 = kernel.K(t_obs,t_I)
            self.GP_Models[i].Cxx = kernel.K(self.x_I) + 1e-6 * torch.eye(self.n_I)
            Sig = torch.diag(sigma_gp[:,i]) @ K @ torch.diag(sigma_gp[:,i]) + 1e-6 * torch.eye(t_I.shape[0]) - torch.diag(sigma_gp[:,i]) @ K_12.T @ torch.linalg.inv(K_obs) @ K_12 @ torch.diag(sigma_gp[:,i])
            u[:,i] = mu_gp[:,i] + torch.diag(sigma_gp[:,i]) @ K_12.T @ sigma_gp_obs @ torch.linalg.inv(sigma_gp_obs @ K_obs @ sigma_gp_obs) @ (u_obs[:,i] - mu_gp_obs) + torch.linalg.cholesky(Sig) @ torch.randn(t_I.shape[0])
        return(u)

    def mle_joint(self, nEpoch = 2500, opt_algorithm = 2):
        self.mle()
        #setup initial values
        self.opt_algorithm = opt_algorithm
        u, theta_GP_mu= self._Pre_Process()
        u=u.requires_grad_()
        theta_SDE = self.SDE_Module.para_theta.requires_grad_()

        u_lr = 1e-2 * (self.sde_operator < 0 ) + 1e-1 * (self.sde_operator >= 0 )
        time0 = time.time()
        if self.opt_algorithm == 1:
            u_lr = u_lr * 10
            self.optimizer_u_theta = torch.optim.LBFGS([u,theta_SDE], lr = u_lr)
            pointwise_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_u_theta, lr_lambda = lambda epoch: 1/((epoch+1)**0.5), last_epoch=-1)
            print('start optimiza theta and u:')
            def closure():
                self.optimizer_u_theta.zero_grad()
                loss = self.minus_log_likelihood(theta_SDE, u, self.x_I)
                loss.backward()
                return loss
            for epoch in range(nEpoch):
                self.optimizer_u_theta.zero_grad()
                loss_u_theta = self.minus_log_likelihood(theta_SDE, u, self.x_I)
                if epoch==0:
                    loss_u_theta_opt=loss_u_theta.clone().detach()
                    u_opt=u.clone().detach()
                    theta_opt=theta_SDE.clone().detach()
                else:
                    #if para_theta[0]<0: para_theta[0] = torch.abs(para_theta[0])
                    if loss_u_theta<loss_u_theta_opt:
                        loss_u_theta_opt=loss_u_theta.clone().detach()
                        u_opt=u.clone().detach()
                        theta_opt=theta_SDE.clone().detach()
                loss_u_theta.backward()
                self.optimizer_u_theta.step(closure)
                pointwise_lr_scheduler.step()
                if (np.isnan(self.minus_log_likelihood(theta_SDE, u, self.x_I).detach().numpy())):
                    u = u_opt
                    theta_SDE = theta_opt
                if (epoch+1) % 500 == 0 :
                    print(epoch+1, '/', nEpoch, 'current opt: theta:', theta_opt.numpy())
                    #print('gradient', torch.mean(torch.abs(u.grad.squeeze())).numpy(), theta_SDE.grad.numpy())
            u.requires_grad_(False)
            theta_SDE.requires_grad_(False)
        
        elif self.opt_algorithm == 2:
            u_lr = u_lr/10
            torch.autograd.set_detect_anomaly(True)
            optimizer_u_theta = torch.optim.Adam([u,theta_SDE], lr = u_lr)
            pointwise_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_u_theta, lr_lambda = lambda epoch: 1/((epoch+1)**0.2), last_epoch=-1)
            print('start optimiza theta and u:')
            for epoch in range(nEpoch):
                optimizer_u_theta.zero_grad()
                loss_u_theta = self.minus_log_likelihood(theta_SDE, u, self.x_I)
                if epoch == 0:
                    loss_u_theta_opt = loss_u_theta.clone().detach()
                    u_opt = u.clone().detach()
                    theta_opt = theta_SDE.clone().detach()
                else:
                    #if para_theta[0]<0: para_theta[0] = torch.abs(para_theta[0])
                    if loss_u_theta<loss_u_theta_opt:
                        loss_u_theta_opt=loss_u_theta.clone().detach()
                        u_opt=u.clone().detach()
                        theta_opt=theta_SDE.clone().detach()
                loss_u_theta.backward()
                optimizer_u_theta.step()
                pointwise_lr_scheduler.step()
                if (np.isnan(self.minus_log_likelihood(theta_SDE, u, self.x_I).detach().numpy())):
                    u = u_opt
                    theta_SDE = theta_opt
                if (epoch+1) % 500 == 0 :
                    print(epoch+1, '/', nEpoch, 'current opt: theta:', theta_opt.numpy())
                    #print('gradient', torch.mean(torch.abs(u.grad.squeeze())).numpy(), theta_SDE.grad.numpy())
            u.requires_grad_(False)
            theta_SDE.requires_grad_(False)
        u_est_err = torch.sqrt(torch.mean(torch.square(u_opt[:,0]-self.SDE_Module.u_true[:,0])))
        theta_err = (torch.mean((theta_opt[0:self.SDE_Module.theta_true.shape[0]]-self.SDE_Module.theta_true).square())).sqrt().clone()
        theta_opt = theta_opt[0:self.SDE_Module.theta_true.shape[0]]
        theta_err_relative = (torch.mean(((theta_opt[0:self.SDE_Module.theta_true.shape[0]]-self.SDE_Module.theta_true)/self.SDE_Module.theta_true).square())).sqrt().clone()
        print('Estimated parameter:', (theta_opt.clone().detach()).numpy(), 'True parameter:',self.SDE_Module.theta_true.numpy(), 'Error of theta:', theta_err,'relative err',theta_err_relative)
        time1 = time.time() - time0
        mle_est={
            'theta_err': theta_err, 
            'theta_mle' : theta_opt.clone().detach(), 
            'u_MLE' : u_opt, 
            'u_err': u_est_err,
            'time': time1,
        }
        return (mle_est)

    def HMC_lkh(
        self, 
        n_epoch = 5000, 
        lsteps=100, 
        epsilon=1e-5, 
        n_samples=20000, 
        MLE_Estimation = None, 
        opt_algorithm = 2
        ):
        if MLE_Estimation is None : MLE_Estimation=self.map(nEpoch = n_epoch, opt_algorithm = opt_algorithm)
        self.MLE_Estimation = MLE_Estimation
        log_sigma=torch.log(MLE_Estimation['sigma_e_sq_MAP']).double() / 2
        u_KL=MLE_Estimation['u_MAP']
        theta_SDE=MLE_Estimation['theta_SDE']
        self.sampler_lkh = HMC2.Posterior_Density_Inference(self.SDE_Module, self.minus_log_likelihood, u_KL, theta_SDE, log_sigma, u_KL.shape, theta_SDE.shape, log_sigma.shape, self.ind_non_obs, noisy_known=self.noisy_known, lsteps=lsteps, epsilon=epsilon, n_samples=n_samples)
        self.HMC_sample = self.sampler_lkh.Sampling()
        return (self.HMC_sample)

    def minus_log_likelihood(self,theta_para, u, t_list):
        delta_u = u[1:self.n_I,:]-u[0:-1,:]
        mu_sde, sigma_sde = self.SDE_Module.Drift(t_list, u, theta_para)
        lik = 0.5 * torch.sum(torch.square(delta_u - mu_sde)/sigma_sde**2 ) +  torch.sum(torch.log(sigma_sde)) + 1e8*torch.sum(torch.square(u[self.SDE_Module.ind_obs]-self.y_obs))
        return (lik)

    def Minus_Log_Posterior_c(self, u, theta_SDE, theta_GP_mu, theta_GP_sigma, logsigma = torch.log(torch.tensor([1e-6]))/2) :
        u = torch.abs(u)
        #theta_SDE = torch.exp(theta_SDE)
        #theta_GP_sigma = torch.exp(theta_GP_sigma)
        # print(theta_SDE)
        # print(theta_GP_sigma)
        lkh = torch.zeros((self.p, 3))
        mu_sde, sigma_sde = self.SDE_Module.Source_c(self.x_I, u, theta_SDE)
        
        mu = self.GP_mean_c(self.x_I,theta_GP_mu)
        mu_c = self.GP_mean_c(self.x_I_c,theta_GP_mu)
        sigma_gp = self.GP_std_c(mu, theta_GP_sigma)
        sigma_gp_c = self.GP_std_c(mu_c, theta_GP_sigma)        
        for i in range(self.p):
            outputscale = self.GP_Components[i]['outputscale']
            sigma_sq_hat = ((u[:,i]-mu[:,i]) / sigma_gp[:,i]) @ self.GP_Models[i].Cinv @ ((u[:,i]-mu[:,i]) / sigma_gp[:,i])
            lkh[i,0] = - 0.5 * sigma_sq_hat - torch.sum(torch.log(sigma_gp[:,i])) - 1e6 * torch.square(mu[0,i]-self.SDE_Module.y_ini[i])
            noisescale = self.noisy_known* (self.GP_Components[i]['noisescale'].clone()) + (1-self.noisy_known) * torch.max(torch.exp(2 * logsigma[i]), 1e-6 * outputscale)
            lkh[i,1] = - 0.5 / noisescale * torch.sum (torch.square(u[self.SDE_Module.ind_obs,i]-self.y_obs[:,i])) - 0.5 * self.n_obs * torch.log(noisescale)
        mu_gp_cond, sigma_gp_cond = self.GP_predict_c(u, mu, sigma_gp, sigma_gp_c, theta_GP_mu)
        lkh[0,2] = lkh[0,2] - 0.5 * torch.sum(torch.log(sigma_sde)-torch.log(sigma_gp_cond) + 
                                                  0.5 * torch.square(sigma_gp_cond/sigma_sde) + 
                                                  0.5 * torch.square((mu_sde-mu_gp_cond)/sigma_sde))
        #print(torch.sum(torch.square(u-self.y_obs)))
        return (-torch.sum(lkh))

    def Margin_Log_Posterior_c(self, theta_SDE, theta_GP_mu, theta_GP_sigma, logsigma = torch.log(torch.tensor([1e-6,1e-6]))/2) :
        lkh = torch.zeros(1)
        mu = self.GP_mean_c(self.x_I,theta_GP_mu)
        mu_c = self.GP_mean_c(self.x_I_c,theta_GP_mu)
        sigma_gp = self.GP_std_c(mu, theta_GP_sigma)
        sigma_gp_c = self.GP_std_c(mu_c, theta_GP_sigma)
        #mu_gp_c = torch.empty((self.n_I-1, self.p))
        for simu in range(self.num_sem):
            if self.n_I == self.n_obs : u = self.y_obs
            else: 
                u = self.GP_simu(sigma_gp, mu, self.y_obs, self.x_obs, self.x_I)
            sigma_sq_hat = ((self.y_obs[:,0]-mu[self.ind_obs,0]) / sigma_gp[self.ind_obs,0]) @ self.GP_Models[0].Cinv_obs @ ((self.y_obs[:,0]-mu[self.ind_obs,0]) / sigma_gp[self.ind_obs,0])
            lkh += - 0.5 * sigma_sq_hat - torch.sum(torch.log(sigma_gp[self.ind_obs]))
            mu_sde, sigma_sde = self.SDE_Module.Source_c(self.x_I, u, theta_SDE)
            mu_gp_cond,sigma_gp_c = self.GP_predict_c(u, mu, sigma_gp, sigma_gp_c, theta_GP_mu)
            lkh += - 0.5 * torch.sum(2 * torch.log(sigma_sde)-torch.log(sigma_gp_c) + torch.square(sigma_gp_c/sigma_sde) + torch.square((mu_sde- mu_gp_cond) / sigma_sde))
        #print(torch.sum(torch.square(u-self.SDE_Module.y_I)))
        #self.u_rest = u

        return (-lkh)

    def GP_mean_c(self, t_grid, theta) :
        if self.para is False:
            r = self.RBF_kernel(t_grid, self.x_I)
            pred = r @ theta
        else:
            t_grid = t_grid.reshape(-1,1)
            if self.sde_operator == 0 :
                pred = theta[0] * t_grid + self.SDE_Module.y_ini[0]

            elif self.sde_operator == 2 :
                pred = theta[1] - torch.exp(-theta[0] * t_grid) * (theta[1] - self.SDE_Module.y_ini[0])
            elif self.sde_operator == 3 :
                pred = torch.sqrt(theta[0] * 2 * t_grid + self.SDE_Module.y_ini[0] ** 2)
            elif self.sde_operator == 4 :
                pred = self.SDE_Module.True_Solution_no_err(t_grid, theta[0:4])
            elif self.sde_operator == 5 :
                pred = 2*theta[0] * t_grid**2 + t_grid**4 + self.SDE_Module.y_ini[0]

            elif self.sde_operator == -1 :
                pred = - self.SDE_Module.y_ini[0] * torch.exp(theta[0] * t_grid)
            elif self.sde_operator == -2 :
                pred = theta[1] - torch.exp(-theta[0] * t_grid) * (theta[1] - self.SDE_Module.y_ini[0])
            elif self.sde_operator == -3 :
                u2 = theta[1] - torch.exp(-theta[0] * t_grid) * (theta[1] - self.SDE_Module.y_ini[1])
                u1 = torch.zeros(u2.shape)
                pred = torch.cat((u1,u2),1)
            elif self.sde_operator == -4 :
                u1 = theta[1] - torch.exp(-theta[0] * t_grid) * (theta[1] - self.SDE_Module.y_ini[1])
                u2 = torch.zeros(u2.shape)
                pred = torch.cat((u1,u2),1)
        return (pred)

    def GP_predict_c(self, u, mu, sigma_gp, sigma_gp_c, theta_GP_mu):
        mu_gp_c = torch.empty(self.n_I-1, self.p)
        mu_c = self.GP_mean_c(self.x_I_c,theta_GP_mu)
        sigma_gp_cond = torch.zeros(sigma_gp_c.size())
        for i in range(self.p):
            U = torch.stack((u[0:-1,i],u[1:self.n_I,i]))
            MU = torch.stack((mu[0:-1,i],mu[1:self.n_I,i]))
            mu_gp_c[:,i]  = mu_c[:,i]  +  self.GP_Models[i].r_pred @ (U- MU)    
            #mu_gp_c[:,i]  = mu_c[:,i]  +  self.GP_Models[i].r_pred * (u[0:-1,i]- mu[0:-1,i])    
            sigma_gp_cond[:,i] = sigma_gp_c[:,i] * torch.sqrt(self.corr_cond_mid[i])
        return(mu_gp_c, sigma_gp_cond)

    def GP_std_c(self, mu, theta) : 
        n = mu.shape[0]
        if self.sde_operator == 0 :
            sigma_gp = torch.abs(theta[0]) * torch.ones(n,1)

        elif self.sde_operator == 2 :
            sigma_gp = torch.abs(theta[0]) * torch.ones(n,1)
        elif self.sde_operator == 3 :
            sigma_gp = torch.abs(theta[0]) * torch.ones(n,1) 
        elif self.sde_operator == 4 :
            sigma_gp = torch.abs(theta[0]) * torch.abs(mu) ** theta[1]
        elif self.sde_operator == 5 :
            sigma_gp = torch.abs(theta[0]) * torch.ones(n,1) 

        elif self.sde_operator == -1 :
            sigma_gp = torch.abs(theta[0]) * torch.abs(mu)
        elif self.sde_operator == -2 :
            sigma_gp = torch.abs(theta[0]) * torch.abs(mu)
        elif self.sde_operator == -3 :
            sigma_gp = torch.empty(mu.shape)
            sigma_gp[:,1] = torch.abs(theta[0]) * torch.abs(mu[:,1])
            sigma_gp[:,0] = torch.sqrt(torch.abs(mu[:,1]))
        elif self.sde_operator == -4 :
            sigma_gp = torch.empty((mu.shape))
            mu = torch.abs(mu)
            sigma_gp[:,0] = theta[0]*mu[:,0]
            sigma_gp[:,1] = theta[1]*mu[:,1]
        return (sigma_gp)



