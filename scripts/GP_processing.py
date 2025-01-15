
import numpy as np
import scipy.special as fun
import torch
from scipy.optimize import minimize
import time

#import mkl

# Set the number of threads to 1
#mkl.set_num_threads(1)

torch.set_default_dtype(torch.double)

class Bessel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, nu):
        ctx._nu = nu
        ctx.save_for_backward(inp)
        return (torch.from_numpy(np.array(fun.kv(nu,inp.detach().numpy()))))

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        nu = ctx._nu
        grad_in = grad_out.numpy() * np.array(fun.kvp(nu,inp.detach().numpy()))
        return (torch.from_numpy(grad_in), None)

class Matern_Product(object):
    # has_lengthscale = True
    def __init__(self, nu = 2.1, lengthscale = 1e-1, sde_operator=None, **kwargs):
        # super(Matern,self).__init__(**kwargs)
        if (sde_operator is None): sde_operator=1
        self.nu = nu
        self.sde_operator=sde_operator
        self.log_lengthscale = torch.tensor(np.log(lengthscale))
        self.log_lengthscale.requires_grad_(True)
        if sde_operator == 4:
            self.theta=torch.rand(3)
        if sde_operator == 5:
            self.theta=torch.rand(2)
    def _set_lengthscale(self, lengthscale):
        self.log_lengthscale = torch.tensor(np.log(lengthscale))
    def lengthscale(self):
        return (torch.exp(self.log_lengthscale).detach())
    def forward(self, x1, x2 = None, **params):
        x1 = x1.squeeze()
        if (len(x1.shape)==1):
            d=1
            if x2 is None: x2 = x1
            return (self.corr_comp(x1, x2))
        else:
            d=x1.shape[1]
        n1=x1.shape[0]
        if x2 is None: x2 = x1
        n2=x2.shape[0]
        C_=torch.ones(n1,n2)
        for i in range(d):
            C_ = C_ * self.corr_comp(x1[:,i], x2[:,i], ind=i)
        return (C_)

    def corr_comp(self, x1, x2 = None, ind=0, **params):
        lengthscale = torch.exp(self.log_lengthscale)
        if x2 is None: x2 = x1
        if x1.shape[0]==0 or x2.shape[0]==0 : 
            return (torch.zeros(0))
        x1 = x1.squeeze()
        x2 = x2.squeeze()

        if self.nu == 0 :
            r_ = torch.min(x1.reshape(-1,1), x2.reshape(1,-1))
            # handle limit at 0, allows more efficient backprop
            r_ = r_.clamp_(1e-15) 
            C_ = r_

        else:
            r_ = (x1.reshape(-1,1) - x2.reshape(1,-1)).abs()
            r_ = np.sqrt(2.*self.nu) * r_ / lengthscale[ind]
            # handle limit at 0, allows more efficient backprop
            r_ = r_.clamp_(1e-15) 
            C_ = np.power(2,1-self.nu)*np.exp(-fun.loggamma(self.nu))*torch.pow(r_,self.nu)
            C_ = C_ * Bessel.apply(r_,self.nu)
        return (C_)

    def K(self, x1, x2 = None):
        if (x2 is None): x2 = x1
        if x1.shape[0] * x1.shape[1] * x2.shape[0] * x2.shape[1]== 0 : return (torch.zeros(0))
        return (self.forward(x1,x2).detach())

    def LK(self, x1, x2 = None, sde_operator=None, theta = None):  # change here when operator changes: LK
        if (sde_operator is None): sde_operator=self.sde_operator
        if x2 is None: x2 = x1
        if (sde_operator==1):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_1=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.dC_dx_L_comp(x1[:,1], x2[:,1], ind=1)
            return(comp_1+comp_2)
        elif (sde_operator==2):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_1=self.d2C_dx_Ldx_L_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)
            return(comp_1+comp_2)
        elif (sde_operator==3):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_1=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_3=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Ldx_L_comp(x1[:,2], x2[:,2], ind=2)
            return(comp_1-comp_2-comp_3)
        elif (sde_operator==4 or sde_operator==5):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return ((torch.zeros(0),torch.zeros(0),torch.zeros(0)))
            if theta is None: theta=self.theta
            with torch.no_grad():
                comp_1=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.dC_dx_L_comp(x1[:,1], x2[:,1], ind=1)
                comp_3=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)
            return((comp_2, comp_3, comp_1))
        elif (sde_operator==6 or sde_operator==7 or sde_operator==8):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return ((torch.zeros(0),torch.zeros(0)))
            with torch.no_grad():
                comp_1=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_3=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Ldx_L_comp(x1[:,2], x2[:,2], ind=2)
                LK_all = torch.cat((comp_2+comp_3,comp_1),0)
            return(comp_2+comp_3, comp_1, LK_all)
        elif (sde_operator==7 or sde_operator==8):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_1=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_3=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Ldx_L_comp(x1[:,2], x2[:,2], ind=2)
            return(comp_1-comp_2-comp_3)
    def KL(self, x1, x2 = None, sde_operator=None,theta = None): # change here when operator changes: KL
        if (sde_operator is None): sde_operator=self.sde_operator
        if x2 is None: x2 = x1
        if (sde_operator==1):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_1=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.dC_dx_R_comp(x1[:,1], x2[:,1], ind=1)
            return (comp_1+comp_2)
        elif (sde_operator==2):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_1=self.d2C_dx_Rdx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)
            return(comp_1+comp_2)
        elif (sde_operator==3):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_1=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_3=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
            return(comp_1-comp_2-comp_3)
        elif (sde_operator==4 or sde_operator==5):
            if theta is None: theta=self.theta
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0),torch.zeros(0),torch.zeros(0))
            with torch.no_grad():
                comp_1=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.dC_dx_R_comp(x1[:,1], x2[:,1], ind=1)
                comp_3=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)
            return((comp_2, comp_3, comp_1))
        elif (sde_operator==6 or sde_operator==7 or sde_operator==8):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0), torch.zeros(0))
            with torch.no_grad():
                comp_1=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_3=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
                KL_all = torch.cat((comp_2+comp_3,comp_1),1)
            return(comp_2+comp_3, comp_1,KL_all)
        elif (sde_operator==7 or sde_operator==8):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_1=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_2=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_3=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
            return(comp_1-comp_2-comp_3)
    def LKL(self, x1, x2 = None, sde_operator=None, theta = None): 
        if (sde_operator is None): sde_operator=self.sde_operator
        if x2 is None: x2 = x1
        if (sde_operator==1):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_12=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.dC_dx_L_comp(x1[:,1], x2[:,1], ind=1)
                comp_21=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.dC_dx_R_comp(x1[:,1], x2[:,1], ind=1)
                comp_11=self.d2C_dx_Ldx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)
                comp_22=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_R_comp(x1[:,1], x2[:,1], ind=1)
            return(comp_11+comp_22+comp_12+comp_21)
        elif (sde_operator==2):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            comp_12=self.d2C_dx_Rdx_R_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)
            comp_21=self.d2C_dx_Ldx_L_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)
            comp_11=self.d4C_dx_Ldx_Ldx_Rdx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)
            comp_22=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d4C_dx_Ldx_Ldx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)
            return(comp_11+comp_22+comp_12+comp_21)
        elif (sde_operator==3): 
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_11=self.d2C_dx_Ldx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_12=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_13=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
                comp_21=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_22=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d4C_dx_Ldx_Ldx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_23=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
                comp_31=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Ldx_L_comp(x1[:,2], x2[:,2], ind=2)
                comp_32=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Ldx_L_comp(x1[:,2], x2[:,2], ind=2)
                comp_33=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d4C_dx_Ldx_Ldx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
            return(comp_11-comp_12-comp_13-comp_21+comp_22+comp_23-comp_31+comp_32+comp_33)
        elif (sde_operator==4 or sde_operator==5):
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0),torch.zeros(0),torch.zeros(0),torch.zeros(0),torch.zeros(0),torch.zeros(0),torch.zeros(0),torch.zeros(0),torch.zeros(0))
            if theta is None: theta=self.theta
            with torch.no_grad():
                comp_11=self.d2C_dx_Ldx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)
                comp_12=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.dC_dx_R_comp(x1[:,1], x2[:,1], ind=1)
                comp_13=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)
                comp_21=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.dC_dx_L_comp(x1[:,1], x2[:,1], ind=1)
                comp_22=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_R_comp(x1[:,1], x2[:,1], ind=1)
                comp_23=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d3C_dx_Ldx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)
                comp_31=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)
                comp_32=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d3C_dx_Ldx_Ldx_R_comp(x1[:,1], x2[:,1], ind=1)
                comp_33=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d4C_dx_Ldx_Ldx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1) 
            return((comp_22, comp_23, comp_21, comp_32, comp_33, comp_31, comp_12, comp_13, comp_11))
        elif (sde_operator==6 or sde_operator==7 or sde_operator==8): 
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0),torch.zeros(0),torch.zeros(0),torch.zeros(0))
            with torch.no_grad():
                comp_11=self.d2C_dx_Ldx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_12=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_13=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
                comp_21=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_22=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d4C_dx_Ldx_Ldx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_23=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
                comp_31=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Ldx_L_comp(x1[:,2], x2[:,2], ind=2)
                comp_32=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Ldx_L_comp(x1[:,2], x2[:,2], ind=2)
                comp_33=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d4C_dx_Ldx_Ldx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
                LKL_all = torch.cat((torch.cat((comp_22+comp_23+comp_32+comp_33,comp_21+comp_31),1),torch.cat((comp_12+comp_13,comp_11),1)),0)
            return(comp_22+comp_23+comp_32+comp_33, comp_11, comp_21+comp_31, comp_12+comp_13,LKL_all)
        elif (sde_operator==7 or sde_operator==8): 
            if x1.shape[0] == 0 or x2.shape[0] == 0 : return (torch.zeros(0))
            with torch.no_grad():
                comp_11=self.d2C_dx_Ldx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_12=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_13=self.dC_dx_L_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
                comp_21=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_22=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d4C_dx_Ldx_Ldx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.corr_comp(x1[:,2], x2[:,2], ind=2)
                comp_23=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Ldx_L_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
                comp_31=self.dC_dx_R_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Ldx_L_comp(x1[:,2], x2[:,2], ind=2)
                comp_32=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.d2C_dx_Rdx_R_comp(x1[:,1], x2[:,1], ind=1)*self.d2C_dx_Ldx_L_comp(x1[:,2], x2[:,2], ind=2)
                comp_33=self.corr_comp(x1[:,0], x2[:,0], ind=0)*self.corr_comp(x1[:,1], x2[:,1], ind=1)*self.d4C_dx_Ldx_Ldx_Rdx_R_comp(x1[:,2], x2[:,2], ind=2)
            return(comp_11-comp_12-comp_13-comp_21+comp_22+comp_23-comp_31+comp_32+comp_33)
    def dC_dx_L_comp(self, x1, x2 = None, ind=0):
        lengthscale = torch.exp(self.log_lengthscale[ind])
        if x2 is None: x2 = x1
        if x1.shape[0]==0 or x2.shape[0]==0 : 
            return (torch.zeros(0))
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        with torch.no_grad():
            #C_ = self.corr_comp(x1, x2, ind)
            dist_ = (x1.reshape(-1,1) - x2.reshape(1,-1))
            r_ = dist_.abs()
            r_ = np.sqrt(2.*self.nu) * r_ / lengthscale
            r_ = r_.clamp_(1e-15)
            C_ = np.power(2,1-self.nu)*np.exp(-fun.loggamma(self.nu))
            dC_ = C_ *( - torch.pow(r_,self.nu) * Bessel.apply(r_,self.nu-1) )
            dC_ = dC_ * np.sqrt(2*self.nu) / lengthscale  
            # limit at 0 is taken care by the sign function
            dC_ = dC_ * torch.sign(dist_)
        return (dC_)

    def dC_dx_R_comp(self, x1, x2 = None, ind=0):
        return (-self.dC_dx_L_comp(x1,x2, ind))

    def d2C_dx_Ldx_R_comp(self, x1, x2 = None, ind=0):
        lengthscale = torch.exp(self.log_lengthscale[ind])
        if x2 is None: x2 = x1
        if x1.shape[0]==0 or x2.shape[0]==0 : 
            return (torch.zeros(0))
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        with torch.no_grad():
            #C_ = self.corr_comp(x1, x2, ind)
            dist_ = (x1.reshape(-1,1) - x2.reshape(1,-1))
            r_ = dist_.abs()
            r_ = np.sqrt(2.*self.nu) * r_ / lengthscale
            r_ = r_.clamp_(1e-15)
            C_ = np.power(2,1-self.nu)*np.exp(-fun.loggamma(self.nu))
            dC2_ = C_ *( - torch.pow(r_,self.nu-1) * Bessel.apply(r_,self.nu-1) + torch.pow(r_,self.nu) * Bessel.apply(r_,self.nu-2))
            dC2_ = dC2_ * 2. * self.nu / torch.square(lengthscale).double()
            dC2_ = -dC2_
        return (dC2_)

    def d2C_dx_Ldx_L_comp(self, x1, x2 = None, ind=0):
        return (-self.d2C_dx_Ldx_R_comp(x1, x2, ind))

    def d2C_dx_Rdx_R_comp(self, x1, x2 = None, ind=0):
        return (-self.d2C_dx_Ldx_R_comp(x1, x2, ind))

    def d3C_dx_Ldx_Ldx_R_comp(self, x1, x2 = None, ind=0):
        lengthscale = torch.exp(self.log_lengthscale[ind])
        if x2 is None: x2 = x1
        if x1.shape[0]==0 or x2.shape[0]==0 : 
            return (torch.zeros(0))
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        with torch.no_grad():
            #C_ = self.corr_comp(x1, x2, ind)
            dist_ = (x1.reshape(-1,1) - x2.reshape(1,-1))
            r_ = dist_.abs()
            r_ = np.sqrt(2.*self.nu) * r_ / lengthscale
            r_ = r_.clamp_(1e-15)
            C_ = np.power(2,1-self.nu)*np.exp(-fun.loggamma(self.nu))
            dC3_ = C_ *( 3 * torch.pow(r_,self.nu-1) * Bessel.apply(r_,self.nu-2) - torch.pow(r_,self.nu) * Bessel.apply(r_,self.nu-3))
            dC3_ = dC3_ * torch.pow(np.sqrt(2.*self.nu) / lengthscale, 3)
            dC3_ = dC3_ * torch.sign(dist_)
        return (dC3_)

    def d3C_dx_Ldx_Rdx_R_comp(self, x1, x2 = None, ind=0):
        return (-self.d3C_dx_Ldx_Ldx_R_comp(x1, x2, ind))

    def d4C_dx_Ldx_Ldx_Rdx_R_comp(self, x1, x2 = None, ind=0):
        lengthscale = torch.exp(self.log_lengthscale[ind])
        if x2 is None: x2 = x1
        if x1.shape[0]==0 or x2.shape[0]==0 : 
            return (torch.zeros(0))
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        with torch.no_grad():
            #C_ = self.corr_comp(x1, x2, ind)
            dist_ = (x1.reshape(-1,1) - x2.reshape(1,-1))
            r_ = dist_.abs()
            r_ = np.sqrt(2.*self.nu) * r_ / lengthscale
            r_ = r_.clamp_(1e-15)
            C_ = np.power(2,1-self.nu)*np.exp(-fun.loggamma(self.nu))
            dC4_ = C_ *(3*torch.pow(r_,self.nu-2) * Bessel.apply(r_,self.nu-2)-6*torch.pow(r_,self.nu-1) * Bessel.apply(r_,self.nu-3)+torch.pow(r_,self.nu) * Bessel.apply(r_,self.nu-4))
            dC4_ = dC4_ * torch.square(2. * self.nu / torch.square(lengthscale).double())
        return (dC4_)

class GP_modeling(object):
    def __init__(self, PDEmodel, noisy = True, nu = 2.01, noisy_known = False):
        #super().__init__()
        #self.PDEmodel=PDEmodel
        #self.Train_GP(self, PDEmodel, noisy = True, max_iter = 500, verbose = False, eps=1e-6, nu=2.01, sigma_e=None, sde_operator=None)
        self.noisy=noisy
        self.nu=nu
        self.sigma_e=PDEmodel.sigma_e
        self.sde_operator=PDEmodel.sde_operator
        self.x_I=PDEmodel.x_I
        self.x_obs=PDEmodel.x_obs
        self.y_obs=PDEmodel.y_obs
        self.noisy_known = noisy_known

    def Feed_data(self, PDEmodel, ind_y):
        self.x_all=PDEmodel.x_obs
        self.y_all=PDEmodel.y_obs
        print('Standard diviation of data',(torch.var(self.y_all)).sqrt().numpy())
        self.y_all = self.y_all[:,ind_y]
        self.y_all = self.y_all.reshape(-1,1)
        return (self.x_all, self.y_all)
        #return (PDEmodel.x_obs, PDEmodel.y_obs)

    def Train_GP(self, PDEmodel, train_x = None, train_y = None, noisy = True, max_iter = 500, verbose = False, eps=1e-6, ind_y = 0, ind_aug = 0, optimize_alg = True, band_width = 1):
        time_GP_start=time.time()
        nu=self.nu
        sigma_e=PDEmodel.sigma_e[0][ind_y]
        if self.noisy_known is False:
            sigma_e = None
        sde_operator=self.sde_operator
        if train_x is not None: 
            n_obs = train_x.size(0)
            train_y = train_y.squeeze()
        else: 
            train_x, train_y = self.Feed_data(PDEmodel, ind_y)
            n_obs = self.x_obs.size(0)
        n = train_x.size(0)
        x_min = torch.min(train_x,0).values
        x_max = torch.max(train_x,0).values
        x_range=x_max-x_min
        scale_x =  np.log (x_range).numpy()
        if (len(train_x.shape)==1): d=1
        else: d=train_x.shape[1]
        kernel = Matern_Product(nu, lengthscale = 1*np.ones(d), sde_operator=sde_operator)
        self.kernel = kernel
        # set up optimizer
        if optimize_alg == True:
            bnds=((np.log(2e-6),np.log(1e-1)),)
            start_1=(np.log(1e-4),)
            start_2=(np.log(1e-4),)
            for i in range(d):
                bnds=bnds+((-2+scale_x[i],1+scale_x[i]),)
                start_1=start_1+(0+scale_x[i],)
                start_2=start_2+(-1+scale_x[i],)
            print('Optimizing GP parameter:')
            res1 = minimize(self.Loss_mle,start_1, args=(train_x,train_y,sigma_e,kernel, n_obs,ind_aug), method='Nelder-Mead', bounds=bnds, options={'ftol': 1e-6,})
            res2 = minimize(self.Loss_mle,start_2, args=(train_x,train_y,sigma_e,kernel, n_obs,ind_aug), method='Nelder-Mead', bounds=bnds, options={'ftol': 1e-6,})
            if res1['fun']<res2['fun']:
                res=res1
            else:
                res=res2
            if res['x'][0] >0.05 : res['x'] = start_2
            print('Initial Optimization:', np.exp(res['x']))
            for i in range(d):
                if res['x'][i+1] < -2+scale_x[i]: res['x'][i+1] = -2+scale_x[i]
            lengthscale_ini=np.exp(res['x'][1 : d+1])
            kernel = Matern_Product(nu, lengthscale = lengthscale_ini, sde_operator=sde_operator)

            
            if (noisy is True): # means observation has noise
                # lambda = noise/outputscale
                log_lambda = torch.min(torch.tensor(res['x'][0]), torch.tensor(np.log(1e-3)))
                log_lambda.requires_grad_(True)
                optimizer = torch.optim.Adam([kernel.log_lengthscale,log_lambda], lr=1e-1)
            else:
                # nugget term to avoid numerical issue
                log_lambda = torch.tensor(np.log (1e-6))
                optimizer = torch.optim.Adam([kernel.log_lengthscale], lr=1e-1)
                sigma_e = None
            # training
            prev_loss = np.Inf
            for i in range(max_iter):
                optimizer.zero_grad()
                err_term=torch.cat((torch.exp(log_lambda)*torch.ones(n_obs), 1e-6 * torch.ones(n-n_obs)),0)
                R = kernel.forward(train_x) + torch.diag(err_term)   
                #print(torch.exp(log_lambda))
                #print(R)             
                e,v = torch.linalg.eig(R)
                e = torch.real(e) # eigenvalues
                v = torch.real(v) 
                #print(e)
                #print(v)
                #e,v = torch.eig(R, eigenvectors = True)
                #e = e[:,0] # eigenvalues
                a = v.T @ torch.ones(n)
                b = v.T @ train_y
                mean = ((a/e) @ b) / ((a/e) @ a) * (ind_aug == 0)
                err = v.T @ (train_y - mean)
                if sigma_e is None:
                    outputscale = 1./n * (err/e) @ err
                    loss = torch.log(outputscale) + torch.mean(torch.log(e))     
                    #Rinv=torch.inverse(R)
                    #loss=torch.mean(torch.square(torch.inverse(torch.diag(torch.diag(Rinv))) @ Rinv @ (train_y - mean)))
                else : 
                    outputscale =  (sigma_e**2)/torch.exp(log_lambda)
                    loss = -log_lambda + torch.mean(torch.log(e)) + torch.exp(log_lambda)/(n*sigma_e**2) * (err/e) @ err
                    #Rinv=torch.inverse(R)
                    #loss=torch.mean(torch.square(torch.inverse(torch.diag(torch.diag(Rinv))) @ Rinv @ (train_y - mean)))
                
                loss = loss + 1e6 * torch.exp((np.log(1e-6)-log_lambda) * (log_lambda < np.log(1e-6))) # penealty for ratio of GP variance and error varaince
                loss = loss + 1e6 * (torch.max(kernel.log_lengthscale-torch.tensor(scale_x)) > 1) + 1e6 * (torch.min(kernel.log_lengthscale-torch.tensor(scale_x)) < -3)# penalty for lengthscale
                if i==0:
                    loss_opt=loss.clone()
                    log_lengthscale_opt=kernel.log_lengthscale.clone().detach()
                    if (noisy is True): log_lambda_opt=log_lambda.clone().detach()
                else:
                    if loss<loss_opt:
                        loss_opt=loss.clone()
                        log_lengthscale_opt=kernel.log_lengthscale.clone().detach()
                        if (noisy is True): log_lambda_opt=log_lambda.clone().detach()
                loss.backward()
                optimizer.step()

                # early termination check every 10 iterations
                if ((i+1)%10 == 0):
                    if (verbose):
                        print('Iter %d/%d - Loss: %.3f' % (i+1, max_iter, loss.item()))
                    if (np.nan_to_num((prev_loss-loss.item())/abs(prev_loss),nan=1) > eps):
                        prev_loss = loss.item()
                    else:
                        if (verbose): print('Early Termination!')
                        break
            kernel.log_lengthscale=log_lengthscale_opt
            if (noisy is True): log_lambda=log_lambda_opt
        else: 
            self.Reset_Hyper_parameters(ind = ind_aug, ind_y = ind_y, band_width = band_width )
            log_lambda = self.log_lambda

        n , d = train_x.shape
        err_term=torch.cat((torch.exp(log_lambda)*torch.ones(n_obs), 1e-6 * torch.ones(n-n_obs)),0)
        R = kernel.K(train_x) + torch.diag(err_term)
        Rinv = torch.inverse(R)
        ones = torch.ones(n)
        mean = ((ones @ Rinv @ train_y) / (ones @ Rinv @ ones)).item() * (ind_aug == 0)
        if (sigma_e is not None and noisy is True): 
            outputscale = ((sigma_e**2)/torch.exp(log_lambda))
            noisescale=sigma_e**2
        if (sigma_e is None or noisy is False): 
            outputscale = (1/n * (train_y - mean) @ Rinv @ (train_y - mean))
            noisescale = outputscale * torch.exp(log_lambda)
        kernel.log_lengthscale.requires_grad_(False)
        print('GP_var', outputscale, 'Noise_var', noisescale,'GP_parameter', torch.exp(kernel.log_lengthscale).numpy(),'Standardized', torch.exp(kernel.log_lengthscale-scale_x).numpy())

        self.mean=mean
        self.outputscale=outputscale.squeeze().detach()
        self.noisescale=noisescale.squeeze().detach()
        self.kernel=kernel
        self.R=R
        time_GP_end=time.time()
        print('Time for training GP:', int(time_GP_end-time_GP_start), ' secs')

    def Reset_Hyper_parameters(self,ind = 0,ind_y = 0, band_width = 1.):
        hyper_para = torch.tensor([band_width])
        log_lambda = torch.log(torch.tensor(1e-9))
        self.kernel.log_lengthscale = torch.log(hyper_para)
        self.log_lambda = log_lambda            


    def Loss_mle(self, x, train_x, train_y, sigma_e, kernel, n_obs,ind_aug=0):
        log_lambda=torch.tensor(x[0])
        n , d = train_x.shape
        kernel.log_lengthscale=torch.tensor(x[1:d+1])
        err_term=torch.cat((torch.exp(log_lambda)*torch.ones(n_obs), 1e-6 * torch.ones(n-n_obs)),0)
        R = kernel.K(train_x) + torch.diag(err_term)
        e,v = torch.linalg.eig(R)
        e = torch.real(e) # eigenvalues
        v = torch.real(v) 
        a = v.T @ torch.ones(n)
        b = v.T @ train_y
        mean = ((a/e) @ b) / ((a/e) @ a) * (ind_aug ==0 )
        d = v.T @ (train_y - mean)
        if sigma_e is None:
            outputscale = 1./n * (d/e) @ d
            loss = torch.log(outputscale) + torch.mean(torch.log(e))     
            #Rinv=torch.inverse(R)
            #loss=torch.mean(torch.square(torch.inverse(torch.diag(torch.diag(Rinv))) @ Rinv @ (train_y - mean)))
        else : 
            outputscale =  (sigma_e**2)/torch.exp(log_lambda)
            loss = -log_lambda + torch.mean(torch.log(e)) + 1/(n_obs * outputscale) * (d/e) @ d 
            #Rinv=torch.inverse(R)
            #loss=torch.mean(torch.square(torch.inverse(torch.diag(torch.diag(Rinv))) @ Rinv @ (train_y - mean)))
        loss = loss + 1e6 * (log_lambda < np.log(1e-6)) # penealty for log lambda
        #print(loss)
        return(loss.squeeze().numpy())





