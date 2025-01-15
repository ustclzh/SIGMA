import traceback
import numpy as np
from scripts import SIGMA # inferred module
from scripts import SDE_Model # inferred module
from scipy.io import savemat
import argparse
from concurrent.futures import ProcessPoolExecutor, wait
import torch
import mkl
mkl.set_num_threads(1)
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='specify input parameters')
parser.add_argument('--operator', '-op', help='SDE operator', default=1, type = int)
parser.add_argument('--inst_num', '-in', help='number of instance', default=1, type = int)
parser.add_argument('--num', '-nu', help='number of instance', default=1, type = int)
# parser.add_argument('--boundary_condition', '-b', help='use boundary condition or not', default=0, type = int)
# parser.add_argument('--noisy_known', '-n', help='assumes sigma_e known or not', default=1, type = int)
# parser.add_argument('--sigma_e', '-e', help='sigma_e', default=0.001, type = float)
parser.add_argument('--n_obs', '-o', help='number of observation point', default=100, type = int)
parser.add_argument('--n_I', '-i', help='number of discretization set',default=100, type = int)
# parser.add_argument('--algorithm', '-a', help='algorithm, 1 for LBFGS, 2 for Adam', default=1, type = int)
# parser.add_argument('--batch_num', '-ba', help='batch_num', default=0, type = int)
# parser.add_argument('--num_run', '-nr', help='number of discretization set',default=30, type = int)
args = parser.parse_args()

# operator = -4
n_I = 100
# n_obs = 10
num = args.num
print(f"Process {num} started.")

n_I = 100
# n_obs = 10

cand_op = np.array([0,2,3,4,-1,-3,-4])
cand_obs = np.array([10,20,25,50,100])
operator = cand_op[int(num//25)]
num = num%25
n_obs = cand_obs[int(num//5)]
inst_num = num%5


sigma_e = 0.000000001
SDE=SDE_Model.SDE_Module(
    sde_operator = operator,
    sigma_e = sigma_e,
    n_I = n_I,
    n_obs = n_obs,
    noisy_known = True,
    optimize_alg = False,
    band_width = 1.
    )

SIGMA_o= SIGMA.SIGMA(SDE, para = False ,rho= 1.) # call inference class
MAP_margin = SIGMA_o.map_margin(nEpoch = 3000, opt_algorithm = 2, center_modify = False)
MAP_margin_c = SIGMA_o.map_margin(nEpoch = 3000, opt_algorithm = 2, center_modify = True)
MAP = SIGMA_o.map(nEpoch = 3000, opt_algorithm = 2, center_modify = False)
MAP_c = SIGMA_o.map(nEpoch = 3000, opt_algorithm = 2, center_modify = True)
MLE = SIGMA_o.mle_joint(nEpoch = 3000, opt_algorithm = 2)
res_hmc_lkh = SIGMA_o.HMC_lkh(
        n_epoch = 500, 
        lsteps=200, 
        epsilon=5e-5, 
        n_samples=2000, 
        opt_algorithm = 2
        )
sample = res_hmc_lkh['samples']
res_hmc_c,map = SIGMA_o.Sample_Using_HMC(
        n_epoch = 500, 
        lsteps=100, 
        epsilon=1e-4, 
        n_samples=2000, 
        opt_algorithm = 2,
        center_modify = True
        )
sample_post_c = res_hmc_c['samples']
res_hmc,map = SIGMA_o.Sample_Using_HMC(
        n_epoch = 500, 
        lsteps=100, 
        epsilon=1e-4, 
        n_samples=2000, 
        opt_algorithm = 2,
        center_modify = False
        )
sample_post = res_hmc['samples']
theta_mle = SIGMA_o.theta_MLE
theta_mle_a = SIGMA_o.theta_MLE_a
# SIGMA_p= SIGMA.SIGMA(SDE, para = True ,rho= 1.) # call inference class
# res_margin = SIGMA_p.map_margin(nEpoch = 3000, opt_algorithm = 2)
# res = SIGMA_p.map(nEpoch = 3000, opt_algorithm = 2)


filename =  'res/' + 'mcmc_source' + str(operator) + 'inst' + str(inst_num) + 'nobs' + str(n_obs) + 'n_I'+ str(n_I) + 'mcmc.mat'
ERR_summary = {
    'sample_lkh':sample,
    'sample_post':sample_post,
    'sample_post_c':sample_post_c,
    'theta_true': SDE.theta_true.numpy(),
    'theta_MLE': theta_mle,
    'theta_MLE_a':theta_mle_a,
    'theta_MLE_joint': MLE['theta_mle'].numpy(),
    'theta_MAP':MAP['theta_MAP'].numpy(),
    'theta_MAP_margin' : MAP_margin['theta_MAP'].numpy(),
    'theta_MAP_c':MAP_c['theta_MAP'].numpy(),
    'theta_MAP_margin_c' : MAP_margin_c['theta_MAP'].numpy(),
}
savemat(filename,ERR_summary)







