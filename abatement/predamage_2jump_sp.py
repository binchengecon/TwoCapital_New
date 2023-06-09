"""
pre_damage.py
=================
Solver for pre damage HJBs, tech III, tech I
"""
# Optimization of post jump HJB
#Required packages
import os
import sys
sys.path.append('./src')
import csv
from supportfunctions import *
from supportfunctions import finiteDiff_3D
sys.stdout.flush()
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime
# from solver import solver_3d
from PostSolver import hjb_post_damage_post_tech, hjb_pre_damage_post_tech
from src.solver_sp import pde_one_interation
from src.solver_sp import hjb_pre_tech
import argparse

reporterror = True
# Linear solver choices
# Chosse among petsc, petsc4py, eigen, both
# petsc: matrix assembled in C
# petsc4py: matrix assembled in Python
# eigen: matrix assembled in C++
# both: petsc+petsc4py
#
now = datetime.now()
current_time = now.strftime("%d-%H:%M")

parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--xi_a", type=float, default=1000.)
parser.add_argument("--xi_p", type=float, default=1000.)
parser.add_argument("--psi_0", type=float, default=0.003)
parser.add_argument("--psi_1", type=float, default=0.5)
parser.add_argument("--psi_2", type=float, default=0.4)
parser.add_argument("--s", type=float, default=0.0)
parser.add_argument("--tau", type=float, default=0.0)
parser.add_argument("--Tr", type=float, default=0.0)
parser.add_argument("--num_gamma",type=int,default=6)
parser.add_argument("--name",type=str,default="ReplicateSuri")
parser.add_argument("--hXarr",nargs='+',type=float)
parser.add_argument("--Xminarr",nargs='+',type=float)
parser.add_argument("--Xmaxarr",nargs='+',type=float)
parser.add_argument("--epsilonarr",nargs='+',type=float)
parser.add_argument("--fractionarr",nargs='+',type=float)
parser.add_argument("--maxiterarr",nargs='+',type=int)

args = parser.parse_args()


epsilonarr = args.epsilonarr
fractionarr = args.fractionarr
maxiterarr = args.maxiterarr


start_time = time.time()
# Parameters as defined in the paper
xi_a = args.xi_a # Smooth ambiguity
xi_p = args.xi_p  # Damage poisson
xi_b = 1000. # Brownian misspecification
xi_g = args.xi_p  # Technology jump

# DataDir = "./res_data/6damage/xi_a_" + str(xi_a) + "_xi_g_" + str(xi_g) +  "/"
# if not os.path.exists(DataDir):
    # os.mkdir(DataDir)

# Model parameters
delta   = 0.010
alpha   = 0.115
kappa   = 6.667
mu_k    = -0.043
sigma_k = np.sqrt(0.0087**2 + 0.0038**2)
# Technology
theta        = 3
lambda_bar   = 0.1206
vartheta_bar = 0.0453
# Damage function
gamma_1 = 1.7675/10000
gamma_2 = 0.0022 * 2
# gamma_3 = 0.3853 * 2

num_gamma = args.num_gamma
gamma_3_list = np.linspace(0,1./3.,num_gamma)


y_bar = 2.
y_bar_lower = 1.5


s_star = args.s
tau_star = args.tau
Tr_star=args.Tr

theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
sigma_y   = 1.2 * np.mean(theta_ell)
beta_f    = 1.86 / 1000
# Jump intensity
zeta      = 0.00
psi_0     = args.psi_0
psi_1     = args.psi_1
psi_2     = args.psi_2

sigma_g   = 0.016
# Tech jump
lambda_bar_first = lambda_bar / 2
vartheta_bar_first = vartheta_bar / 2
lambda_bar_second = 1e-9
vartheta_bar_second = 0.

Xminarr = args.Xminarr
Xmaxarr = args.Xmaxarr
hXarr = args.hXarr

K_min = Xminarr[0]
K_max = Xmaxarr[0]
hK    = hXarr[0]
K     = np.arange(K_min, K_max + hK, hK)
nK    = len(K)
Y_min = Xminarr[1]
Y_max = Xmaxarr[1]
hY    = hXarr[1] # make sure it is float instead of int
Y     = np.arange(Y_min, Y_max + hY, hY)
nY    = len(Y)
L_min = Xminarr[2]
L_max = Xmaxarr[2]
hL    = hXarr[2]
L     = np.arange(L_min, L_max+hL,  hL)
nL    = len(L)

X1     = K
nX1    = len(X1)
hX1    = X1[1] - X1[0]
X1_min = X1.min()
X1_max = X1.max()
X2     = Y
nX2    = len(X2)
hX2    = X2[1] - X2[0]
X2_min = X2.min()
X2_max = X2.max()
X3     = L
nX3    = len(X3)
hX3    = X3[1] - X3[0]
X3_min = X3.min()
X3_max = X3.max()


Output_Dir = "/scratch/bincheng/"
Data_Dir = Output_Dir+"abatement/data_2tech/"+args.name+"/"

File_Name = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)

os.makedirs(Data_Dir, exist_ok=True)

# if not os.path.exists(DataDir):
print("Grid dimension: [{}, {}, {}]\n".format(nX1, nX2, nX3))
print("Grid step: [{}, {}, {}]\n".format(hX1, hX2, hX3))

# Discretization of the state space for numerical PDE solution.
######## post jump, 3 states
(X1_mat, X2_mat, X3_mat) = np.meshgrid(X1, X2, X3, indexing = 'ij')
stateSpace = np.hstack([X1_mat.reshape(-1,1,order = 'F'), X2_mat.reshape(-1,1,order = 'F'), X3_mat.reshape(-1, 1, order='F')])
K_mat = X1_mat
Y_mat = X2_mat
L_mat = X3_mat
# For PETSc
X1_mat_1d = X1_mat.ravel(order='F')
X2_mat_1d = X2_mat.ravel(order='F')
X3_mat_1d = X3_mat.ravel(order='F')
lowerLims = np.array([X1_min, X2_min, X3_min], dtype=np.float64)
upperLims = np.array([X1_max, X2_max, X3_max], dtype=np.float64)


# Post damage, tech II
print("-------------------------------------------")
print("------------Post damage, Tech II----------")
print("-------------------------------------------")
model_tech2_post_damage =  []
for i in range(len(gamma_3_list)):
    gamma_3_i = gamma_3_list[i]
    model_i = pickle.load(open(Data_Dir+ File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb"))
    model_tech2_post_damage.append(model_i)

# model_tech3_post_damage.append(v_post_i)
with open(Data_Dir+ File_Name + "model_tech2_post_damage", "wb") as f:
    pickle.dump(model_tech2_post_damage, f)

model_tech2_post_damage = pickle.load(open(Data_Dir+ File_Name + "model_tech2_post_damage", "rb"))
print("Compiled.")

# Post damage, tech I
print("-------------------------------------------")
print("------------Post damage, Tech I-----------")
print("-------------------------------------------")
model_tech1_post_damage = []
for i in range(len(gamma_3_list)):
    gamma_3_i = gamma_3_list[i]
    model_i = pickle.load(open(Data_Dir+ File_Name + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb"))
    model_tech1_post_damage.append(model_i)

with open(Data_Dir+ File_Name + "model_tech1_post_damage", "wb") as f:
    pickle.dump(model_tech1_post_damage, f)

model_tech1_post_damage = pickle.load(open(Data_Dir+ File_Name + "model_tech1_post_damage", "rb"))
print("Compiled.")

# # delete the separate files
# for i in range(len(gamma_3_list)):
#     gamma_3_i = gamma_3_list[i]
#     # Tech II
#     model_i_dir = Data_Dir+ File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i) 
#     if os.path.exists(model_i_dir):
#         os.remove(model_i_dir)
#     # Tech I
#     model_i_dir = Data_Dir+ File_Name  + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i) 
#     if os.path.exists(model_i_dir):
#         os.remove(model_i_dir)


print("-------------------------------------------")
print("---------Pre damage, Tech II--------------")
print("-------------------------------------------")
id_2 = np.abs(Y - y_bar).argmin()
Y_min_short = Xminarr[3]
Y_max_short = Xmaxarr[3]
Y_short     = np.arange(Y_min_short, Y_max_short + hY, hY)
nY_short    = len(Y_short)
# Pre damage, tech III
# # Compile v_i
v_i = []
for i in range(len(gamma_3_list)):
    gamma_3_i = gamma_3_list[i]
    v_post_damage_i = np.zeros((nK, nY_short))
    for j in range(nY_short):
        v_post_damage_i[:, j] = model_tech2_post_damage[i]["v"][:, id_2]

    v_i.append(v_post_damage_i)

v_i = np.array(v_i)
pi_d_o = np.ones(len(gamma_3_list)) / len(gamma_3_list)
pi_d_o = np.array([temp * np.ones((nK, nY_short)) for temp in pi_d_o])

theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
pi_c_o = np.array([temp * np.ones((nK, nY_short)) for temp in pi_c_o])
theta_ell = np.array([temp * np.ones((nK, nY_short)) for temp in theta_ell])

################################
####Start of Compute############
################################

model_tech2_pre_damage = hjb_pre_damage_post_tech(
        K, Y_short, 
        model_args=(delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, xi_p, pi_d_o, v_i, gamma_1, gamma_2, theta, lambda_bar_second, vartheta_bar_second, y_bar_lower),
        v0=np.mean(v_i, axis=0), epsilon=epsilonarr[0], fraction=fractionarr[0],
        tol=1e-8, max_iter=maxiterarr[0], print_iteration=True
        )

with open(Data_Dir+ File_Name + "model_tech2_pre_damage", "wb") as f:
    pickle.dump(model_tech2_pre_damage, f)

model_tech2_pre_damage = pickle.load(open(Data_Dir+ File_Name + "model_tech2_pre_damage", "rb"))
######################################
##########End of Compute##############
######################################


# Pre damage, tech I
pi_d_o = np.ones(len(gamma_3_list)) / len(gamma_3_list)
pi_d_o = np.array([temp * np.ones((nK, nY_short)) for temp in pi_d_o])

theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
pi_c_o = np.array([temp * np.ones((nK, nY_short, nL)) for temp in pi_c_o])
pi_c = pi_c_o.copy()
theta_ell = np.array([temp * np.ones((nK, nY_short, nL)) for temp in theta_ell])


v_i = []
for model in model_tech1_post_damage:
    v_post_damage_i = model["v0"]
    v_post_damage_temp = np.zeros((nK, nY_short, nL))
    for j in range(nY_short):
        v_post_damage_temp[:, j, :] = v_post_damage_i[:, id_2, :]
    v_i.append(v_post_damage_temp)
v_i = np.array(v_i)
v_post = model_tech2_pre_damage["v"][:, :nY_short]
v_tech2 = np.zeros((nK, nY_short, nL))
for i in range(nL):
    v_tech2[:, :, i] = v_post

model_args =(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, v_tech2, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p,s_star,tau_star,Tr_star)

#########################################
######### Start of Compute###############
#########################################

# Guess = pickle.load(open("./res_data/6damage/psi_0_{:.3f}_psi_1_{:.3f}/xi_a_1000.0_xi_g_1000.0/model_tech2_pre_damage".format(psi_0, psi_1), "rb"))
Guess = None
model_tech1_pre_damage = hjb_pre_tech(
        state_grid=(K, Y_short, L), 
        model_args=model_args, V_post_damage=v_i, 
        tol=1e-8, epsilon=epsilonarr[1], fraction=fractionarr[1], max_iter=maxiterarr[1],
        v0=np.mean(v_i, axis=0),
        smart_guess=Guess,
        )


File_Name_sp = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_psi_2_{}_s_{}_tau_{}_Tr_{}_" .format(xi_a,xi_g,psi_0,psi_1,psi_2,s_star,tau_star,Tr_star)

with open(Data_Dir+ File_Name_sp + "model_tech1_pre_damage", "wb") as f:
    pickle.dump(model_tech1_pre_damage, f)

