"""
pre_damage.py
=================
Solver for pre damage HJBs, tech III, tech I

srun python3 /home/bcheng4/TwoCapital_Shrink/abatement_UD/predamage_2jump_CRS.py --num_gamma 5 --xi_a 0.0002 --xi_p 0.050  --epsilonarr 0.05 0.05  --fractionarr 0.1 0.05   --maxiterarr 80000 200000  --psi_0 0.105830 --psi_1 0.5    --name 2jump_step_4.00,9.00_0.0,4.0_1.0,6.0_SS_0.2,0.2,0.2_LR_0.1_CRS_PETSCFK --hXarr 0.2 0.2 0.2 --Xminarr 4.00 0.0 1.0 0.0 --Xmaxarr 9.00 4.0 6.0 3.0

"""
# Optimization of post jump HJB
#Required packages
import os
import sys
sys.path.append('./src')
import csv
from src.Utility import *
from src.Utility import finiteDiff_3D
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
from src.PreSolver_CRS_FK_Tech import fk_pre_tech, fk_pre_tech_petsc
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
parser.add_argument("--xi_k", type=float, default=1000.)
parser.add_argument("--xi_c", type=float, default=1000.)
parser.add_argument("--xi_j", type=float, default=1000.)
parser.add_argument("--xi_d", type=float, default=1000.)
parser.add_argument("--xi_g", type=float, default=1000.)
parser.add_argument("--varrho", type=float, default=1000.)
parser.add_argument("--phi_0", type=float)
parser.add_argument("--rho", type=float)
parser.add_argument("--delta", type=float)
parser.add_argument("--psi_0", type=float, default=0.003)
parser.add_argument("--psi_1", type=float, default=0.5)
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
xi_b = 1000. # Brownian misspecification
xi_k = args.xi_k  # Technology jump
xi_c = args.xi_c  # Technology jump
xi_j = args.xi_j  # Technology jump
xi_d = args.xi_d # Hold place for arguments, no real effects 
xi_g = args.xi_g # Hold place for arguments, no real effects 
varrho = args.varrho # Hold place for arguments, no real effects 
rho = args.rho

# DataDir = "./res_data/6damage/xi_a_" + str(xi_a) + "_xi_g_" + str(xi_g) +  "/"
# if not os.path.exists(DataDir):
    # os.mkdir(DataDir)

# Model parameters
delta   = args.delta
alpha   = 0.115
kappa   = 6.667
mu_k    = -0.043
# sigma_k = np.sqrt(0.0087**2 + 0.0038**2)
sigma_k = 0.0100


# Technology
theta        = 3
lambda_bar   = 0.1206
# vartheta_bar = 0.0453
# vartheta_bar = 0.05
# vartheta_bar = 0.056
# vartheta_bar = 0.5
vartheta_bar = args.phi_0

# Damage function
gamma_1 = 1.7675/10000
gamma_2 = 0.0022 * 2
# gamma_3 = 0.3853 * 2

num_gamma = args.num_gamma
gamma_3_list = np.linspace(0,1./3.,num_gamma)


y_bar = 2.
y_bar_lower = 1.5


theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
sigma_y   = 1.2 * np.mean(theta_ell)
beta_f    = 1.86 / 1000
# Jump intensity
zeta      = 0.00
psi_0     = args.psi_0
psi_1     = args.psi_1
# sigma_g   = 0.016
sigma_g   = 0.0078
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

File_Name = "xi_a_{}_xi_k_{}_xi_c_{}_xi_j_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_rho_{}_delta_{}_" .format(xi_a, xi_k, xi_c, xi_j, xi_d, xi_g, psi_0,psi_1, varrho, rho, delta)

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


id_2 = np.abs(Y - y_bar).argmin()
Y_min_short = Xminarr[3]
Y_max_short = Xmaxarr[3]
Y_short     = np.arange(Y_min_short, Y_max_short + hY, hY)
nY_short    = len(Y_short)


# Post damage, tech I
print("-------------------------------------------")
print("------------Post damage, Tech I-----------")
print("-------------------------------------------")

model_tech1_post_damage = pickle.load(open(Data_Dir+ File_Name + "model_tech1_post_damage", "rb"))

Phi_m = []
for model in model_tech1_post_damage:
    post_damage_i = model["v0"]
    post_damage_temp = np.zeros((nK, nY_short, nL))
    for j in range(nY_short):
        post_damage_temp[:, j, :] = post_damage_i[:, id_2, :]
    Phi_m.append(post_damage_temp)
Phi_m = np.array(Phi_m)


print("Compiled.")

print("-------------------------------------------")
print("---------Pre damage, Tech II--------------")
print("-------------------------------------------")


model_tech2_pre_damage = pickle.load(open(Data_Dir+ File_Name + "model_tech2_pre_damage", "rb"))
Phi_II = model_tech2_pre_damage["v0"]


print("-------------------------------------------")
print("---------Pre damage, Tech I--------------")
print("-------------------------------------------")


model_tech1_pre_damage = pickle.load(open(Data_Dir+ File_Name + "model_tech1_pre_damage", "rb"))
Phi = model_tech1_pre_damage['v0']
ii = model_tech1_pre_damage['i_star']
e = model_tech1_pre_damage['e_star']
x = model_tech1_pre_damage['x_star']
pi_c = model_tech1_pre_damage['pi_c']
g_tech = model_tech1_pre_damage['g_tech']
g_damage = model_tech1_pre_damage['g_damage']
h = model_tech1_pre_damage['h']
h_k = model_tech1_pre_damage['h_k']
h_j = model_tech1_pre_damage['h_j']



(K_Short_mat, Y_Short_mat, L_Short_mat) = np.meshgrid(K, Y_short, L, indexing = 'ij')


j = alpha*vartheta_bar*(1-e/(lambda_bar * alpha * np.exp(K_Short_mat)))**theta
j[j<=1e-16] = 1e-16
c = alpha-ii-x-j

print(c.min(),c.max(),flush=True)


pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
pi_c_o = np.array([temp * np.ones((nK, nY_short, nL)) for temp in pi_c_o])
theta_ell = np.array([temp * np.ones((nK, nY_short, nL)) for temp in theta_ell])

print("Compiled.")
# print((Phi_II-Phi).max())

# Post damage, tech I
print("-------------------------------------------")
print("------------Load FK Distorted: Post damage, Tech I-----------")
print("-------------------------------------------")


# FK_model_tech1_post_damage = []

# for i in range(len(gamma_3_list)):
#     gamma_3_i = gamma_3_list[i]
#     model_i = pickle.load(open(Data_Dir+ File_Name + "FK_Distorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb"))
#     FK_model_tech1_post_damage.append(model_i)



# with open(Data_Dir+ File_Name + "FK_Distorted_model_tech1_post_damage", "wb") as f:
#     pickle.dump(FK_model_tech1_post_damage, f)

# FK_Distorted_model_tech1_post_damage = pickle.load(open(Data_Dir+ File_Name + "FK_Distorted_model_tech1_post_damage", "rb"))


# F_m = []
# for model in FK_Distorted_model_tech1_post_damage:
#     # print(model.keys())
#     F_post_damage_i = model["v0"]
#     F_post_damage_temp = np.zeros((nK, nY_short, nL))
#     for j in range(nY_short):
#         F_post_damage_temp[:, j, :] = F_post_damage_i[:, id_2, :]
#     F_m.append(F_post_damage_temp)
# F_m = np.array(F_m)
# print("Compiled.")

F_m = []
for i_temp in range(len(gamma_3_list)):
    gamma_3_i = gamma_3_list[i_temp]
    
    model_i = pickle.load(open(Data_Dir+ File_Name + "FK_Distorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb"))
    F_post_damage_i = model_i["v0"]
    F_post_damage_temp = np.zeros((nK, nY_short, nL))
    for j_temp in range(nY_short):
        F_post_damage_temp[:, j_temp, :] = F_post_damage_i[:, id_2, :]
    F_m.append(F_post_damage_temp)
    
    

F_m = np.array(F_m)


print("Compiled.")

print("-------------------------------------------")
print("------------Load FK Distorted: Pre damage, Tech II-----------")
print("-------------------------------------------")

F_II = np.zeros_like(Phi_II)


print("Compiled.")



print("-------------------------------------------")
print("------------FK Pre damage, Distorted----------")
print("-------------------------------------------")


# FK_Distorted_model_tech1_pre_damage = fk_pre_tech(
FK_Distorted_model_tech1_pre_damage = fk_pre_tech_petsc(
        state_grid=(K, Y_short, L), 
        model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_k, xi_c, xi_j, xi_d, xi_g,rho, varrho),
        control=(ii, e, x, pi_c, g_tech, g_damage, h, h_k, h_j),
        VF = (Phi_II, Phi, F_II, F_m)
        )


with open(Data_Dir+ File_Name + "FK_Distorted_model_tech1_pre_damage", "wb") as f:
    pickle.dump(FK_Distorted_model_tech1_pre_damage, f)






print("-------------------------------------------")
print("------------Load FK Undistorted: Post damage, Tech I-----------")
print("-------------------------------------------")
FK_Undistorted_model_tech1_post_damage = []
for i_temp in range(len(gamma_3_list)):
    gamma_3_i = gamma_3_list[i_temp]
    model_i = pickle.load(open(Data_Dir+ File_Name + "FK_Undistorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb"))
    FK_Undistorted_model_tech1_post_damage.append(model_i)

with open(Data_Dir+ File_Name + "FK_Undistorted_model_tech1_post_damage", "wb") as f:
    pickle.dump(FK_Undistorted_model_tech1_post_damage, f)

FK_Undistorted_model_tech1_post_damage = pickle.load(open(Data_Dir+ File_Name + "FK_Undistorted_model_tech1_post_damage", "rb"))




F_m_undis = []
for model in FK_Undistorted_model_tech1_post_damage:
    # print(model.keys())
    F_post_damage_i = model["v0"]
    F_post_damage_temp = np.zeros((nK, nY_short, nL))
    for j_temp in range(nY_short):
        F_post_damage_temp[:, j_temp, :] = F_post_damage_i[:, id_2, :]
    F_m_undis.append(F_post_damage_temp)
F_m_undis = np.array(F_m)

print("Compiled.")

print("-------------------------------------------")
print("------------Load FK Undistorted: Pre damage, Tech II-----------")
print("-------------------------------------------")

F_II_undis = np.zeros((nK, nY_short, nL))



print("-------------------------------------------")
print("------------FK Pre damage, Undistorted----------")
print("-------------------------------------------")


g_tech_undis = np.ones_like(g_tech)
g_damage_undis = np.ones_like(g_damage)
h_undis  = np.zeros_like(h)
h_k_undis  = np.zeros_like(h_k)
h_j_undis  = np.zeros_like(h_j)


# FK_Undistorted_model_tech1_pre_damage = fk_pre_tech(
FK_Undistorted_model_tech1_pre_damage = fk_pre_tech_petsc(
        state_grid=(K, Y_short, L), 
        model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_k, xi_c, xi_j, xi_d, xi_g,rho, varrho),
        control=(ii,e,x,pi_c, g_tech_undis, g_damage_undis, h_undis, h_k_undis, h_j_undis),
        VF = (Phi_II, Phi, F_II_undis, F_m_undis)
        )


with open(Data_Dir+ File_Name + "FK_Undistorted_model_tech1_pre_damage", "wb") as f:
    pickle.dump(FK_Undistorted_model_tech1_pre_damage, f)

print(FK_Undistorted_model_tech1_pre_damage['dvdL'].shape)
