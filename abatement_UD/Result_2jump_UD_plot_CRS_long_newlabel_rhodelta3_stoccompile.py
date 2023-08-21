import numpy as np
import pandas as pd
import sys
print(sys.path)
import copy

sys.path.append('./src')

import pickle
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
# from src.supportfunctions import finiteDiff_3D
import os
import argparse
import time
import petsc4py
from petsc4py import PETSc
import petsclinearsystem
# from Result_support import *
sys.stdout.flush()


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--dataname",type=str)
parser.add_argument("--pdfname",type=str)

parser.add_argument("--xiaarr",nargs='+', type=float)
parser.add_argument("--xikarr",nargs='+', type=float)
parser.add_argument("--xicarr",nargs='+', type=float)
parser.add_argument("--xijarr",nargs='+', type=float)
parser.add_argument("--xidarr",nargs='+', type=float)
parser.add_argument("--xigarr",nargs='+', type=float)

parser.add_argument("--varrhoarr",nargs='+', type=float)
parser.add_argument("--phi_0", type=float)
parser.add_argument("--rhoarr", type=float)
parser.add_argument("--delta", type=float)

parser.add_argument("--psi0arr",nargs='+',type=float)
parser.add_argument("--psi1arr",nargs='+',type=float)
# parser.add_argument("--psi2arr",nargs='+',type=float)
parser.add_argument("--num_gamma",type=int)

parser.add_argument("--hXarr",nargs='+',type=float)
parser.add_argument("--Xminarr",nargs='+',type=float)
parser.add_argument("--Xmaxarr",nargs='+',type=float)

parser.add_argument("--auto",type=int)
parser.add_argument("--IntPeriod",type=int)

parser.add_argument("--scheme",type=str)
parser.add_argument("--HJB_solution",type=str)


parser.add_argument("--SimPathNum",type=int)

# parser.add_argument("--Update",type=int)


args = parser.parse_args()

dataname = args.dataname

# Update = args.Update
IntPeriod = args.IntPeriod
timespan = 1/12

psi0arr = args.psi0arr
psi1arr = args.psi1arr
# psi2arr = args.psi2arr
xiaarr = args.xiaarr
xikarr = args.xikarr 
xicarr = args.xicarr 
xijarr = args.xijarr 
xidarr = args.xidarr 
xigarr = args.xigarr 
varrhoarr = args.varrhoarr
rho = args.rhoarr

# if len(xicarr)==4:
#     labellist = ['Capital Aversion', 'Climate Aversion', 'Technology Aversion', 'Damage Aversion']
#     Filename = 'Uncertainty Channels'
#     colors = ['blue','red', 'green', 'cyan', 'purple']
    
if len(xicarr)==4 and min(xikarr)==0.050:
    labellist = ['Climate Uncertainty', 'Damage Uncertainty', 'Productivity Uncertainty', 'Technology Uncertainty']
    Filename = 'Uncertainty Channels'
    colors = ['blue','red', 'green', 'cyan', 'purple']
    

if len(xicarr)==4 and min(xikarr)==0.075:
    labellist = ['Climate Uncertainty', 'Damage Uncertainty', 'Productivity Uncertainty', 'Technology Uncertainty']
    Filename = 'Uncertainty Channels More'
    colors = ['blue','red', 'green', 'cyan', 'purple']
    
if len(xicarr)==4 and min(xikarr)==0.150:
    labellist = ['Climate Uncertainty', 'Damage Uncertainty', 'Productivity Uncertainty', 'Technology Uncertainty']
    Filename = 'Uncertainty Channels Less'
    colors = ['blue','red', 'green', 'cyan', 'purple']
    
if len(xicarr)==5:
    labellist = ['Capital Aversion', 'Climate Aversion', 'Technology Aversion', 'Damage Aversion', 'Full Aversion']
    Filename = 'Uncertainty Channels'
    # if rho==0.66:
    #     Filename = 'Uncertainty Channels_Rho<1'
    # if rho==1:
    #     Filename = 'Uncertainty Channels_Rho=1'    
    # if rho==1.5:
    #     Filename = 'Uncertainty Channels_Rho>1'

    colors =['blue','red', 'green', 'cyan', 'purple']
    
if len(xicarr)==3:
    labellist = ['More Aversion', 'Less Aversion', 'Neutrality']
    Filename = 'Aversion Intensity'
    # Filename = 'Aversion Intensity_old'
    # Filename = 'Aversion Intensity_onlyj'
    # Filename = 'Aversion Intensity_onlyk'
    colors = ['blue','red', 'green', 'cyan', 'purple']
    colors2 = ['blue','red', 'green', 'cyan', 'purple']

    
# if len(xicarr)==3 and min(xikarr)==0.075:
#     labellist = ['Even Less Aversion', 'Much Less Aversion', 'Neutrality']
#     Filename = 'Aversion Intensity'
#     # Filename = 'Aversion Intensity_old'
#     # Filename = 'Aversion Intensity_onlyj'
#     # Filename = 'Aversion Intensity_onlyk'
#     colors = ['blue','red', 'green', 'cyan', 'purple']
#     colors2 = ['blue','red', 'green', 'cyan', 'purple']

# if len(xicarr)==3 and min(xikarr)==0.150:
#     labellist = ['Very Less Aversion', 'Very Very Less Aversion', 'Neutrality']
#     Filename = 'Aversion Intensity'
#     # Filename = 'Aversion Intensity_old'
#     # Filename = 'Aversion Intensity_onlyj'
#     # Filename = 'Aversion Intensity_onlyk'
#     colors = ['blue','red', 'green', 'cyan', 'purple']
#     colors2 = ['blue','red', 'green', 'cyan', 'purple']

    
    
# colors = ['blue','green', 'red', 'cyan']

Xminarr = args.Xminarr
Xmaxarr = args.Xmaxarr
hXarr = args.hXarr
auto = args.auto

num_gamma = args.num_gamma
gamma_3_list = np.linspace(0,1./3.,num_gamma)

scheme = args.scheme
HJB_solution = args.HJB_solution


delta = args.delta
alpha = 0.115
kappa = 6.667
mu_k  = -0.043
sigma_k = 0.01
beta_f = 1.86/1000
sigma_y = 1.2 * 1.86 / 1000
zeta = 0.0
# psi_0 = 0.00025
# psi_1 = 1/2
sigma_g = 0.0078
gamma_1 = 1.7675 / 1000
gamma_2 = 0.0022 * 2


y_bar = 2.
y_bar_lower = 1.5

# Tech
theta = 3
lambda_bar = 0.1206
# vartheta_bar = 0.0453
# vartheta_bar = 0.05
# vartheta_bar = 0.056
# vartheta_bar = 0.5
vartheta_bar = args.phi_0

lambda_bar_first = lambda_bar / 2.
vartheta_bar_first = vartheta_bar / 2.

lambda_bar_second = 1e-3
vartheta_bar_second = 0.

SimPathNum = args.SimPathNum

# print(plt.rcParamsDefault)
# print("Before, figure default size is: ", plt.rcParams["figure.figsize"])
# print("Before, figure default dpi is: ", plt.rcParams["figure.dpi"])
# print("Before, figure default size is: ", plt.rcParams["font.size"])
# print("Before, legend.frameon is: ", plt.rcParams["legend.frameon"])
# print("Before, lines.linewidth is: ", plt.rcParams["lines.linewidth"])

plt.style.use('classic')
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 12
plt.rcParams["legend.frameon"] = False
plt.rcParams["lines.linewidth"] = 5
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['blue','red', 'green', 'cyan', 'purple']) 
    # colors = ['blue','green', 'red', 'cyan', 'purple']

print("After, figure default size is: ", plt.rcParams["savefig.bbox"])
print("After, figure default size is: ", plt.rcParams["figure.figsize"])
print("After, figure default dpi is: ", plt.rcParams["figure.dpi"])
print("After, figure default size is: ", plt.rcParams["font.size"])
print("After, legend.frameon is: ", plt.rcParams["legend.frameon"])
print("After, lines.linewidth is: ", plt.rcParams["lines.linewidth"])


os.makedirs("./abatement_UD/pdf_2tech/"+args.dataname+"/"+scheme+"_"+HJB_solution+"/", exist_ok=True)

Plot_Dir = "./abatement_UD/pdf_2tech/"+args.dataname+"/"+scheme+"_"+HJB_solution+"/"

def model_simulation_compile(xi_a,xi_k,xi_c,xi_j,xi_d,xi_g,psi_0,psi_1,varrho,rho):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    # File_Dir = "xi_a_{}_xi_k_{}_xi_c_{}_xi_j_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_rho_{}_" .format(xi_a,xi_k,xi_c,xi_j,xi_d,xi_g,psi_0,psi_1,varrho,rho)
    File_Dir = "xi_a_{}_xi_k_{}_xi_c_{}_xi_j_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_rho_{}_delta_{}_" .format(xi_a, xi_k, xi_c, xi_j, xi_d, xi_g, psi_0,psi_1, varrho, rho, delta)
    
    i = 0

    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_UD_simulstoc2_{}_path_{}".format(IntPeriod, i)+ scheme + "_" +HJB_solution, "rb") as f:
        res = pickle.load(f)



    keys_list = res.keys()

    res_final = copy.deepcopy(res)

    for item in keys_list:

        print("item ={}. original res_final item shape={}".format(item , res_final[item].shape))

        # if item != "years":
        #     res_final[item] = np.expand_dims(res_final[item],axis=-1)
        #     for i in range(SimPathNum):


        #         with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_UD_simulstoc_{}_path_{}".format(IntPeriod, i)+ scheme + "_" +HJB_solution, "rb") as f:
        #             res = pickle.load(f)

        #         res[item] = np.expand_dims(res[item],axis=-1)

        #         if i>0:
        #             res_final[item] = np.append(res_final[item], res[item],axis=-1)

        # else:
        #     res_final[item] = np.expand_dims(res_final[item],axis=-1)
            
            
        res_final[item] = np.expand_dims(res_final[item],axis=-1)
        
        for i in range(SimPathNum):


            with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_UD_simulstoc2_{}_path_{}".format(IntPeriod, i)+ scheme + "_" +HJB_solution, "rb") as f:
                res = pickle.load(f)

            res[item] = np.expand_dims(res[item],axis=-1)

            if i>0:
                res_final[item] = np.append(res_final[item], res[item],axis=-1)
        
                
        print("item ={}, final res_final item shape={}".format(item, res_final[item].shape))
        
    print(keys_list)
    # print(res_final["RD_Plot"])
            
    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_UD_simulstoccompile2_{}".format(IntPeriod)+ scheme + "_" +HJB_solution, "wb") as f:
        pickle.dump(res_final,f)

    return res




for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                model_simulation_compile(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                