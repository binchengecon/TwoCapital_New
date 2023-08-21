import numpy as np
import pandas as pd
import sys
print(sys.path)

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
parser.add_argument("--plot_year_gamma",type=int)
parser.add_argument("--plot_year_theta",type=int)

parser.add_argument("--scheme",type=str)
parser.add_argument("--HJB_solution",type=str)

# parser.add_argument("--Update",type=int)


args = parser.parse_args()

dataname = args.dataname

# Update = args.Update
IntPeriod = args.IntPeriod
timespan = 1/12
plot_year_gamma=args.plot_year_gamma
plot_year_theta=args.plot_year_theta

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
phi_0 = args.phi_0

# if len(xicarr)==4:
#     labellist = ['Capital Aversion', 'Climate Aversion', 'Technology Aversion', 'Damage Aversion']
#     Filename = 'Uncertainty Channels'
#     colors = ['blue','red', 'green', 'cyan', 'purple']
    
# if len(xicarr)==4 and min(xikarr)==0.050 and phi_0==0.5:
#     labellist = ['Climate Uncertainty', 'Damage Uncertainty', 'Productivity Uncertainty', ' No Technology Uncertainty']
#     Filename = 'Uncertainty Channels Less High Cost'
#     colors = ['blue','red', 'green', 'cyan', 'purple']
        
    
# if len(xicarr)==4 and min(xikarr)==0.025 and phi_0==0.5:
#     labellist = ['Climate Uncertainty', 'Damage Uncertainty', 'Productivity Uncertainty', 'No Technology Uncertainty']
#     Filename = 'Uncertainty Channels More High Cost'
#     colors = ['blue','red', 'green', 'cyan', 'purple']
        
        
        
# if len(xicarr)==4 and min(xikarr)==0.050:
#     labellist = ['Climate Uncertainty', 'Damage Uncertainty', 'Productivity Uncertainty', 'Technology Uncertainty']
#     Filename = 'Uncertainty Channels'
#     colors = ['blue','red', 'green', 'cyan', 'purple']
    

# if len(xicarr)==4 and min(xikarr)==0.075:
#     labellist = ['Climate Uncertainty', 'Damage Uncertainty', 'Productivity Uncertainty', 'Technology Uncertainty']
#     Filename = 'Uncertainty Channels More'
#     colors = ['blue','red', 'green', 'cyan', 'purple']
    
if len(xicarr)==4 and min(xikarr)==0.150:
    labellist = ['Climate Uncertainty', 'Damage Uncertainty', 'Productivity Uncertainty', 'Technology Uncertainty']
    Filename = 'Uncertainty Channels Less'
    colors = ['blue','red', 'green', 'cyan', 'purple']
    
# # if len(xicarr)==5:
# #     labellist = ['Capital Aversion', 'Climate Aversion', 'Technology Aversion', 'Damage Aversion', 'Full Aversion']
# #     Filename = 'Uncertainty Channels'
# #     # if rho==0.66:
# #     #     Filename = 'Uncertainty Channels_Rho<1'
# #     # if rho==1:
# #     #     Filename = 'Uncertainty Channels_Rho=1'    
# #     # if rho==1.5:
# #     #     Filename = 'Uncertainty Channels_Rho>1'

# #     colors =['blue','red', 'green', 'cyan', 'purple']
    
if len(xicarr)==3:
    labellist = ['More Aversion', 'Less Aversion', 'Neutrality']
    Filename = 'Aversion Intensity'
    # Filename = 'Aversion Intensity_old'
    # Filename = 'Aversion Intensity_onlyj'
    # Filename = 'Aversion Intensity_onlyk'
    colors = ['blue','red', 'green', 'cyan', 'purple']
    colors2 = ['blue','red', 'green', 'cyan', 'purple']

    
# # if len(xicarr)==3 and min(xikarr)==0.075:
# #     labellist = ['Even Less Aversion', 'Much Less Aversion', 'Neutrality']
# #     Filename = 'Aversion Intensity'
# #     # Filename = 'Aversion Intensity_old'
# #     # Filename = 'Aversion Intensity_onlyj'
# #     # Filename = 'Aversion Intensity_onlyk'
# #     colors = ['blue','red', 'green', 'cyan', 'purple']
# #     colors2 = ['blue','red', 'green', 'cyan', 'purple']

# # if len(xicarr)==3 and min(xikarr)==0.150:
# #     labellist = ['Very Less Aversion', 'Very Very Less Aversion', 'Neutrality']
# #     Filename = 'Aversion Intensity'
# #     # Filename = 'Aversion Intensity_old'
# #     # Filename = 'Aversion Intensity_onlyj'
# #     # Filename = 'Aversion Intensity_onlyk'
# #     colors = ['blue','red', 'green', 'cyan', 'purple']
# #     colors2 = ['blue','red', 'green', 'cyan', 'purple']

# if len(xicarr)==5:
#     labellist = ['Climate Uncertainty', 'Damage Uncertainty', 'Productivity Uncertainty', 'Technology Uncertainty', 'No Technology Uncertainty']
#     Filename = 'Uncertainty Channel5'
#     # Filename = 'Aversion Intensity_old'
#     # Filename = 'Aversion Intensity_onlyj'
#     # Filename = 'Aversion Intensity_onlyk'
#     colors = ['blue','red', 'green', 'cyan', 'purple']
#     colors2 = ['blue','red', 'green', 'cyan', 'purple']

# if len(xicarr)==1:
#     labellist = ['No Technology Uncertainty']
#     Filename = 'Uncertainty Channel1'
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

def model_simulation_generate(xi_a,xi_k,xi_c,xi_j,xi_d,xi_g,psi_0,psi_1,varrho,rho):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    # File_Dir = "xi_a_{}_xi_k_{}_xi_c_{}_xi_j_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_rho_{}_" .format(xi_a,xi_k,xi_c,xi_j,xi_d,xi_g,psi_0,psi_1,varrho,rho)
    File_Dir = "xi_a_{}_xi_k_{}_xi_c_{}_xi_j_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_rho_{}_delta_{}_" .format(xi_a, xi_k, xi_c, xi_j, xi_d, xi_g, psi_0,psi_1, varrho, rho, delta)
    

    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_UD_simul_{}".format(IntPeriod)+ scheme + "_" +HJB_solution, "rb") as f:
        res = pickle.load(f)


    
    return res

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                plt.xlabel('Years')
                plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                if vartheta_bar==0.1:
                    plt.ylim(0,4)
                if vartheta_bar==0.5:
                    plt.ylim(0,15)
                plt.xlim(0,30)

                plt.legend(loc='upper left')        
print(res.keys())
plt.savefig(Plot_Dir+"/RD_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/RD_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["x"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["x"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                plt.xlabel('Years')
                # plt.ylabel('')
                plt.title("Raw R&D investment")
                # if auto==0:   
                # plt.ylim(0,1)
                plt.xlim(0,30)

                plt.legend(loc='upper left')        
print(res.keys())
plt.savefig(Plot_Dir+"/RD_Raw"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/RD_Raw"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)*(1-res["true_tech_prob"]))[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)*(1-res["true_tech_prob"]))[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                plt.xlabel('Years')
                plt.ylabel('$\%$ of GDP')
                plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                plt.ylim(0,1)
                plt.xlim(0,30)

                plt.legend(loc='upper left')        
print(res.keys())
plt.savefig(Plot_Dir+"/RD_Expected_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/RD_Expected_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["i"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["i"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                plt.xlabel('Years')
                # plt.title("Capital investment")
                # if auto==0:   
                plt.ylim(30,160)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/CapI_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/CapI_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], (res["i"]/(alpha*np.exp(res["states"][:,0]))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (res["i"]/(alpha*np.exp(res["states"][:,0]))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                plt.xlabel('Years')
                # plt.title("Capital investment as percentage of GDP")
                # if auto==0:   
                plt.ylim(50,80)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/CapI_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/CapIRatio_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["e"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["e"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                # plt.title("Carbon Emissions")
                # if auto==0:   
                plt.ylim(6.0,15.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/E_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/E_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], ( res["e"] * (1-res["true_tech_prob"]) )[res["states"][:, 1]<1.5] ,label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], ( res["e"] * (1-res["true_tech_prob"]) )[res["states"][:, 1]<1.5] ,label=labellist[id_xiag],linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Carbon Emissions")
                # if auto==0:   
                plt.ylim(6.0,12.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/ETrue_Expected_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/ETrue_Expected_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["states"][:, 1][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["states"][:, 1][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Temperature Anomaly")
                # if auto==0:   
                plt.ylim(1.1,1.5)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TA_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/TA_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.exp(res["states"][:, 2])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.exp(res["states"][:, 2])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                # plt.title("Knowledge Stock $J_g$")
                # if auto==0:   
                plt.ylim(11.0,45.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')


plt.savefig(Plot_Dir+"/Ig_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/Ig_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["distorted_tech_prob"],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"], res["distorted_tech_prob"],label=labellist[id_xiag],linewidth=5.0)
                # plt.plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_tech_prob"],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                # plt.title("Distorted Probability of a Technology Jump")
                plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/ProbTechJump_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/ProbTechJump_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["distorted_damage_prob"],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"], res["distorted_damage_prob"],label=labellist[id_xiag],linewidth=5.0)
                # plt.plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_damage_prob"],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                # plt.title("Distorted Probability of Damage Changes")
                plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/ProbDamageChange_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/ProbDamageChange_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["true_tech_prob"],label=labellist[id_xiag],linewidth=5.0,linestyle = 'dashed',color=color_one)
                else:
                    plt.plot(res["years"], res["true_tech_prob"],label=labellist[id_xiag] ,linewidth=5.0,linestyle = 'dashed',color=color_one)
                    
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["distorted_tech_prob"],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"], res["distorted_tech_prob"],label=labellist[id_xiag] ,linewidth=5.0,color=color_one)
                # plt.plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_tech_prob"],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Distorted(Solid) and True(Dashed) Probability of a Technology Jump")
                plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/CombProbTechJump_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/CombProbTechJump_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["true_damage_prob"],label=labellist[id_xiag],linewidth=5.0,linestyle = 'dashed',color=color_one)
                else:
                    plt.plot(res["years"], res["true_damage_prob"],label=labellist[id_xiag] ,linewidth=5.0,linestyle = 'dashed',color=color_one)
                    
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["distorted_damage_prob"],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"], res["distorted_damage_prob"],label=labellist[id_xiag] ,linewidth=5.0,color=color_one)
                # plt.plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_tech_prob"],label=labellist[id_xiag],linewidth=7.0)
                # plt.plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_damage_prob"],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Distorted Probability of Damage Changes")
                plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/CombProbDamageChange_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/CombProbDamageChange_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["true_tech_prob"],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"], res["true_tech_prob"],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.title("True Probability of a Technology Jump")
                plt.ylim(0.0,1.0)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TPIg_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/TPIg_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["true_damage_prob"],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"], res["true_damage_prob"],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.title("True Probability of Damage Changes")
                plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TPId_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/TPId_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.title("Log of Social Cost of Carbon")
                # if auto==0:   
                plt.ylim(4.0,8.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSCC_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/logSCC_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                # plt.title("Log of Social Value of R&D")
                # if auto==0:   
                plt.ylim(3.0,10.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSVRD_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/logSVRD_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scgw"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scgw"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                # plt.title("Log of Social Cost of Global Warming")
                # if auto==0:   
                plt.ylim(11.0,14.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSCGW_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/logSCGW_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scgw2"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scgw2"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                # plt.title("Log of Social Cost of Global Warming")
                # if auto==0:   
                plt.ylim(11.0,14.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSCGW2_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/logSCGW2_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_2"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_2"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("Log of Term (ii)")
                # if auto==0:   
                # plt.ylim(4.0,8.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSVRD2_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/logSVRD2_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["spo"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["spo"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("Log of Term (iii)")
                # if auto==0:   
                # plt.ylim(4.0,8.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/spo_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/spo_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["spo2"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["spo2"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("Log of Social Payoff")
                # if auto==0:   
                # plt.ylim(4.0,8.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/spo2_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/spo2_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["uncertainty_adjusted_diff"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["uncertainty_adjusted_diff"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("Uncertainty Adjusted Difference")
                # if auto==0:   
                plt.ylim(-0.1, 0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/UAD_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/UAD_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["uncertainty_adjusted_diff2"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=1.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["uncertainty_adjusted_diff2"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=1.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("Uncertainty Adjusted Difference")
                # if auto==0:   
                # plt.ylim(-0.1, 0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/UAD2_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/UAD2_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["v_post_techt"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5],res["v_post_techt"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("$V_g$")
                # if auto==0:   
                plt.ylim(4.0,5.3)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/Vgt_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/Vgt_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["vt"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["vt"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("$V$")
                # if auto==0:   
                plt.ylim(4.0,5.3)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/Vt_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/Vt_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], (res["v_post_techt"]-res["vt"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (res["v_post_techt"]-res["vt"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("$V_g-V$")
                # if auto==0:   
                plt.ylim(0.0,1)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/Vg-Vt_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/Vg-Vt_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], (res["gt_tech"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (res["gt_tech"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("$g_t$")
                # if auto==0:   
                # plt.ylim(0.0,1)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/gt_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".pdf")
plt.savefig(Plot_Dir+"/gt_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png")
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], ( sigma_y * res["e"] * res["ht"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], ( sigma_y * res["e"] * res["ht"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                plt.title(r"$\sigma_y e h$")
                plt.ylim(0,0.005)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/h_{}_"+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/h_{}_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], -(sigma_k * res["hkt"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], -(sigma_k * res["hkt"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                plt.title(r"-$\sigma_k h_k$")
                plt.ylim(0,0.005)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/hk_{}_"+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/hk_{}_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], -(sigma_g * res["hjt"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], -(sigma_g * res["hjt"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                plt.title(r"-$\sigma_j h_j$")
                plt.ylim(0,0.005)

                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/hj_{}_"+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/hj_{}_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["ht"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["ht"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                plt.title(r"$ h$")
                plt.ylim(0,0.45)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/h2_{}_"+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/h2_{}_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], -( res["hkt"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], -( res["hkt"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                plt.title(r"-$ h_k$")
                plt.ylim(0,0.45)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/hk2_{}_"+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/hk2_{}_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], -( res["hjt"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], -( res["hjt"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                plt.title(r"-$h_j$")
                plt.ylim(0,0.45)

                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/hj2_{}_"+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/hj2_{}_"+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["RelativeEntropy_hk"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["RelativeEntropy_hk"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                # plt.title("Relative Entropy for Capital")
                # plt.ylim(0,0.45)
                plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/h2_{}_".format(IntPeriod)+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/RE_K_{}_".format(IntPeriod)+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["RelativeEntropy_h"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["RelativeEntropy_h"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                # plt.title(" Relative Entropy for Temperature Anomaly")
                # plt.ylim(0,0.45)
                plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/h2_{}_".format(IntPeriod)+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/RE_TA_{}_".format(IntPeriod)+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["RelativeEntropy_hj"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["RelativeEntropy_hj"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                # plt.title("Relative Entropy for Knowledge Capital")
                plt.ticklabel_format(style='plain')
                # plt.ylim(0,0.45)
                plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/h2_{}_".format(IntPeriod)+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/RE_J_{}_".format(IntPeriod)+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["RelativeEntropy_TechJump"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["RelativeEntropy_TechJump"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                plt.title(r"$ h$")
                # plt.ylim(0,0.45)
                plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/h2_{}_".format(IntPeriod)+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/RE_TechJump_{}_".format(IntPeriod)+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"], (  res["RelativeEntropy_TechJump"]),label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"], (  res["RelativeEntropy_TechJump"]),label=labellist[id_xiag]  )

                plt.xlabel("Years")
                # plt.title("Relative Entropy for Technology Jump")
                # plt.ylim(0,0.45)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/h2_{}_".format(IntPeriod)+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/RE_TechJump2_{}_".format(IntPeriod)+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["RelativeEntropy_DamageJump"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (  res["RelativeEntropy_DamageJump"])[res["states"][:, 1]<1.5],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                plt.title(r"$ h$")
                # plt.ylim(0,0.45)
                plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/h2_{}_".format(IntPeriod)+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/RE_DamageJump_{}_".format(IntPeriod)+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)
                
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"], res["RelativeEntropy_DamageJump"],label=labellist[id_xiag]  )
                else:
                    plt.plot(res["years"], res["RelativeEntropy_DamageJump"],label=labellist[id_xiag]  )

                plt.xlabel("Years")
                # plt.title("Relative Entropy for Damage Jump")
                # plt.ylim(0,0.45)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/h2_{}_".format(IntPeriod)+Filename+".pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.savefig(Plot_Dir+"/RE_DamageJump2_{}_".format(IntPeriod)+Filename+"_rho={}_delta={}_phi0={}".format(rho,delta,phi_0)+".png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
plt.close()

plt.style.use('default')
plt.rcParams["lines.linewidth"] = 20
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.figsize"] = (16,10)
plt.rcParams["font.size"] = 25
plt.rcParams["legend.frameon"] = False


print("After, figure default size is: ", plt.rcParams["savefig.bbox"])
print("After, figure default size is: ", plt.rcParams["figure.figsize"])
print("After, figure default dpi is: ", plt.rcParams["figure.dpi"])
print("After, figure default size is: ", plt.rcParams["font.size"])
print("After, legend.frameon is: ", plt.rcParams["legend.frameon"])
print("After, lines.linewidth is: ", plt.rcParams["lines.linewidth"])


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                
                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                NUM_DAMAGE = res["gt_dmg"].shape[0]
                gamma_3_list = np.linspace(0., 1./3., NUM_DAMAGE)

                # γ3_distort = np.load("γ3_5.npy")
                print(NUM_DAMAGE)
                
                time_frame = int(plot_year_gamma/timespan)
                γ3_distort = res["gt_dmg"][:, time_frame] 
                # plt.figure(figsize=(16,10))
                plt.hist(gamma_3_list, weights=np.ones(len(gamma_3_list)) / len(gamma_3_list), 
                        alpha=0.5, color="C3", ec="darkgray",label='Baseline' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]), bins=NUM_DAMAGE)
                plt.hist(gamma_3_list, weights= γ3_distort / np.sum(γ3_distort), 
                        alpha=0.5, color="C0", ec="darkgray",label=labellist[id_xiag], bins=NUM_DAMAGE)
                if vartheta_bar==0.1:

                    plt.ylim(0, 0.15)
                if vartheta_bar==0.5:
                    plt.ylim(0, 0.15)
                # plt.title("Distorted Probability of Damage Models")
                # plt.xlabel("Damage Curvature")
                plt.legend(loc='upper left',frameon=False)

                    
                # plt.savefig(Plot_Dir+"/Gamma3_{},xia={:.5f},xik={:.3f},xic={:.3f},xij={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},varrho={:.1f}_rho={}_delta={}_phi0={}.png".format(plot_year_gamma,xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho,delta,phi_0))
                plt.savefig(Plot_Dir+"/Gamma3_{},".format(plot_year_gamma)+Filename+labellist[id_xiag] + "_rho={}_delta={}_phi0={}.png".format(rho,delta,phi_0))
                plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

        
                res = model_simulation_generate(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho)

                time_frame = int(plot_year_theta/timespan)

                # histogram of beta_f
                theta_ell = pd.read_csv("./data/model144.csv", header=None).to_numpy()[:, 0]
                # print("theta_ell")
                # print(theta_ell)
                # print("theta_ell_new")
                # print(theta_ell_new)
                pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
                # pi_c = np.load("πc_5.npy")
                theta_ell_new = res["theta_ell_new"][:,time_frame]

                pi_c = res["pic_t"][:, time_frame]
                # plt.figure(figsize=(16,10))

                # plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                #         alpha=0.5, ec="darkgrey", color="C3",label='Baseline')
                # plt.hist(theta_ell, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                #         alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_r={:.3f}$'.format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag])  )
                # plt.legend(loc='upper left')
                # plt.title("Distorted probability of Climate Models")

                # plt.ylim(0, 1.4)
                # plt.xlabel("Climate Sensitivity")
                # plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ClimateSensitivity_25,xia={:.4f},xic={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},BC.pdf".format(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
                # plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ClimateSensitivity_25,xia={:.4f},xic={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},BC.png".format(xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
                # plt.close()

                plt.figure(figsize=(16,10))

                plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                        alpha=0.5, ec="darkgrey", color="C3",label='Baseline')
                plt.hist(theta_ell_new*1000, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                        alpha=0.5, ec="darkgrey", color="C0",label=labellist[id_xiag]  )
                plt.legend(loc='upper left')
                # plt.title("Distorted Probability of Climate Models")


                print("mean of uncondition = {}" .format(np.average(theta_ell,weights = pi_c_o)))
                print("mean of condition = {}" .format(np.average(theta_ell_new*1000,weights = pi_c)))
                    

                plt.ylim(0, 1.4)
                # plt.xlabel("Climate Sensitivity")
                # plt.savefig(Plot_Dir+"/ClimateSensitivity_pmean_{},xia={:.5f},xik={:.3f},xic={:.3f},xij={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},varrho={:.1f}_rho={}_delta={}_phi0={}.pdf".format(plot_year_theta, xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho,delta,phi_0))
                # plt.savefig(Plot_Dir+"/ClimateSensitivity_pmean_{},xia={:.5f},xik={:.3f},xic={:.3f},xij={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},varrho={:.1f}_rho={}_delta={}_phi0={}.png".format(plot_year_theta, xiaarr[id_xiag],xikarr[id_xiag],xicarr[id_xiag],xijarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho],rho,delta,phi_0))
                plt.savefig(Plot_Dir+"/ClimateSensitivity_pmean_{},".format(plot_year_theta) +Filename+labellist[id_xiag]+"_rho={}_delta={}_phi0={}.png".format(rho,delta,phi_0))
                plt.close()
