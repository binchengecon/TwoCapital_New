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
parser.add_argument("--SimPathNum",type=int)

parser.add_argument("--xiaarr",nargs='+', type=float)
parser.add_argument("--xicarr",nargs='+', type=float)
parser.add_argument("--xidarr",nargs='+', type=float)
parser.add_argument("--xigarr",nargs='+', type=float)

parser.add_argument("--varrhoarr",nargs='+', type=float)


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

# parser.add_argument("--Update",type=int)


args = parser.parse_args()

dataname = args.dataname

# Update = args.Update
IntPeriod = args.IntPeriod
timespan = 1/12
SimPathNum=args.SimPathNum

psi0arr = args.psi0arr
psi1arr = args.psi1arr
# psi2arr = args.psi2arr
xiaarr = args.xiaarr
xicarr = args.xicarr 
xidarr = args.xidarr 
xigarr = args.xigarr 
varrhoarr = args.varrhoarr

if min(xicarr)==0.050:
    labellist = ['Climate Aversion', 'Damage Aversion', 'Technology Aversion']
    Filename = 'Uncertainty Channels'
    colors = ['blue','green', 'red', 'cyan']

elif min(xicarr)==0.025:
    labellist = ['More Aversion', 'Less Aversion', 'Neutrality']
    Filename = 'Aversion VS Neutrality'
    colors = ['blue', 'red', 'green', 'cyan']




Xminarr = args.Xminarr
Xmaxarr = args.Xmaxarr
hXarr = args.hXarr
auto = args.auto

num_gamma = args.num_gamma
gamma_3_list = np.linspace(0,1./3.,num_gamma)

scheme = args.scheme
HJB_solution = args.HJB_solution


delta = 0.01
alpha = 0.115
kappa = 6.667
mu_k  = -0.043
sigma_k = 0.0095
beta_f = 1.86/1000
sigma_y = 1.2 * 1.86 / 1000
zeta = 0.0
# psi_0 = 0.00025
# psi_1 = 1/2
sigma_g = 0.016
gamma_1 = 1.7675 / 1000
gamma_2 = 0.0022 * 2


y_bar = 2.
y_bar_lower = 1.5

# Tech
theta = 3
lambda_bar = 0.1206
vartheta_bar = 0.0453

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
plt.rcParams["font.size"] = 18
plt.rcParams["legend.frameon"] = False
plt.rcParams["lines.linewidth"] = 5

print("After, figure default size is: ", plt.rcParams["savefig.bbox"])
print("After, figure default size is: ", plt.rcParams["figure.figsize"])
print("After, figure default dpi is: ", plt.rcParams["figure.dpi"])
print("After, figure default size is: ", plt.rcParams["font.size"])
print("After, legend.frameon is: ", plt.rcParams["legend.frameon"])
print("After, lines.linewidth is: ", plt.rcParams["lines.linewidth"])


os.makedirs("./abatement_UD/pdf_2tech/"+args.dataname+"/"+scheme+"_"+HJB_solution+"/", exist_ok=True)

Plot_Dir = "./abatement_UD/pdf_2tech/"+args.dataname+"/"+scheme+"_"+HJB_solution+"/"

def model_simulation_total(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_c_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_" .format(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho)

    res_total = []
    for sim in range(SimPathNum):
        
        with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_path_{}".format(IntPeriod, sim)+ scheme + "_" +HJB_solution, "rb") as f:
            res = pickle.load(f)

        res_total.append(res)
    
    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_pathtotal".format(IntPeriod)+ scheme + "_" +HJB_solution, "wb") as f:
        pickle.dump(res,f)
        
    return res_total



def model_simulation_var(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho, varname):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_c_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_" .format(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho)


    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_pathtotal".format(IntPeriod)+ scheme + "_" +HJB_solution, "rb") as f:
        res_total = pickle.load(f)


    var_matrix = np.zeros((SimPathNum, IntPeriod/timespan+1))
    var_matrix90 = np.zeros((1, IntPeriod/timespan+1))
    var_matrix50 = np.zeros((1, IntPeriod/timespan+1))
    var_matrix10 = np.zeros((1, IntPeriod/timespan+1))
    
    for sim in range(SimPathNum):

        if varname == "x":
            var_matrix[sim,:] =  (res_total[sim]["x"]/(alpha*np.exp( res_total[sim]["states"][:,0])))*100
        else :
            var_matrix[sim,:] = res_total[sim][varname] 
            
    var_matrix90 = np.quantile(var_matrix,0.9,axis=0)
    var_matrix50 = np.quantile(var_matrix,0.5,axis=0)
    var_matrix10 = np.quantile(var_matrix,0.1,axis=0)

    var_year = res_total[sim]["years"]
    
    return var_matrix90, var_matrix50, var_matrix10, var_year



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                res = model_simulation_total(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "x")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                axes.set_ylabel("$\%$ of GDP")
    
    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,1)
                axes.set_xlim(0,30)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/RD_"+Filename+labellist[id_xiag]+".png")
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)*(1-res["true_tech_prob"]))[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)*(1-res["true_tech_prob"]))[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                plt.xlabel('Years')
                plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                plt.ylim(0,1)
                plt.xlim(0,30)

                plt.legend(loc='upper left')        
print(res.keys())
plt.savefig(Plot_Dir+"/RD_Expected_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/RD_Expected_"+Filename+".png")
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):
                
                color_one = colors[id_xiag % len(xiaarr)]   


                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["i"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["i"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                plt.xlabel('Years')
                # plt.title("Capital investment")
                # if auto==0:   
                plt.ylim(65,110)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/CapI_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/CapI_"+Filename+".png")
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["e"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["e"][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                # plt.title("Carbon Emissions")
                # if auto==0:   
                plt.ylim(6.0,12.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/E_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/E_"+Filename+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], ( res["e"] * (1-res["true_tech_prob"]) )[res["states"][:, 1]<1.5] ,label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], ( res["e"] * (1-res["true_tech_prob"]) )[res["states"][:, 1]<1.5] ,label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                # plt.title("Carbon Emissions")
                # if auto==0:   
                plt.ylim(6.0,12.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/ETrue_Expected_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/ETrue_Expected_"+Filename+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["states"][:, 1][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["states"][:, 1][res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                # plt.title("Temperature Anomaly")
                # if auto==0:   
                plt.ylim(1.1,1.5)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TA_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/TA_"+Filename+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.exp(res["states"][:, 2])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.exp(res["states"][:, 2])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                # plt.title("Knowledge Stock $J_g$")
                # if auto==0:   
                plt.ylim(11.0,40.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')


plt.savefig(Plot_Dir+"/Ig_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/Ig_"+Filename+".png")
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["distorted_tech_prob"],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"], res["distorted_tech_prob"],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_tech_prob"],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                # plt.title("Distorted Probability of a Technology Jump")
                plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/ProbTechJump_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/ProbTechJump_"+Filename+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["distorted_damage_prob"],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"], res["distorted_damage_prob"],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_a=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_damage_prob"],label=labellist[id_xiag],linewidth=7.0)
                plt.xlabel('Years')
                # plt.title("Distorted Probability of Damage Changes")
                plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/ProbDamageChange_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/ProbDamageChange_"+Filename+".png")
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

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
                # plt.title("Distorted(Solid) and True(Dashed) Probability of a Technology Jump")
                plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/CombProbTechJump_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/CombProbTechJump_"+Filename+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):


                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

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
                # plt.title("Distorted Probability of Damage Changes")
                plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/CombProbDamageChange_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/CombProbDamageChange_"+Filename+".png")
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["true_tech_prob"],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"], res["true_tech_prob"],label=labellist[id_xiag],linewidth=5.0,color=color_one)

                plt.xlabel("Years")
                # plt.title("True Probability of a Technology Jump")
                plt.ylim(0.0,1.0)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TPIg_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/TPIg_"+Filename+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["true_damage_prob"],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"], res["true_damage_prob"],label=labellist[id_xiag],linewidth=5.0,color=color_one)

                plt.xlabel("Years")
                # plt.title("True Probability of Damage Changes")
                plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TPId_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/TPId_"+Filename+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)

                plt.xlabel("Years")
                # plt.title("Log of Social Cost of Carbon")
                # if auto==0:   
                plt.ylim(3.0,6.5)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSCC_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/logSCC_"+Filename+".png")
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scgw"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scgw"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)

                plt.xlabel("Years")
                # plt.title("Log of Social Cost of Carbon")
                # if auto==0:   
                plt.ylim(10.0,12.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSCGW_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/logSCGW_"+Filename+".png")
plt.close()



# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):


#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc_dis"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc_dis"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

#                 plt.xlabel("Years")
#                 # plt.title("Log of Social Cost of Carbon, dis")
#                 if auto==0:   
#                     plt.ylim(3.0,6.5)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSCC_dis_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSCC_dis_"+Filename+".png")
# plt.close()

# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):


#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc_undis"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc_undis"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

#                 plt.xlabel("Years")
#                 # plt.title("Log of Social Cost of Carbon, Undis")
#                 if auto==0:   
#                     plt.ylim(3.0,6.5)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSCC_undis_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSCC_undis_"+Filename+".png")
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

#                 color_one = colors[id_xiag % len(xiaarr)]

#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc_undis"])[res["states"][:, 1]<1.5],label='FK: Infinity',linewidth=5.0,linestyle = 'dashed',color=color_one)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc_undis"])[res["states"][:, 1]<1.5],label='FK: $\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]) ,linewidth=5.0,linestyle = 'dashed',color=color_one)
                
#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc"])[res["states"][:, 1]<1.5],label=labellist[id_xiag] ,linewidth=5.0,color=color_one)

#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Cost of Crabon: Comparison")
#                 if auto==0:   
#                     plt.ylim(6.5,8.0)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSCC_orig_undis_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSCC_orig_undis_"+Filename+".png")
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

#                 color_one = colors[id_xiag % len(xiaarr)]
                
#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scc_undis"]),label='FK: Infinity',linewidth=5.0,linestyle = 'dashed',color=color_one)
#                 else:
#                     plt.plot(res["years"], np.log(res["scc_undis"]),label='FK: $\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]) ,linewidth=5.0,linestyle = 'dashed',color=color_one)
                
#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scc_dis"]),label=labellist[id_xiag],linewidth=5.0,color=color_one)
#                 else:
#                     plt.plot(res["years"], np.log(res["scc_dis"]),label=labellist[id_xiag] ,linewidth=5.0,color=color_one)

#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Cost of Crabon: Comparison")
#                 if auto==0:   
#                     plt.ylim(6.5,8.0)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSCC_dis_undis_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSCC_dis_undis_"+Filename+".png")
# plt.close()



# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

#                 color_one = colors[id_xiag % len(xiaarr)]
                
#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scc_dis"]),label='FK: Infinity',linewidth=5.0,linestyle = 'dashed',color=color_one)
#                 else:
#                     plt.plot(res["years"], np.log(res["scc_dis"]),label='FK: $\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]) ,linewidth=5.0,linestyle = 'dashed',color=color_one)
                
#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scc"]),label=labellist[id_xiag],linewidth=5.0,color=color_one)
#                 else:
#                     plt.plot(res["years"], np.log(res["scc"]),label=labellist[id_xiag] ,linewidth=5.0,color=color_one)

#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Cost of Crabon: Comparison")
#                 if auto==0:   
#                     plt.ylim(6.5,8.0)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSCC_orig_dis_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSCC_orig_dis_"+Filename+".png")
# plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                # plt.title("Log of Social Value of R&D")
                # if auto==0:   
                plt.ylim(4.0,8.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSVRD_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/logSVRD_"+Filename+".png")
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            
                color_one = colors[id_xiag % len(xiaarr)]   

                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_dis"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,linestyle = 'dashed',color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_dis"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,linestyle = 'dashed',color=color_one)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)


                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                # plt.title("Log of Social Value of R&D")
                if auto==0:   
                    plt.ylim(6.5,8.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSVRD_dis_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/logSVRD_dis_"+Filename+".png")
plt.close()



# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):


#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_dis"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_dis"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0)

#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Value of R&D")
#                 if auto==0:   
#                     plt.ylim(6.5,8.0)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSVRD_dis_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSVRD_dis_"+Filename+".png")
# plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                
                color_one = colors[id_xiag % len(xiaarr)]   
                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_bundis"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,linestyle = 'dashed',color = color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_bundis"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,linestyle = 'dashed',color = color_one)

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"])[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)


                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("Solid(HJB) Dashed(FK) ")
                # if auto==0:   
                plt.ylim(5.0,8.0)
                plt.xlim(0,30)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSVRD_undis_"+Filename+".pdf")
plt.savefig(Plot_Dir+"/logSVRD_undis_"+Filename+".png")
plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
            

#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scrd_undis"]),label=labellist[id_xiag],linewidth=5.0)
#                 else:
#                     plt.plot(res["years"], np.log(res["scrd_undis"]),label=labellist[id_xiag],linewidth=5.0)

#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Value of R&D")
#                 if auto==0:   
#                     plt.ylim(6.5,8.0)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSVRD_undis_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSVRD_undis_"+Filename+".png")
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):


#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scrd_dis_HJB"]),label=labellist[id_xiag],linewidth=5.0)
#                 else:
#                     plt.plot(res["years"], np.log(res["scrd_dis_HJB"]),label=labellist[id_xiag],linewidth=5.0)

#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Value of R&D")
#                 if auto==0:   
#                     plt.ylim(6.5,8.0)
#                 plt.ylim(5.0,8.0)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSVRD_dis_HJB_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSVRD_dis_HJB_"+Filename+".png")
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):


#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scrd_undis_HJB"]),label=labellist[id_xiag],linewidth=5.0)
#                 else:
#                     plt.plot(res["years"], np.log(res["scrd_undis_HJB"]),label=labellist[id_xiag],linewidth=5.0)



#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Value of R&D")
#                 if auto==0:   
#                     plt.ylim(6.5,8.0)
#                 # plt.ylim(5.5,7.5)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSVRD_undis_HJB_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSVRD_undis_HJB_"+Filename+".png")
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):


#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scrd_undis_HJB"]),label=labellist[id_xiag],linewidth=5.0,linestyle = 'dashed')
#                 else:
#                     plt.plot(res["years"], np.log(res["scrd_undis_HJB"]),label=labellist[id_xiag] ,linewidth=5.0,linestyle = 'dashed')
                
#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scrd"]),label=labellist[id_xiag],linewidth=5.0)
#                 else:
#                     plt.plot(res["years"], np.log(res["scrd"]),label=labellist[id_xiag],linewidth=5.0)

#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Value of R&D: Comparison")
#                 if auto==0:   
#                     plt.ylim(6.5,8.0)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSVRD_orig_undis_HJB_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSVRD_orig_undis_HJB_"+Filename+".png")
# plt.close()




                
# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

#                 color_one = colors[id_xiag % len(xiaarr)]   
                
#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scrd_dis"]),label='FK: Infinity',linewidth=5.0,linestyle = 'dashed',color=color_one)
#                 else:
#                     plt.plot(res["years"], np.log(res["scrd_dis"]),label='FK: $\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]) ,linewidth=5.0,linestyle = 'dashed',color=color_one)
                
#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["scrd"]),label=labellist[id_xiag],linewidth=5.0,color=color_one)
#                 else:
#                     plt.plot(res["years"], np.log(res["scrd"]),label=labellist[id_xiag] ,linewidth=5.0,color=color_one)

#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Value of R&D: Comparison")
#                 if auto==0:   
#                     plt.ylim(6.5,8.0)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSVRD_orig_dis_FK_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSVRD_orig_dis_FK_"+Filename+".png")
# plt.close()



# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

#                 color_one = colors[id_xiag % len(xiaarr)]   

#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_bundis"])[res["states"][:, 1]<1.5],label='FK:  $\\xi_c={:.3f}, \\xi_d={:.3f}, \\xi_g={:.3f}$' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0,linestyle = 'dashed',color=color_one)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_bundis"])[res["states"][:, 1]<1.5],label='FK:  $\\xi_c={:.3f}, \\xi_d={:.3f}, \\xi_g={:.3f}$' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]) ,linewidth=5.0,linestyle = 'dashed',color=color_one)
                
#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"])[res["states"][:, 1]<1.5],label='HJB: $\\xi_c={:.3f}, \\xi_d={:.3f}, \\xi_g={:.3f}$' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0,color=color_one)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"])[res["states"][:, 1]<1.5],label='HJB: $\\xi_c={:.3f}, \\xi_d={:.3f}, \\xi_g={:.3f}$' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]) ,linewidth=5.0,color=color_one)

#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Value of R&D")
#                 # if auto==0:   
#                 plt.ylim(5.5,7.5)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/logSVRD_orig_undis_FK_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/logSVRD_orig_undis_FK_"+Filename+".png")
# plt.close()



# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

#                 color_one = colors[id_xiag % len(xiaarr)]   

#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_cundis"])[res["states"][:, 1]<1.5],label='Only Climate Distorted' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_cundis"])[res["states"][:, 1]<1.5],label='Only Climate Distorted' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0)

#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_tundis"])[res["states"][:, 1]<1.5],label='Only Technology Distorted' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_tundis"])[res["states"][:, 1]<1.5],label='Only Technology Distorted' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0)
                
#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_dundis"])[res["states"][:, 1]<1.5],label='Only Damage Distorted' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_dundis"])[res["states"][:, 1]<1.5],label='Only Damage Distorted' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0)
                
#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_bundis"])[res["states"][:, 1]<1.5],label='All Undistorted' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_bundis"])[res["states"][:, 1]<1.5],label='All Undistorted' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0)
                
#                 if xiaarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_dis"])[res["states"][:, 1]<1.5],label='All Distorted' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0)
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd_dis"])[res["states"][:, 1]<1.5],label='All Distorted' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]),linewidth=5.0)
                                

#                 plt.xlabel("Years")
#                 plt.ticklabel_format(useOffset=False)

#                 # plt.title("Log of Social Value of R&D")
#                 # plt.title("Log of Social Value of R&D")
#                 # if auto==0:   
#                 plt.ylim(5.5,7.5)
#                 plt.xlim(0,30)
#                 plt.legend(loc='upper left')

#     plt.savefig(Plot_Dir+"/logSVRD_decomp_FK,xia={},xic={},xid={},xig={},psi0={},psi1={},varrho={}.pdf".format(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr,psi1arr,varrhoarr))
#     plt.savefig(Plot_Dir+"/logSVRD_decomp_FK,xia={},xic={},xid={},xig={},psi0={},psi1={},varrho={}.png".format(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr,psi1arr,varrhoarr))
#     plt.close()



# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):


#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xigarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000)[res["states"][:, 1]<1.5],label=labellist[id_xiag])
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000)[res["states"][:, 1]<1.5],label='$\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xigarr[id_xiag], xigarr[id_xiag], xigarr[id_xiag],xigarr[id_xiag]))

#                 plt.xlabel("Years")
#                 # plt.title("Mean Difference")
#                 plt.ylim(0,0.8)   
#                 plt.legend()


# plt.savefig(Plot_Dir+"/MeanDiff_"+Filename+".pdf")
# plt.savefig(Plot_Dir+"/MeanDiff_"+Filename+".png")
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

                
#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xigarr[id_xiag]>10:

#                     plt.plot(res["years"], res["ME_total"],label=labellist[id_xiag])
#                 else:
#                     plt.plot(res["years"], res["ME_total"],label='$\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xigarr[id_xiag], xigarr[id_xiag], xigarr[id_xiag],xigarr[id_xiag]))
                    
#                 plt.xlabel("Years")

#                 # plt.title("ME_total")
#                 plt.xlim(0,30)
#                 if auto==0:   
#                     plt.ylim(0,0.000110)   
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_total_"+Filename+".png")
# plt.close()

# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
            
                
#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 if xigarr[id_xiag]>10:

#                     plt.plot(res["years"], np.log(res["temp_Lars"]),label=labellist[id_xiag])
#                 else:
#                     plt.plot(res["years"], np.log(res["temp_Lars"]),label='$\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xigarr[id_xiag], xigarr[id_xiag], xigarr[id_xiag],xigarr[id_xiag]))
                    
#                 plt.xlabel("Years")

#                 # plt.title("Temp")
#                 plt.xlim(0,30)
#                 if auto==0:   
#                     plt.ylim(0,0.000110)   
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/Temp_"+Filename+".png")
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             
#                     grid_info = (Xminarr, Xmaxarr, hXarr)
#                     data_info = (dataname)
#                     varying_argument_extraction = (xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], IntPeriod, timespan)
#                     constant_argument_extraction = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list)
#                     res = model_simulation_graph(grid_info, data_info, varying_argument_extraction, constant_argument_extraction)

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_total2"],label=labellist[id_xiag])
#                     else:
#                         plt.plot(res["years"], res["ME_total2"],label='$\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xigarr[id_xiag], xigarr[id_xiag], xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     # plt.title("ME_total2")
#                     plt.xlim(0,30)
#                     plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_total2_"+Filename+".png")
# plt.close()

# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

#                     res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_base"],label=labellist[id_xiag])
#                     else:
#                         plt.plot(res["years"], res["ME_base"],label='$\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xigarr[id_xiag], xigarr[id_xiag], xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     # plt.title("ME_base")
#                     plt.xlim(0,30)
#                     if auto==0:   
#                         plt.ylim(0,0.000050)   
#                     plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_base_"+Filename+".png")
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             
#                     grid_info = (Xminarr, Xmaxarr, hXarr)
#                     data_info = (dataname)
#                     varying_argument_extraction = (xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], IntPeriod, timespan)
#                     constant_argument_extraction = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list)
#                     res = model_simulation_graph(grid_info, data_info, varying_argument_extraction, constant_argument_extraction)

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_SCC"],label=labellist[id_xiag])
#                     else:
#                         plt.plot(res["years"], res["ME_SCC"],label='$\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xigarr[id_xiag], xigarr[id_xiag], xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     # plt.title("ME_SCC")
#                     plt.xlim(0,30)
#                     plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_SCC_"+Filename+".png")
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             
#                     grid_info = (Xminarr, Xmaxarr, hXarr)
#                     data_info = (dataname)
#                     varying_argument_extraction = (xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], IntPeriod, timespan)
#                     constant_argument_extraction = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list)
#                     res = model_simulation_graph(grid_info, data_info, varying_argument_extraction, constant_argument_extraction)

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_consumption"],label=labellist[id_xiag])
#                     else:
#                         plt.plot(res["years"], res["ME_consumption"],label='$\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xigarr[id_xiag], xigarr[id_xiag], xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     # plt.title("ME_consumption")
#                     plt.xlim(0,30)
#                     plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_consumption_"+Filename+".png")
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

#                     res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_total_base"],label=labellist[id_xiag])
#                     else:
#                         plt.plot(res["years"], res["ME_total_base"],label='$\\xi_a={:.1f}$,$\\xi_r={:.3f}$' .format(xigarr[id_xiag], xigarr[id_xiag], xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     # plt.title("ME_total_base")
#                     plt.xlim(0,30)
#                     # if auto==0:   
#                     #     plt.ylim(0,150)   
#                     plt.legend()

# plt.savefig(Plot_Dir+"/ME_total_base_"+Filename+".png")
# plt.close()

# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])
                
                
#                 if xigarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], (res["Ambiguity_mean_dis"][res["states"][:, 1]<1.5]-res["Ambiguity_mean_undis"][res["states"][:, 1]<1.5])*1000,label='Baseline'  )
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], (res["Ambiguity_mean_dis_h"][res["states"][:, 1]<1.5]-res["Ambiguity_mean_undis"][res["states"][:, 1]<1.5])*1000,label='$\\xi_c={:.3f}, \\xi_d={:.3f}, \\xi_g={:.3f}$' .format(xigarr[id_xiag], xigarr[id_xiag], xigarr[id_xiag],xigarr[id_xiag])  )

#                 plt.xlabel("Years")
#                 # plt.title("Mean Difference")
#                 plt.ylim(0,0.10)
#                 plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/MeanDiff_h_{},xia={},xic={},xid={},xig={},psi0={},psi1={},varrho={}.pdf".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
# plt.savefig(Plot_Dir+"/MeanDiff_h_{},xia={},xic={},xid={},xig={},psi0={},psi1={},varrho={}.png".format(IntPeriod, xiaarr,xicarr,xidarr,xigarr,psi0arr,psi1arr,varrhoarr))
# plt.close()




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

# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

                
#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 # theta_ell_new = res["theta_ell_new"][:,-1]
#                 # histogram of beta_f
#                 theta_ell = pd.read_csv("./data/model144.csv", header=None).to_numpy()[:, 0]
#                 # print("theta_ell")
#                 # print(theta_ell)
#                 # print("theta_ell_new")
#                 # print(theta_ell_new)
#                 pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
#                 # pi_c = np.load("πc_5.npy")
#                 time = 1/timespan
#                 pi_c = res["pic_t"][:, int(time)]


#                 # plt.figure(figsize=(16,10))

#                 print("mean of uncondition = {}" .format(np.average(theta_ell,weights = pi_c_o)))
#                 print("mean of condition = {}" .format(np.average(theta_ell,weights = pi_c)))
                    
#                 plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
#                         alpha=0.5, ec="darkgrey", color="C3",label='Baseline' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]))
#                 plt.hist(theta_ell, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
#                         alpha=0.5, ec="darkgrey", color="C0",label=labellist[id_xiag])
#                 plt.legend(loc='upper left')
#                 # plt.title("Distorted Probability of Climate Models")

#                 plt.ylim(0, 1.4)
#                 plt.xlabel("Climate Sensitivity")
                
#                 plt.savefig(Plot_Dir+"/ClimateSensitivity_0,xia={:.5f},xic={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},varrho={:.1f}.png".format(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho]))
#                 plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
            
                
#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 # theta_ell_new = res["theta_ell_new"][:,-1]
#                 # histogram of beta_f
#                 psi_2 = pd.read_csv("./data/psi2value_p.csv", header=None).to_numpy()[:, 0]
#                 # print("theta_ell")
#                 # print(theta_ell)
#                 # print("theta_ell_new")
#                 # print(theta_ell_new)
#                 pi_c_o = np.ones(len(psi_2)) / len(psi_2)
#                 # pi_c = np.load("πc_5.npy")
#                 time = 1/timespan
#                 pi_c = res["pic_t"][:, int(time)]


#                 # plt.figure(figsize=(16,10))

#                 print("mean of uncondition = {}" .format(np.average(psi_2,weights = pi_c_o)))
#                 print("mean of condition = {}" .format(np.average(psi_2,weights = pi_c)))
                    
#                 # plt.hist(psi_2, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
#                 plt.hist(psi_2, weights=pi_c_o, density=True, 
#                         alpha=0.5, ec="darkgrey", color="C3",label=labellist[id_xiag])
#                 # plt.hist(psi_2, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
#                 plt.hist(psi_2, weights=pi_c, density=True, 
#                         alpha=0.5, ec="darkgrey", color="C0",label=labellist[id_xiag])
#                 plt.legend(loc='upper left')
#                 # plt.title("Distorted probability of R&D Parameters")

#                 plt.ylim(0, 24)
#                 plt.xlabel("R&D Parameter Sensitivity")
                
#                 plt.savefig(Plot_Dir+"/DRSSensitivity_0,xia={:.5f},xic={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},varrho={:.1f}.png".format(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
#                 plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):

                
#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 # theta_ell_new = res["theta_ell_new"][:,-1]
#                 # histogram of beta_f
#                 theta_ell = pd.read_csv("./data/model144.csv", header=None).to_numpy()[:, 0]
#                 # print("theta_ell")
#                 # print(theta_ell)
#                 # print("theta_ell_new")
#                 # print(theta_ell_new)
#                 pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
#                 # pi_c = np.load("πc_5.npy")
#                 time = 1/timespan
#                 pi_c = res["pic_t"][:, -1]

#                 # plt.figure(figsize=(16,10))

#                 print("mean of uncondition = {}" .format(np.average(theta_ell,weights = pi_c_o)))
#                 print("mean of condition = {}" .format(np.average(theta_ell,weights = pi_c)))
                    
#                 plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
#                         alpha=0.5, ec="darkgrey", color="C3",label='Baseline' .format(xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag]))
#                 plt.hist(theta_ell, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
#                         alpha=0.5, ec="darkgrey", color="C0",label=labellist[id_xiag])
#                 plt.legend(loc='upper left')
#                 # plt.title("Distorted Probability of Climate Models")

#                 plt.ylim(0, 1.4)
#                 plt.xlabel("Climate Sensitivity")
                
#                 plt.savefig(Plot_Dir+"/ClimateSensitivity_{},xia={:.5f},xic={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},varrho={:.1f}.png".format(IntPeriod, xiaarr[id_xiag],xicarr[id_xiag], xidarr[id_xiag], xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho]))
#                 plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
            
                
#                 res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

#                 # theta_ell_new = res["theta_ell_new"][:,-1]
#                 # histogram of beta_f
#                 psi_2 = pd.read_csv("./data/psi2value_p.csv", header=None).to_numpy()[:, 0]
#                 # print("theta_ell")
#                 # print(theta_ell)
#                 # print("theta_ell_new")
#                 # print(theta_ell_new)
#                 pi_c_o = np.ones(len(psi_2)) / len(psi_2)
#                 # pi_c = np.load("πc_5.npy")
#                 time = 1/timespan
#                 pi_c = res["pic_t"][:, -1]

#                 # plt.figure(figsize=(16,10))

#                 print("mean of uncondition = {}" .format(np.average(psi_2,weights = pi_c_o)))
#                 print("mean of condition = {}" .format(np.average(psi_2,weights = pi_c)))
                    
#                 # plt.hist(psi_2, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
#                 plt.hist(psi_2, weights=pi_c_o, density=True, 
#                         alpha=0.5, ec="darkgrey", color="C3",label=labellist[id_xiag])
#                 # plt.hist(psi_2, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
#                 plt.hist(psi_2, weights=pi_c, density=True, 
#                         alpha=0.5, ec="darkgrey", color="C0",label=labellist[id_xiag])
#                 plt.legend(loc='upper left')
#                 # plt.title("Distorted probability of R&D Parameters")

#                 plt.ylim(0, 24)
#                 plt.xlabel("R&D Parameter Sensitivity")
                
#                 plt.savefig(Plot_Dir+"/DRSSensitivity_25,xia={:.5f},xic={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},varrho={:.1f}.png".format(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
#                 plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                
                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])

                NUM_DAMAGE = res["gt_dmg"].shape[0]
                gamma_3_list = np.linspace(0., 1./3., NUM_DAMAGE)

                # γ3_distort = np.load("γ3_5.npy")
                print(NUM_DAMAGE)
                
                Year_use = 25
                γ3_distort = res["gt_dmg"][:, int(Year_use/timespan+1)] 
                
                
                # plt.figure(figsize=(16,10))
                plt.hist(gamma_3_list, weights=np.ones(len(gamma_3_list)) / len(gamma_3_list), 
                        alpha=0.5, color="C3", ec="darkgray",label='Baseline', bins=NUM_DAMAGE)
                plt.hist(gamma_3_list, weights= γ3_distort / np.sum(γ3_distort), 
                        alpha=0.5, color="C0", ec="darkgray",label=labellist[id_xiag], bins=NUM_DAMAGE)
                plt.ylim(0, 0.15)
                # plt.title("Distorted Probability of Damage Models")
                plt.xlabel("Damage Curvature")
                plt.legend(loc='upper left',frameon=False)

                plt.savefig(Plot_Dir+"/Gamma3_{}".format(Year_use)+Filename+labellist[id_xiag]+".pdf")
                plt.savefig(Plot_Dir+"/Gamma3_{}".format(Year_use)+Filename+labellist[id_xiag]+".png")
                plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

        
                res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])


                # histogram of beta_f
                theta_ell = pd.read_csv("./data/model144.csv", header=None).to_numpy()[:, 0]
                # print("theta_ell")
                # print(theta_ell)
                # print("theta_ell_new")
                # print(theta_ell_new)
                pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
                # pi_c = np.load("πc_5.npy")

                # pi_c = res["pic_t"][:, -1]
                Year_use = 10
                pi_c = res["pic_t"][:, int(Year_use/timespan+1)]
                
                theta_ell_new = res["theta_ell_new"][:,int(Year_use/timespan+1)]

                # plt.figure(figsize=(16,10))

                # plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                #         alpha=0.5, ec="darkgrey", color="C3",label='Baseline')
                # plt.hist(theta_ell, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                #         alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_r={:.3f}$'.format(xigarr[id_xiag], xigarr[id_xiag], xigarr[id_xiag],xigarr[id_xiag])  )
                # plt.legend(loc='upper left')
                # plt.title("Distorted probability of Climate Models")

                # plt.ylim(0, 1.4)
                # plt.xlabel("Climate Sensitivity")
                # plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ClimateSensitivity_25,xia={:.4f},xic={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},BC.pdf".format(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
                # plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ClimateSensitivity_25,xia={:.4f},xic={:.3f},xid={:.3f},xig={:.3f},psi0={:.3f},psi1={:.3f},BC.png".format(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
                # plt.close()

                plt.figure(figsize=(16,10))

                plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                        alpha=0.5, ec="darkgrey", color="C3",label='Baseline')
                plt.hist(theta_ell_new*1000, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                        alpha=0.5, ec="darkgrey", color="C0",label=labellist[id_xiag])
                plt.legend(loc='upper left')
                # plt.title("Distorted Probability of Climate Models")


                print("mean of uncondition = {}" .format(np.average(theta_ell,weights = pi_c_o)))
                print("mean of condition = {}" .format(np.average(theta_ell_new*1000,weights = pi_c)))
                    

                plt.ylim(0, 1.4)
                plt.xlabel("Climate Sensitivity")
                plt.savefig(Plot_Dir+"/ClimateSensitivity_pmean_{}_".format(Year_use)+Filename+labellist[id_xiag]+".pdf")
                plt.savefig(Plot_Dir+"/ClimateSensitivity_pmean_{}_".format(Year_use)+Filename+labellist[id_xiag]+".png")
                plt.close()
