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
        # print(sim,flush=True)
    
    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_pathtotal_{}".format(IntPeriod,SimPathNum)+ scheme + "_" +HJB_solution, "wb") as f:
        pickle.dump(res_total,f)
        
    return res_total



def model_simulation_var(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho, varname):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_c_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_" .format(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho)


    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_pathtotal_{}".format(IntPeriod,SimPathNum)+ scheme + "_" +HJB_solution, "rb") as f:
        res_total = pickle.load(f)


    var_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))
    TA_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))
    var_matrix90 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix50 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix10 = np.zeros((1, int(IntPeriod/timespan+2)))
    
    print(varname)
    for sim in range(SimPathNum):

        if varname == "x":

            var_matrix[sim,:] =  (res_total[sim]["x"]/(alpha*np.exp( res_total[sim]["states"][:,0])))*100
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        if varname == "rowx":

            var_matrix[sim,:] = res_total[sim]["x"]
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "TA":
            
            var_matrix[sim,:] =  res_total[sim]["states"][:, 1]
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "Jg":
            var_matrix[sim,:] =  np.exp(res_total[sim]["states"][:, 2])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]

        elif varname == "scc":
            var_matrix[sim,:] =  np.log(res_total[sim]["scc"])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "scrd":
            var_matrix[sim,:] =  np.log(res_total[sim]["scrd"])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "scgw":
            var_matrix[sim,:] =  np.log(res_total[sim]["scgw"])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        else :
            var_matrix[sim,:] = res_total[sim][varname] 
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]

    TA_matrix50 = np.quantile(TA_matrix,0.5,axis=0)


    var_matrix90 = np.quantile(var_matrix,0.9,axis=0)
    var_matrix50 = np.quantile(var_matrix,0.5,axis=0)
    var_matrix10 = np.quantile(var_matrix,0.1,axis=0)

    var_year = res_total[sim]["years"]
    
    return var_matrix90, var_matrix50, var_matrix10, var_year



def model_simulation_var_prob(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho, varname):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_c_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_" .format(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho)


    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_pathtotal_{}".format(IntPeriod,SimPathNum)+ scheme + "_" +HJB_solution, "rb") as f:
        res_total = pickle.load(f)


    var_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))
    TA_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))
    var_matrix90 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix50 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix10 = np.zeros((1, int(IntPeriod/timespan+2)))
    
    print(varname)
    print(res_total[0].keys())
    count = 0
    yearcount = 0
    for sim in range(SimPathNum):

        if varname == "x":
            # print((res_total[sim]["x"]).shape)
            # print((res_total[sim]["states"]).shape)
            # print((res_total[sim]["states"][:,0]).shape)
            # print((res_total[sim]["years"]).shape)
            var_matrix[sim,:] =  (res_total[sim]["x"]/(alpha*np.exp( res_total[sim]["states"][:,0])))*100
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]

        elif varname == "TA":
            
            var_matrix[sim,:] =  res_total[sim]["states"][:, 1]
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "Jg":
            var_matrix[sim,:] =  np.exp(res_total[sim]["states"][:, 2])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]

        elif varname == "scc":
            var_matrix[sim,:] =  np.log(res_total[sim]["scc"])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "scrd":
            var_matrix[sim,:] =  np.log(res_total[sim]["scrd"])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "scgw":
            var_matrix[sim,:] =  np.log(res_total[sim]["scgw"])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "tech_activate_year":
            # print(res_total[sim][varname])
            count += (res_total[sim][varname] < 10000)
            yearcount += (res_total[sim][varname] < 10000) * res_total[sim][varname]  + (res_total[sim][varname] > 1000) * IntPeriod
            # print(res_total[sim].keys())
            index = min(int(res_total[sim][varname]/timespan), int(IntPeriod/timespan+2)-1)
            # print("techindex={}".format(index))
            var_matrix[sim,index:] = 1
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "damage_activate_year":
            count += (res_total[sim][varname] < 10000)
            yearcount += (res_total[sim][varname] < 10000) * res_total[sim][varname]  + (res_total[sim][varname] > 1000) * IntPeriod
           
            index = min(int(res_total[sim][varname]/timespan), int(IntPeriod/timespan+2)-1)
            var_matrix[sim,index:] = 1
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        else :
            var_matrix[sim,:] = res_total[sim][varname] 
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]

    TA_matrix50 = np.quantile(TA_matrix,0.5,axis=0)


    var_matrix90 = np.quantile(var_matrix,0.9,axis=0)
    var_matrix50 = np.quantile(var_matrix,0.5,axis=0)
    var_matrix10 = np.quantile(var_matrix,0.1,axis=0)
    var_year = res_total[sim]["years"]
    print("count={},yearaverage={}".format(count,yearcount/SimPathNum))
    return var_matrix90, var_matrix50, var_matrix10, var_year


def model_simulation_var_prob99(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho, varname):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_c_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_" .format(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho)


    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_pathtotal_{}".format(IntPeriod,SimPathNum)+ scheme + "_" +HJB_solution, "rb") as f:
        res_total = pickle.load(f)


    var_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))
    TA_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))
    var_matrix90 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix50 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix10 = np.zeros((1, int(IntPeriod/timespan+2)))
    
    print(varname)
    print(res_total[0].keys())
    count = 0
    yearcount = 0
    for sim in range(SimPathNum):

        if varname == "x":
            # print((res_total[sim]["x"]).shape)
            # print((res_total[sim]["states"]).shape)
            # print((res_total[sim]["states"][:,0]).shape)
            # print((res_total[sim]["years"]).shape)
            var_matrix[sim,:] =  (res_total[sim]["x"]/(alpha*np.exp( res_total[sim]["states"][:,0])))*100
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]

        elif varname == "TA":
            
            var_matrix[sim,:] =  res_total[sim]["states"][:, 1]
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "Jg":
            var_matrix[sim,:] =  np.exp(res_total[sim]["states"][:, 2])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]

        elif varname == "scc":
            var_matrix[sim,:] =  np.log(res_total[sim]["scc"])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "scrd":
            var_matrix[sim,:] =  np.log(res_total[sim]["scrd"])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "scgw":
            var_matrix[sim,:] =  np.log(res_total[sim]["scgw"])
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "tech_activate_year":
            # print(res_total[sim][varname])
            count += (res_total[sim][varname] < 10000)
            yearcount += (res_total[sim][varname] < 10000) * res_total[sim][varname]  + (res_total[sim][varname] > 1000) * IntPeriod
            # print(res_total[sim].keys())
            index = min(int(res_total[sim][varname]/timespan), int(IntPeriod/timespan+2)-1)
            # print("techindex={}".format(index))
            var_matrix[sim,index:] = 1
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        elif varname == "damage_activate_year":
            count += (res_total[sim][varname] < 10000)
            yearcount += (res_total[sim][varname] < 10000) * res_total[sim][varname]  + (res_total[sim][varname] > 1000) * IntPeriod
           
            index = min(int(res_total[sim][varname]/timespan), int(IntPeriod/timespan+2)-1)
            var_matrix[sim,index:] = 1
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]
        else :
            var_matrix[sim,:] = res_total[sim][varname] 
            TA_matrix[sim,:] = res_total[sim]["states"][:, 1]

    TA_matrix50 = np.quantile(TA_matrix,0.5,axis=0)


    var_matrix90 = np.quantile(var_matrix,0.99,axis=0)
    var_matrix50 = np.quantile(var_matrix,0.5,axis=0)
    var_matrix10 = np.quantile(var_matrix,0.01,axis=0)
    var_year = res_total[sim]["years"]
    print("count={},yearaverage={}".format(count,yearcount/SimPathNum))
    return var_matrix90, var_matrix50, var_matrix10, var_year


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_varrho in range(len(varrhoarr)):
#                 print("loading into total data")
#                 res = model_simulation_total(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])


print("Finish loading into total data")

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
                axes.set_title("R&D investment as percentage of  GDP")
    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,3.5)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/RD_"+Filename+labellist[id_xiag]+".png")
                plt.close()




for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "rowx")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Row R&D investment")
    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,3.2)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/RowRD_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "e")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Emission")

    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,20)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/E_"+Filename+labellist[id_xiag]+".png")
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "scrd")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Log of Social Value of R&D")

    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,10.0)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/logSVRD_"+Filename+labellist[id_xiag]+".png")
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "scc")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                axes.set_title("Log of Social Cost of Carbon")

                axes.set_ylim(0,8)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/logSCC_"+Filename+labellist[id_xiag]+".png")
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "scgw")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                axes.set_title("Log of Social Cost of Global Warming")

                axes.set_ylim(0,20.0)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/logSCGW_"+Filename+labellist[id_xiag]+".png")
                plt.close()





for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "TA")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Temperature Anomaly")

    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(1.1,2.5)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/TA_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "Jg")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                axes.set_title("Knowledge Stock")

                axes.set_ylim(11.0,40.0)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/Jg_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var_prob(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "distorted_tech_prob")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Distorted Probability of Tech Jump")

    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,1)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/ProbTechJump_"+Filename+labellist[id_xiag]+".png")
                plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var_prob(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "true_tech_prob")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("True Probability of Tech Jump")

    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,1)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/TrueProbTechJump_"+Filename+labellist[id_xiag]+".png")
                plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var_prob99(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "distorted_tech_prob")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Distorted Probability of Tech Jump")

    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,1)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/ProbTechJump99_"+Filename+labellist[id_xiag]+".png")
                plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var_prob99(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "true_tech_prob")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("True Probability of Tech Jump")

    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,1)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/TrueProbTechJump99_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var_prob(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "distorted_damage_prob")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                axes.set_title("Distorted Probability of Damage Change")

                axes.set_ylim(0,1)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/ProbDamageChange_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var_prob(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "true_damage_prob")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                axes.set_title("True Probability of Damage Change")

                axes.set_ylim(0,1)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/TrueProbDamageChange_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var_prob(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "tech_activate_year")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Year of Tech Jump")

    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,1)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/YearTechJump_"+Filename+labellist[id_xiag]+".png")
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var_prob(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "damage_activate_year")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                axes.set_title("Year of Damage Change")

                axes.set_ylim(0,1)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/YearDamageChange_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var_prob99(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "tech_activate_year")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Year of Tech Jump")

    
                # if xiaarr[id_xiag]>10:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # else:
                #     plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label=labellist[id_xiag],linewidth=5.0,color=color_one)
                # plt.xlabel('Years')
                # plt.ylabel('$\%$ of GDP')
                # plt.title("R&D investment as percentage of  GDP")
                # if auto==0:   
                axes.set_ylim(0,1)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/YearTechJump99_"+Filename+labellist[id_xiag]+".png")
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res90,res50,res10, years = model_simulation_var_prob99(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "damage_activate_year")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res50,label = labellist[id_xiag],ls = "-",color = 'black')
                axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                axes.set_title("Year of Damage Change")

                axes.set_ylim(0,1)
                axes.set_xlim(0,IntPeriod)

                axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/YearDamageChange99_"+Filename+labellist[id_xiag]+".png")
                plt.close()
