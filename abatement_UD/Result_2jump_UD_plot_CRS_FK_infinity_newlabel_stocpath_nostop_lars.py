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

def model_simulation_total_true(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho):

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



def model_simulation_total_distorted(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_c_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_" .format(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho)

    res_total = []
    for sim in range(SimPathNum):
        
        with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simuldistorted_{}_path_{}".format(IntPeriod, sim)+ scheme + "_" +HJB_solution, "rb") as f:
            res = pickle.load(f)

        res_total.append(res)
        # print(sim,flush=True)
    
    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simuldistorted_{}_pathtotal_{}".format(IntPeriod,SimPathNum)+ scheme + "_" +HJB_solution, "wb") as f:
        pickle.dump(res_total,f)
        
    return res_total



def model_simulation_var10paths(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho, varname):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_c_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_" .format(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho)


    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_pathtotal_{}".format(IntPeriod,SimPathNum)+ scheme + "_" +HJB_solution, "rb") as f:
        res_total = pickle.load(f)


    var_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))
    TA_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))

    var_matrix1 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix2 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix3 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix4 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix5 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix6 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix7 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix8 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix9 = np.zeros((1, int(IntPeriod/timespan+2)))
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



    var_matrix1 = np.quantile(var_matrix,0,axis=0)
    var_matrix2 = np.quantile(var_matrix,0.11,axis=0)
    var_matrix3 = np.quantile(var_matrix,0.22,axis=0)
    var_matrix4 = np.quantile(var_matrix,0.33,axis=0)
    var_matrix5 = np.quantile(var_matrix,0.44,axis=0)
    var_matrix6 = np.quantile(var_matrix,0.55,axis=0)
    var_matrix7 = np.quantile(var_matrix,0.66,axis=0)
    var_matrix8 = np.quantile(var_matrix,0.77,axis=0)
    var_matrix9 = np.quantile(var_matrix,0.88,axis=0)
    var_matrix10 = np.quantile(var_matrix,0.99,axis=0)

    var_year = res_total[sim]["years"]
    
    return var_matrix1, var_matrix2, var_matrix3, var_matrix4, var_matrix5, var_matrix6, var_matrix7, var_matrix8, var_matrix9, var_matrix10, var_year



def model_simulation_varrandom10paths(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho, varname):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_c_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_" .format(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho)


    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_pathtotal_{}".format(IntPeriod,SimPathNum)+ scheme + "_" +HJB_solution, "rb") as f:
        res_total = pickle.load(f)


    var_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))
    TA_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))

    var_matrix1 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix2 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix3 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix4 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix5 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix6 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix7 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix8 = np.zeros((1, int(IntPeriod/timespan+2)))
    var_matrix9 = np.zeros((1, int(IntPeriod/timespan+2)))
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



    var_matrix1 = var_matrix[1,:]
    var_matrix2 = var_matrix[10,:]
    var_matrix3 = var_matrix[20,:]
    var_matrix4 = var_matrix[30,:]
    var_matrix5 = var_matrix[40,:]
    var_matrix6 = var_matrix[50,:]
    var_matrix7 = var_matrix[60,:]
    var_matrix8 = var_matrix[70,:]
    var_matrix9 = var_matrix[80,:]
    var_matrix10 = var_matrix[90,:]

    var_year = res_total[sim]["years"]
    
    return var_matrix1, var_matrix2, var_matrix3, var_matrix4, var_matrix5, var_matrix6, var_matrix7, var_matrix8, var_matrix9, var_matrix10, var_year




def model_simulation_var_accprob(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho, varname):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_c_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_" .format(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho)




    if varname == "true_tech_prob" or varname =="true_damage_prob":
        
        
        with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_pathtotal_{}".format(IntPeriod,SimPathNum)+ scheme + "_" +HJB_solution, "rb") as f:
            res_total = pickle.load(f)


        var_matrix = np.zeros((SimPathNum, 1))
        accvar_matrix = np.zeros((int(IntPeriod/timespan+2)))
        TA_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))

        
        # print(varname)
        # print(res_total[0].keys())
        count = 0
        yearcount = 0
        print(varname)

        if varname == "true_tech_prob":
            varname = "tech_activate_year"
            print(varname)

        elif varname == "true_damage_prob":
            varname = "damage_activate_year"    
            print(varname)

        for sim in range(SimPathNum):

            var_matrix[sim] = res_total[sim][varname]

    if varname == "distorted_tech_prob" or varname == "distorted_damage_prob":
        
        
        with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simuldistorted_{}_pathtotal_{}".format(IntPeriod,SimPathNum)+ scheme + "_" +HJB_solution, "rb") as f:
            res_total = pickle.load(f)


        var_matrix = np.zeros((SimPathNum, 1))
        accvar_matrix = np.zeros((int(IntPeriod/timespan+2)))
        TA_matrix = np.zeros((SimPathNum, int(IntPeriod/timespan+2)))

        
        print(varname)
        # print(res_total[0].keys())
        count = 0
        yearcount = 0

        if varname == "distorted_tech_prob":
            varname = "tech_activate_year"
            print(varname)
        elif varname == "distorted_damage_prob":
            varname = "damage_activate_year"    
            print(varname)

            
        for sim in range(SimPathNum):

            var_matrix[sim] = res_total[sim][varname]




    var_year = res_total[sim]["years"]
    
    for time in range(int(IntPeriod/timespan+2)):
        accvar_matrix[time] = np.sum((var_matrix<=var_year[time]))/SimPathNum
    
    
    print("count={},yearaverage={}".format(count,yearcount/SimPathNum))
    return accvar_matrix, var_year



print("loading into total data true")

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):
                res = model_simulation_total_true(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])
                
                
print("loading into total data distorted")

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):
                res = model_simulation_total_distorted(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho])


print("Finish loading into total data")

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, years = model_simulation_var10paths(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "x")
                
                fig, axes = plt.subplots(1,1,figsize = (12,9))

                axes.plot(years,res1,label = '0%')
                axes.plot(years,res2,label = '11%')
                axes.plot(years,res3,label = '22%')
                axes.plot(years,res4,label = '33%')
                axes.plot(years,res5,label = '44%')
                axes.plot(years,res6,label = '55%')
                axes.plot(years,res7,label = '66%')
                axes.plot(years,res8,label = '77%')
                axes.plot(years,res9,label = '88%')
                axes.plot(years,res10,label = '99%')
                # axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
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
                 
                plt.savefig(Plot_Dir+"/RD_10paths_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, years = model_simulation_varrandom10paths(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "e")
                
                fig, axes = plt.subplots(1,1,figsize = (12,9))

                axes.plot(years,res1,label = '0%')
                axes.plot(years,res2,label = '11%')
                axes.plot(years,res3,label = '22%')
                axes.plot(years,res4,label = '33%')
                axes.plot(years,res5,label = '44%')
                axes.plot(years,res6,label = '55%')
                axes.plot(years,res7,label = '66%')
                axes.plot(years,res8,label = '77%')
                axes.plot(years,res9,label = '88%')
                axes.plot(years,res10,label = '99%')
                # axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                axes.set_title("Emission")

                axes.set_ylim(0,20)
                axes.set_xlim(0,IntPeriod)


                # axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/E_10paths_"+Filename+labellist[id_xiag]+".png")
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, years = model_simulation_varrandom10paths(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "x")
                
                fig, axes = plt.subplots(1,1,figsize = (12,9))

                axes.plot(years,res1)
                axes.plot(years,res2)
                axes.plot(years,res3)
                axes.plot(years,res4)
                axes.plot(years,res5)
                axes.plot(years,res6)
                axes.plot(years,res7)
                axes.plot(years,res8)
                axes.plot(years,res9)
                axes.plot(years,res10)
                # axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
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

                # axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/RD_random10paths_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, years = model_simulation_var10paths(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "e")
                
                fig, axes = plt.subplots(1,1,figsize = (12,9))

                axes.plot(years,res1)
                axes.plot(years,res2)
                axes.plot(years,res3)
                axes.plot(years,res4)
                axes.plot(years,res5)
                axes.plot(years,res6)
                axes.plot(years,res7)
                axes.plot(years,res8)
                axes.plot(years,res9)
                axes.plot(years,res10)
                # axes.fill_between(years, res10, res90,  color='red', alpha=0.3)
                axes.set_xlabel("Years")
                axes.set_title("Emission")

                axes.set_ylim(0,20)
                axes.set_xlim(0,IntPeriod)


                # axes.legend(loc='upper left')       
                 
                plt.savefig(Plot_Dir+"/E_random10paths_"+Filename+labellist[id_xiag]+".png")
                plt.close()






for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res, years = model_simulation_var_accprob(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "true_tech_prob")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Accumulated True Fraction of Tech Jump")

    
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
                 
                plt.savefig(Plot_Dir+"/AccTrueFracTechJump_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res, years = model_simulation_var_accprob(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "true_damage_prob")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Accumulated True Fraction of Damage Jump")

    
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
                 
                plt.savefig(Plot_Dir+"/AccTrueFracDamageJump_"+Filename+labellist[id_xiag]+".png")
                plt.close()






for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res, years = model_simulation_var_accprob(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "distorted_tech_prob")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Accumulated Distorted Fraction of Tech Jump")

    
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
                 
                plt.savefig(Plot_Dir+"/AccDistortedFracTechJump_"+Filename+labellist[id_xiag]+".png")
                plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):

                color_one = colors[id_xiag % len(xiaarr)]   

                res, years = model_simulation_var_accprob(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], varrhoarr[id_varrho], "distorted_damage_prob")
                
                fig, axes = plt.subplots(1,1,figsize = (14,8))

                axes.plot(years,res)
                axes.set_xlabel("Years")
                # axes.set_ylabel("$\%$ of GDP")
                axes.set_title("Accumulated Distorted Fraction of Damage Jump")

    
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
                 
                plt.savefig(Plot_Dir+"/AccDistortedFracDamageJump_"+Filename+labellist[id_xiag]+".png")
                plt.close()
