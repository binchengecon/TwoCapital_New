import numpy as np
import pandas as pd
import sys
print(sys.path)

sys.path.append('./src')

import pickle
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib as mpl
import matplotlib.pyplot as plt
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
from src.Utility import finiteDiff_3D
import os
import argparse
import src.ResultSolver_CRS
import SolveLinSys


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--dataname",type=str)
parser.add_argument("--pdfname",type=str)

parser.add_argument("--SimEpoch",type=int)
parser.add_argument("--SimEpochNum",type=int)
parser.add_argument("--SimEpisode",type=int)


parser.add_argument("--xiaarr",nargs='+', type=float)
parser.add_argument("--xicarr",nargs='+', type=float)
parser.add_argument("--xidarr",nargs='+', type=float)
parser.add_argument("--xigarr",nargs='+', type=float)

parser.add_argument("--varrhoarr",nargs='+', type=float)

parser.add_argument("--psi0arr",nargs='+',type=float)
parser.add_argument("--psi1arr",nargs='+',type=float)
parser.add_argument("--psi2arr",nargs='+',type=float)

parser.add_argument("--hXarr",nargs='+',type=float)
parser.add_argument("--Xminarr",nargs='+',type=float)
parser.add_argument("--Xmaxarr",nargs='+',type=float)

parser.add_argument("--auto",type=int)
parser.add_argument("--IntPeriod",type=int)

parser.add_argument("--scheme",type=str)
parser.add_argument("--HJB_solution",type=str)


args = parser.parse_args()



# Update = args.Update
IntPeriod = args.IntPeriod
timespan = 1/12
SimEpoch = args.SimEpoch
SimEpochNum = args.SimEpochNum
SimEpisode = args.SimEpisode

# psi0arr = np.array([0.006,0.009])
# # # psi0arr = np.array([0.009])
# # # psi1arr = np.array([.5,.7,.9])
# psi1arr = np.array([.3,.4])

psi0arr = args.psi0arr
psi1arr = args.psi1arr
psi2arr = args.psi2arr
xiaarr = args.xiaarr
xicarr = args.xicarr 
xidarr = args.xidarr 
xigarr = args.xigarr 
varrhoarr = args.varrhoarr

Xminarr = args.Xminarr
Xmaxarr = args.Xmaxarr
hXarr = args.hXarr

auto = args.auto

scheme = args.scheme
HJB_solution = args.HJB_solution


delta = 0.01
alpha = 0.115
kappa = 6.667
mu_k  = -0.043
# sigma_k = np.sqrt(0.0087**2 + 0.0038**2)
sigma_k = 0.100
beta_f = 1.86/1000
sigma_y = 1.2 * 1.86 / 1000
zeta = 0.0
# psi_0 = 0.00025
# psi_1 = 1/2
# sigma_g   = 0.016
sigma_g   = 0.078


gamma_1 = 1.7675 / 1000
gamma_2 = 0.0022 * 2




y_bar = 2.
y_bar_lower = 1.5


# Tech
theta = 3
lambda_bar = 0.1206
# vartheta_bar = 0.0453
# vartheta_bar = 0.05
vartheta_bar = 0.056

lambda_bar_first = lambda_bar / 2.
vartheta_bar_first = vartheta_bar / 2.

lambda_bar_second = 1e-3
vartheta_bar_second = 0.





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


id_2 = np.abs(Y - y_bar).argmin()
Y_min_short = Xminarr[3]
Y_max_short = Xmaxarr[3]
Y_short     = np.arange(Y_min_short, Y_max_short + hY, hY)
nY_short    = len(Y_short)

n_bar1 = len(Y_short)-1
n_bar2 = np.abs(Y_short - y_bar).argmin()


# print("bY_short={:d}".format(nY_short))
(K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_short, L, indexing="ij")







mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (16,10)
mpl.rcParams["font.size"] = 15
mpl.rcParams["legend.frameon"] = False
mpl.style.use('classic')
mpl.rcParams["lines.linewidth"] = 5


# print("After, figure default size is: ", plt.rcParams["savefig.bbox"])
# print("After, figure default size is: ", plt.rcParams["figure.figsize"])
# print("After, figure default dpi is: ", plt.rcParams["figure.dpi"])
# print("After, figure default size is: ", plt.rcParams["font.size"])
# print("After, legend.frameon is: ", plt.rcParams["legend.frameon"])
# print("After, lines.linewidth is: ", plt.rcParams["lines.linewidth"])

def simulate_pre(
    grid = (), 
    model_args = (), 
    controls = (),
    ME = (),
    FK = (),
    n_bar = (),  
    postdamage = (),
    FK_postdamageDistorted =(),
    FK_postdamageUndistorted =(),
    FK_postdamage_Y_Distorted = (),
    FK_postdamage_Y_Undistorted = (),
    model_tech2_post_damage= (),
    Y_long=(),
    initial=(np.log(85/0.115), 1.1, np.log(11.2)), 
    T0=0, T=40, dt=1/12,
    printing=True):

    K, Y, L = grid

    # if printing==True:
    #     print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K.min(),K.max(),Y.min(),Y.max(),L.min(),L.max()))

    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y), max(Y), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]

    delta, mu_k, kappa, sigma_k, beta_f, zeta, psi_0, psi_1, sigma_g, theta, lambda_bar, vartheta_bar, varrho = model_args
    ii, ee, xx, g_tech, g_damage, pi_c, h, v = controls
    ME_base = ME
    # dvdL_dis, dvdL_undis = FK
    # dvdL_dis, dvdL_undis, dvdL_dis_HJB, dvdL_Undis_HJB, dvdL_Undis_HJB_New, dvdY_dis, ddvddY_dis, dvdY_undis, ddvddY_undis  = FK
    # dvdL_dis, dvdL_cundis, dvdL_tundis, dvdL_dundis, dvdL_bundis, dvdL_dis_HJB, dvdL_Undis_HJB, dvdY_dis, ddvddY_dis, dvdY_undis, ddvddY_undis  = FK
    dvdL_dis, dvdL_bundis, dvdY_dis, ddvddY_dis, dvdY_undis, ddvddY_undis  = FK
    n_bar = n_bar
    K_0, Y_0, L_0 = initial
    postdamage = postdamage
    FK_postdamageDistorted = FK_postdamageDistorted
    FK_postdamageUndistorted = FK_postdamageUndistorted
    FK_postdamage_Y_Distorted = FK_postdamage_Y_Distorted
    FK_postdamage_Y_Undistorted = FK_postdamage_Y_Undistorted
    model_tech2_post_damage = model_tech2_post_damage
    Y_long = Y_long
    
    Y = Y[:n_bar+1]
    
    ii = ii[:,:n_bar+1,:]
    ee = ee[:,:n_bar+1,:]
    xx = xx[:,:n_bar+1,:]
    g_tech = g_tech[:,:n_bar+1,:]
    g_damage = g_damage[:,:,:n_bar+1,:]
    pi_c = pi_c[:,:,:n_bar+1,:]
    h = h[:,:n_bar+1,:]
    v = v[:,:n_bar+1,:]
    


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')

    jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
    
    jj[jj <= 1e-16] = 1e-16
    consumption = alpha - ii - jj - xx
    ME_total = delta/ consumption  * alpha * vartheta_bar * theta * (1 - ee / ( alpha * lambda_bar * np.exp(K_mat)))**(theta - 1) /( alpha * lambda_bar * np.exp(K_mat) )


    years  = np.arange(T0, T0 + T + dt, dt)
    pers   = len(years)
       

    # some parameters remaiend unchanged across runs
    gamma_1  = 0.00017675
    gamma_2  = 2. * 0.0022
    beta_f   = 1.86 / 1000
    sigma_y  = 1.2 * 1.86 / 1000
    
    theta_ell = pd.read_csv("./data/model144.csv", header=None).to_numpy()[:, 0]/1000.
    pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
    pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
    # theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])

    dL = finiteDiff_3D(v, 2,1,hL )
    dY = finiteDiff_3D(v, 1,1,hY )

    gridpoints = (K, Y, L)

    i_func = RegularGridInterpolator(gridpoints, ii)
    e_func = RegularGridInterpolator(gridpoints, ee)
    x_func = RegularGridInterpolator(gridpoints, xx)
    tech_func = RegularGridInterpolator(gridpoints, g_tech)
    h_func = RegularGridInterpolator(gridpoints, h)
    
    dL_func   = RegularGridInterpolator(gridpoints, dL)
    dY_func   = RegularGridInterpolator(gridpoints, dY)
    
    dvdL_dis_func   = RegularGridInterpolator(gridpoints, dvdL_dis)
    dvdL_bundis_func   = RegularGridInterpolator(gridpoints, dvdL_bundis)
    dvdY_dis_func   = RegularGridInterpolator(gridpoints, dvdY_dis)
    ddvddY_dis_func   = RegularGridInterpolator(gridpoints, ddvddY_dis)
    dvdY_undis_func   = RegularGridInterpolator(gridpoints, dvdY_undis)
    ddvddY_undis_func   = RegularGridInterpolator(gridpoints, ddvddY_undis)
    
    n_damage = len(g_damage)

    damage_func_list = []
    for i in range(n_damage):
        func_i = RegularGridInterpolator(gridpoints, g_damage[i])
        damage_func_list.append(func_i)
        
    n_climate = len(pi_c)
    
    climate_func_list = []
    for i in range(n_climate):
        func_i = RegularGridInterpolator(gridpoints, pi_c[i])
        climate_func_list.append(func_i)


    techjump_func   = RegularGridInterpolator(gridpoints, np.exp(L_mat)/varrho)
    damagejump_func   = RegularGridInterpolator(gridpoints, Damage_Intensity(Y_mat))

    
    ME_total_func = RegularGridInterpolator(gridpoints, ME_total)
    ME_base_func = RegularGridInterpolator(gridpoints, ME_base)
    
    iipost = np.zeros((len(K), len(Y_long), len(L)))
    
    for i in range(len(L)):
        iipost[:,:,i] = model_tech2_post_damage[0]["i"]
        
    def get_i(x):
        return i_func(x)

    def get_e(x):
        return e_func(x)
    
    def get_x(x):
        return x_func(x)

    def get_dL(x):
        return dL_func(x)


    def mu_K(i_x):
        return mu_k + i_x - 0.5 * kappa * i_x ** 2  - 0.5 * sigma_k ** 2
    
    def mu_L(Xt, state):
        return -zeta + psi_0 * Xt **psi_1 * (np.exp( psi_1 * (state[0]-state[2])) )  - 0.5 * sigma_g**2
    
    
    hist      = np.zeros([pers, 3])
    i_hist    = np.zeros([pers])
    e_hist    = np.zeros([pers])
    x_hist    = np.zeros([pers])
    scc_hist  = np.zeros([pers])
    gt_tech   = np.zeros([pers])
    ht   = np.zeros([pers])
    
    dL_hist    = np.zeros([pers])
    dY_hist    = np.zeros([pers])
    
    dvdL_dis_hist    = np.zeros([pers])
    dvdL_bundis_hist    = np.zeros([pers])
    
    
    dvdY_dis_hist    = np.zeros([pers])
    ddvddY_dis_hist    = np.zeros([pers])
    dvdY_undis_hist    = np.zeros([pers])
    ddvddY_undis_hist    = np.zeros([pers])


    gt_dmg    = np.zeros([n_damage, pers])
    pi_c_t = np.zeros([n_climate, pers])
    Ambiguity_mean_undis = np.zeros([pers])
    Ambiguity_mean_dis = np.zeros([pers])
    Ambiguity_mean_dis_h = np.zeros([pers])

    ME_base_hist = np.zeros([pers])
    ME_total_hist = np.zeros([pers])

    mu_K_hist = np.zeros([pers])
    mu_L_hist = np.zeros([pers])
    techjump_realization = np.zeros([pers])
    damagejump_realization = np.zeros([pers])
    
    theta_ell_hist = np.zeros([len(theta_ell),pers])
    theta_ell_hist2 = np.zeros([pers])


    logK_BM = np.random.normal(loc=0, scale = np.sqrt(dt), size = (pers,1))
    Y_BM = np.random.normal(loc=0, scale =np.sqrt(dt), size = (pers,1))
    logJ_BM = np.random.normal(loc=0, scale =np.sqrt(dt), size = (pers,1))
    
    tech_activate = 0
    damage_activate =0
    
    tech_activate_year = 10000
    damage_activate_year = 10000
    
    
    for tm in range(pers):
        if tm == 0:

            # initial points
            hist[0,:] = [K_0, Y_0, L_0] # logL
            i_hist[0] = get_i(hist[0, :])
            e_hist[0] = get_e(hist[0, :])
            x_hist[0] = get_x(hist[0, :])
            mu_K_hist[0] = mu_K(i_hist[0])
            mu_L_hist[0] = mu_L(x_hist[0], hist[0,:])
            gt_tech[0] = tech_func(hist[0, :])
            ht[0] = h_func(hist[0, :])
            dL_hist[tm] = dL_func(hist[0,:])
            dY_hist[tm] = dY_func(hist[0,:])
            dvdL_dis_hist[tm]    = dvdL_dis_func(hist[0,:])

            dvdL_bundis_hist[tm]    = dvdL_bundis_func(hist[0,:])
            


            dvdY_dis_hist[tm]    = dvdY_dis_func(hist[0,:])
            ddvddY_dis_hist[tm]    = ddvddY_dis_func(hist[0,:])
            dvdY_undis_hist[tm]    = dvdY_undis_func(hist[0,:])
            ddvddY_undis_hist[tm]    = ddvddY_undis_func(hist[0,:])



            for i in range(n_damage):
                damage_func = damage_func_list[i]
                gt_dmg[i, 0] = damage_func(hist[0, :])
            
            for i in range(n_climate):
                climate_func = climate_func_list[i]
                pi_c_t[i, 0] = climate_func(hist[0, :])
            Ambiguity_mean_undis[tm] = np.mean(theta_ell)
            Ambiguity_mean_dis[tm] = np.average(theta_ell,weights=pi_c_t[:,tm])
            Ambiguity_mean_dis_h[tm] = np.average(theta_ell + sigma_y*ht[tm],weights=pi_c_t[:,tm])

            ME_total_hist[0] = ME_total_func(hist[0,:])
            ME_base_hist[0] = ME_base_func(hist[0,:])
            
            theta_ell_hist[:,tm] = theta_ell + sigma_y*ht[tm]
            theta_ell_hist2[tm] = np.average(theta_ell, weights= pi_c_t[:,tm]) 

        else:
            
            DU = np.random.uniform(0,1); 

            ldt = techjump_func(hist[tm-1,:])*dt
            ul = (1-ldt)/2
            ur = (1+ldt)/2
            techjump_realization[tm] = (DU <= ur) * ( DU >= ul)

            DU2 = np.random.uniform(0,1); 

            ldt2 = damagejump_func(hist[tm-1,:])*dt
            ul2 = (1-ldt2)/2
            ur2 = (1+ldt2)/2
            damagejump_realization[tm] = (DU2 <= ur2) * ( DU2 >= ul2)
            
            

            if techjump_realization[tm] == 1 and tech_activate==0:
                tech_activate = 1
                tech_activate_year = tm*timespan
                

            if damagejump_realization[tm] == 1 and damage_activate==0:
                damage_activate = 1
                damage_activate_year = tm*timespan
                gamma_3_list = np.linspace(0,1/3,20)
                id_3_list = np.arange(20)
                id_3 = np.random.choice(id_3_list)
                
                
            if tech_activate==0 and damage_activate ==0: 
                # pre damage and pre tech case
                
                i_hist[tm] = get_i(hist[tm-1,:])
                e_hist[tm] = get_e(hist[tm-1,:])
                x_hist[tm] = get_x(hist[tm-1,:])
                gt_tech[tm] = tech_func(hist[tm-1,:])
                ht[tm] = h_func(hist[tm-1, :])
                
                dL_hist[tm] = dL_func(hist[tm-1,:])
                dY_hist[tm] = dY_func(hist[tm-1,:])
                
                dvdL_dis_hist[tm]    = dvdL_dis_func(hist[tm-1,:])
                dvdL_bundis_hist[tm]    = dvdL_bundis_func(hist[tm-1,:])
                
                dvdY_dis_hist[tm]    = dvdY_dis_func(hist[tm-1,:])
                ddvddY_dis_hist[tm]    = ddvddY_dis_func(hist[tm-1,:])
                dvdY_undis_hist[tm]    = dvdY_undis_func(hist[tm-1,:])
                ddvddY_undis_hist[tm]    = ddvddY_undis_func(hist[tm-1,:])
                
                for i in range(n_damage):
                    damage_func = damage_func_list[i]
                    gt_dmg[i, tm] = damage_func(hist[tm-1, :])

                for i in range(n_climate):
                    climate_func = climate_func_list[i]
                    pi_c_t[i, tm] = climate_func(hist[tm -1, :])
                    
                mu_L_hist[tm] = mu_L(x_hist[tm], hist[tm-1, :])
                hist[tm,2] = hist[tm-1,2] + mu_L_hist[tm] * dt + sigma_g * logJ_BM[tm]
                hist[tm,1] = hist[tm-1,1] + beta_f * e_hist[tm] * dt + sigma_y * Y_BM[tm]                
                mu_K_hist[tm] = mu_K(i_hist[tm])
                hist[tm,0] = hist[tm-1,0] + mu_K_hist[tm] * dt  + sigma_k * logK_BM[tm]


            if tech_activate==0 and damage_activate ==1: 
                

                gridpoints = (K, Y_long, L)

                i_func = RegularGridInterpolator(gridpoints, postdamage[id_3]["i_star"])
                e_func = RegularGridInterpolator(gridpoints, postdamage[id_3]["e_star"])
                x_func = RegularGridInterpolator(gridpoints, postdamage[id_3]["x_star"])
                tech_func = RegularGridInterpolator(gridpoints, postdamage[id_3]["g_tech"])
                h_func = RegularGridInterpolator(gridpoints, postdamage[id_3]["h"])

                dL = finiteDiff_3D(postdamage[id_3]["v0"], 2,1,hL )
                dY = finiteDiff_3D(postdamage[id_3]["v0"], 1,1,hY )


                dL_func   = RegularGridInterpolator(gridpoints, dL)
                dY_func   = RegularGridInterpolator(gridpoints, dY)
                
                
                dvdL_dis_func   = RegularGridInterpolator(gridpoints, FK_postdamageDistorted[id_3]["dvdL"])
                dvdL_bundis_func   = RegularGridInterpolator(gridpoints, FK_postdamageUndistorted[id_3]["dvdL"])
                

                dvdY_dis_func   = RegularGridInterpolator(gridpoints, FK_postdamage_Y_Distorted[id_3]["dvdY"])
                ddvddY_dis_func   = RegularGridInterpolator(gridpoints, FK_postdamage_Y_Distorted[id_3]["ddvddY"])
                dvdY_undis_func   = RegularGridInterpolator(gridpoints, FK_postdamage_Y_Undistorted[id_3]["dvdY"])
                ddvddY_undis_func   = RegularGridInterpolator(gridpoints, FK_postdamage_Y_Undistorted[id_3]["ddvddY"])

                
                i_hist[tm] = i_func(hist[tm-1,:])
                e_hist[tm] = e_func(hist[tm-1,:])
                x_hist[tm] = x_func(hist[tm-1,:])
                gt_tech[tm] = tech_func(hist[tm-1,:])
                ht[tm] = h_func(hist[tm-1, :])
                
                
                dL_hist[tm] = dL_func(hist[tm-1,:])
                dY_hist[tm] = dY_func(hist[tm-1,:])
                
                dvdL_dis_hist[tm]    = dvdL_dis_func(hist[tm-1,:])
                dvdL_bundis_hist[tm]    = dvdL_bundis_func(hist[tm-1,:])
                

                dvdY_dis_hist[tm]    = dvdY_dis_func(hist[tm-1,:])
                ddvddY_dis_hist[tm]    = ddvddY_dis_func(hist[tm-1,:])
                dvdY_undis_hist[tm]    = dvdY_undis_func(hist[tm-1,:])
                ddvddY_undis_hist[tm]    = ddvddY_undis_func(hist[tm-1,:])


                for i in range(n_climate):
                    climate_func = climate_func_list[i]
                    pi_c_t[i, tm] = climate_func(hist[tm -1, :])
                    
                mu_L_hist[tm] = mu_L(x_hist[tm], hist[tm-1, :])
                hist[tm,2] = hist[tm-1,2] + mu_L_hist[tm] * dt + sigma_g * logJ_BM[tm]
                hist[tm,1] = hist[tm-1,1] + beta_f * e_hist[tm] * dt + sigma_y * Y_BM[tm]                
                mu_K_hist[tm] = mu_K(i_hist[tm])
                hist[tm,0] = hist[tm-1,0] + mu_K_hist[tm] * dt  + sigma_k * logK_BM[tm]


            
            if tech_activate==1: 
                
                
                gridpoints = (K, Y_long, L)

                i_func = RegularGridInterpolator(gridpoints, iipost)
                e_func = RegularGridInterpolator(gridpoints, np.zeros_like(iipost))
                x_func = RegularGridInterpolator(gridpoints, np.zeros_like(iipost))
                tech_func = RegularGridInterpolator(gridpoints, np.zeros_like(iipost))
                h_func = RegularGridInterpolator(gridpoints, np.zeros_like(iipost))


                dL = np.zeros(iipost.shape)
                dY = np.zeros(iipost.shape)


                dL_func   = RegularGridInterpolator(gridpoints, dL)
                dY_func   = RegularGridInterpolator(gridpoints, dY)
                
                
                dvdL_dis_func   = RegularGridInterpolator(gridpoints, np.zeros(iipost.shape))
                dvdL_bundis_func   = RegularGridInterpolator(gridpoints, np.zeros(iipost.shape))
                

                dvdY_dis_func   = RegularGridInterpolator(gridpoints, np.zeros(iipost.shape))
                ddvddY_dis_func   = RegularGridInterpolator(gridpoints, np.zeros(iipost.shape))
                dvdY_undis_func   = RegularGridInterpolator(gridpoints, np.zeros(iipost.shape))
                ddvddY_undis_func   = RegularGridInterpolator(gridpoints, np.zeros(iipost.shape))

                
                # i_hist[tm] = i_func(hist[tm-1,:])
                i_hist[tm] = 0
                e_hist[tm] = 0
                x_hist[tm] = 0
                gt_tech[tm] = 0
                ht[tm] = 0
                
                
                dL_hist[tm] = 0
                dY_hist[tm] = 0
                
                dvdL_dis_hist[tm]    = 0
                dvdL_bundis_hist[tm]    = 0
                

                dvdY_dis_hist[tm]    = 0
                ddvddY_dis_hist[tm]    = 0
                dvdY_undis_hist[tm]    = 0
                ddvddY_undis_hist[tm]    = 0


                for i in range(n_climate):
                    climate_func = climate_func_list[i]
                    pi_c_t[i, tm] = 1/144
                    
                mu_L_hist[tm] = mu_L(x_hist[tm], hist[tm-1, :])
                hist[tm,2] = hist[tm-1,2] + mu_L_hist[tm] * dt + sigma_g * logJ_BM[tm]
                hist[tm,1] = hist[tm-1,1] + beta_f * e_hist[tm] * dt + sigma_y * Y_BM[tm]                
                mu_K_hist[tm] = mu_K(i_hist[tm])
                hist[tm,0] = hist[tm-1,0] + mu_K_hist[tm] * dt  + sigma_k * logK_BM[tm]
                    
                    
            # Ambiguity_mean_undis[tm] = np.mean(theta_ell)
            # Ambiguity_mean_dis[tm] = np.average(theta_ell,weights=pi_c_t[:,tm])
            # Ambiguity_mean_dis_h[tm] = np.average(theta_ell + sigma_y*ht[tm],weights=pi_c_t[:,tm])
            # ME_total_hist[tm] = ME_total_func(hist[tm,:])
            # ME_base_hist[tm] = ME_base_func(hist[tm,:])
            
            # theta_ell_hist[:,tm] = theta_ell + sigma_y*ht[tm]
            # theta_ell_hist2[tm] = np.average(theta_ell, weights= pi_c_t[:,tm]) 

    
    # using Kt instead of K0
    jt = 1 - e_hist/ (alpha * lambda_bar * np.exp(hist[:, 0]))
    jt[jt <= 1e-16] = 1e-16
    LHS = theta * vartheta_bar / lambda_bar * jt**(theta -1)
    MC = delta / (alpha  - i_hist - alpha * vartheta_bar * jt**theta - x_hist)


    RHS_undis  = - (dvdY_undis_hist - (gamma_1 + gamma_2*hist[:,1] )) * ( theta_ell_hist2 + sigma_y*ht) 
    RHS_undis  += -(ddvddY_undis_hist - gamma_2)*sigma_y**2 * e_hist
    RHS_undis  = RHS_undis / MC * np.exp(hist[:,0])
 
    RHS_dis  = - (dvdY_dis_hist - (gamma_1 + gamma_2*hist[:,1] )) *  ( theta_ell_hist2 + sigma_y*ht) 
    RHS_dis  += -(ddvddY_dis_hist - gamma_2)*sigma_y**2 * e_hist
    RHS_dis  = RHS_dis / MC * np.exp(hist[:,0])
       
    scc_hist = LHS * 1000 *(e_hist>0) + 1*(e_hist==0)
    
    scc_dis_hist = RHS_dis * 1000
    scc_undis_hist = RHS_undis * 1000


    # scgw_hist = - dY_hist/ MC * np.exp(hist[:,0]) * 1000
    scgw_hist =  - dY_hist/ MC * np.exp(hist[:,0]) * 1000 * (e_hist>0) + 1*(e_hist==0)
    # MU_RD = dL_hist * psi_0* psi_1 * x_hist**(psi_1-1) * np.exp(psi_1*(hist[:,0]-hist[:,2]))

    # scrd_hist = MU_RD/MC*1000

    # scrd_hist = np.exp(hist[:,2]) * dL_hist / MC * np.exp(hist[:, 0])
    scrd_hist = np.exp(hist[:,2]) * dL_hist / MC * np.exp(hist[:, 0]) * (e_hist>0) + 1*(e_hist==0)
    
    scrd_dis_hist = np.exp(hist[:,2]) * dvdL_dis_hist / MC * np.exp(hist[:, 0])
    scrd_bundis_hist = np.exp(hist[:,2]) * dvdL_bundis_hist / MC * np.exp(hist[:, 0])


    distorted_tech_intensity = np.exp(hist[:, 2]) * gt_tech/varrho

    distorted_tech_prob = 1 - np.exp(- np.cumsum(np.insert(distorted_tech_intensity * dt, 0, 0) ))[:-1]

    true_tech_intensity = np.exp(hist[:, 2]) /varrho
    true_tech_prob = 1 - np.exp(- np.cumsum(np.insert(true_tech_intensity * dt, 0, 0) ))[:-1]
        
    damage_intensity = Damage_Intensity(hist[:, 1])
    distorted_damage_intensity = np.mean(gt_dmg, axis=0) * damage_intensity
    distorted_damage_prob = 1 - np.exp(- np.cumsum(np.insert(distorted_damage_intensity * dt, 0, 0) ))[:-1]
    
    true_damage_intensity =  damage_intensity
    true_damage_prob = 1 - np.exp(- np.cumsum(np.insert(true_damage_intensity * dt, 0, 0) ))[:-1]

    print("techjumprealized={}, year={}".format(techjump_realization.max(),tech_activate_year))
    print("damagejumprealized={}, year={}".format(damagejump_realization.max(),damage_activate_year))
    
    res = dict(
        states= hist, 
        i = i_hist * np.exp(hist[:, 0]), 
        e = e_hist,
        # x = x_hist * np.exp(hist[:, 0]),
        x = x_hist * np.exp(hist[:, 0]),
        scc = scc_hist,
        scc_dis = scc_dis_hist,
        scc_undis = scc_undis_hist,
        scgw = scgw_hist,
        scrd = scrd_hist,
        scrd_dis = scrd_dis_hist,
        scrd_bundis = scrd_bundis_hist,
        gt_tech = gt_tech,
        gt_dmg = gt_dmg,
        ht = ht,
        distorted_damage_prob=distorted_damage_prob,
        distorted_tech_prob=distorted_tech_prob,
        pic_t = pi_c_t,
        # ME_total = ME_total_hist,
        # ME_base = ME_base_hist,
        # ME_total_base = np.log(ME_total_hist / ME_base_hist ) * 100,
        jt = jt,
        LHS = LHS,
        years=years,
        # temp_Lars=temp_Lars,
        true_tech_prob = true_tech_prob,
        true_damage_prob = true_damage_prob,
        # theta_ell_new = theta_ell_hist,
        # Ambiguity_mean_undis = Ambiguity_mean_undis,
        # Ambiguity_mean_dis = Ambiguity_mean_dis,
        # Ambiguity_mean_dis_h = Ambiguity_mean_dis_h,
        damagejump_realization = damagejump_realization,
        techjump_realization = techjump_realization,
        tech_activate_year = tech_activate_year,
        damage_activate_year = damage_activate_year,
        )
    

    return res

def Damage_Intensity(Yt, y_bar_lower=1.5):
    r_1 = 1.5
    r_2 = 2.5
    Intensity = r_1 * (np.exp(r_2 / 2 * (Yt - y_bar_lower)**2) -1) * (Yt > y_bar_lower)
    return Intensity






def model_simulation_generate(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_c_{}_xi_d_{}_xi_g_{}_psi_0_{}_psi_1_{}_varrho_{}_" .format(xi_a,xi_c,xi_d,xi_g,psi_0,psi_1,varrho)



    if scheme == "macroannual":
        xi_a_pre = 100000.
        xi_g_pre = 100000.
        xi_p_pre = 100000.
        File_Name_Suffix_pre = "_xiapre_{}_xig_pre_{}_xippre_{}".format(xi_a_pre, xi_g_pre, xi_p_pre) + "_full_" + scheme + "_" +HJB_solution
        n_bar = n_bar2
        with open(Data_Dir+ File_Dir + "model_tech1_pre_damage"+File_Name_Suffix_pre, "rb") as f:
            model_tech1_pre_damage_ME_base = pickle.load(f)

    elif scheme == "newway":
        xi_a_pre = 100000.
        xi_g_pre = 100000.
        xi_p_pre = 100000.
        File_Name_Suffix_pre = "_xiapre_{}_xig_pre_{}_xippre_{}".format(xi_a_pre, xi_g_pre, xi_p_pre) + "_full_" + scheme + "_" +HJB_solution
        n_bar = n_bar2
        with open(Data_Dir+ File_Dir + "model_tech1_pre_damage"+File_Name_Suffix_pre, "rb") as f:
            model_tech1_pre_damage_ME_base = pickle.load(f)

    elif scheme == "check":
        xi_a_pre = xi_a
        xi_g_pre = xi_g
        xi_p_pre = xi_g
        File_Name_Suffix_pre = "_xiapre_{}_xig_pre_{}_xippre_{}".format(xi_a_pre, xi_g_pre, xi_p_pre) + "_full_" + scheme + "_" +HJB_solution
        n_bar = n_bar1
    
        with open(Data_Dir+ File_Dir + "model_tech1_pre_damage"+File_Name_Suffix_pre, "rb") as f:
            model_tech1_pre_damage_ME_base = pickle.load(f)
    elif scheme== 'direct':
        with open(Data_Dir+ File_Dir + "model_tech1_pre_damage", "rb") as f:
            model_tech1_pre_damage_ME_base = pickle.load(f)
        n_bar = n_bar1
    ME_base = model_tech1_pre_damage_ME_base["ME"]


    v = model_tech1_pre_damage_ME_base["v0"]
    i = model_tech1_pre_damage_ME_base["i_star"]
    e = model_tech1_pre_damage_ME_base["e_star"]
    x = model_tech1_pre_damage_ME_base["x_star"]
    pi_c = model_tech1_pre_damage_ME_base["pi_c"]
    g_tech = model_tech1_pre_damage_ME_base["g_tech"]
    g_damage =  model_tech1_pre_damage_ME_base["g_damage"]
    h =  model_tech1_pre_damage_ME_base["h"]



    with open(Data_Dir + File_Dir+"model_tech1_pre_damage", "rb") as f:
        tech1 = pickle.load(f)
    
    with open(Data_Dir + File_Dir+"model_tech1_post_damage", "rb") as f:
        postdamage_pretech = pickle.load(f)
    
    v_orig = tech1["v0"][:,:n_bar+1,:]
    i_orig = tech1["i_star"][:,:n_bar+1,:]
    e_orig = tech1["e_star"][:,:n_bar+1,:]
    x_orig = tech1["x_star"][:,:n_bar+1,:]
    pi_c_orig = tech1["pi_c"][:,:,:n_bar+1,:]
    g_tech_orig = tech1["g_tech"][:,:n_bar+1,:]
    g_damage_orig =  tech1["g_damage"][:,:,:n_bar+1,:]



    # print("--------------Control Check Start--------------")
    # print("Diff_i={}".format(np.max(abs(i-i_orig))))
    # print("Diff_e={}".format(np.max(abs(e-e_orig))))
    # print("Diff_x={}".format(np.max(abs(x-x_orig))))
    # print("--------------Control Check End--------------")
    
    ME_family = ME_base
    
    model_args = (delta, mu_k, kappa,sigma_k, beta_f, zeta, psi_0, psi_1, sigma_g, theta, lambda_bar, vartheta_bar, varrho)

    # FKPDE = FKPDEsolver(grid = (K, Y_short, L),
    #                     model_args = model_args,
    #                     controls = (i,e,x, g_tech, g_damage, pi_c, v)
    #                   )
    with open(Data_Dir + File_Dir+"FK_Distorted_model_tech1_pre_damage", "rb") as f:
        FK_Dis_tech1 = pickle.load(f)
        
    with open(Data_Dir + File_Dir+"FK_BaselineUndistorted_model_tech1_pre_damage", "rb") as f:
        FK_BasleineUndis_tech1 = pickle.load(f)

    with open(Data_Dir + File_Dir+"FK_Y_Distorted_model_tech1_pre_damage", "rb") as f:
        FK_Y_Dis_tech1 = pickle.load(f)
    
    with open(Data_Dir + File_Dir+"FK_Y_Undistorted_model_tech1_pre_damage", "rb") as f:
        FK_Y_Undis_tech1 = pickle.load(f)
        



    FK_model_tech1_post_damage_Distorted = pickle.load(open(Data_Dir+ File_Dir + "FK_Distorted_model_tech1_post_damage", "rb"))
    FK_model_tech1_post_damage_BaselineUndistorted = pickle.load(open(Data_Dir+ File_Dir + "FK_BaselineUndistorted_model_tech1_post_damage", "rb"))
    FK_model_tech1_post_damage_Y_Distorted = pickle.load(open(Data_Dir+ File_Dir + "FK_Y_Distorted_model_tech1_post_damage", "rb"))
    FK_model_tech1_post_damage_Y_Undistorted = pickle.load(open(Data_Dir+ File_Dir + "FK_Y_Undistorted_model_tech1_post_damage", "rb"))
    
    model_tech2_post_damage = pickle.load(open(Data_Dir+ File_Dir + "model_tech2_post_damage", "rb"))


    
    
    # FK_family = FK_Dis_tech1['dvdL'], FK_Undis_tech1['dvdL']
    # FK_family = FK_Dis_tech1['dvdL'], FK_Undis_tech1['dvdL'], HJB_Dis_tech1['dvdL'], HJB_Undis_tech1['dvdL'], HJB_NewUndis_tech1['dvdL'], FK_Y_Dis_tech1['dvdY'], FK_Y_Undis_tech1['dvdY'], FK_Y_Dis_tech1['ddvddY'], FK_Y_Undis_tech1['ddvddY']
    # FK_family = FK_Dis_tech1['dvdL'], FK_Undis_tech1['dvdL'], HJB_Dis_tech1['dvdL'], HJB_Undis_tech1['dvdL'],  FK_Y_Dis_tech1['dvdY'], FK_Y_Undis_tech1['dvdY'], FK_Y_Dis_tech1['ddvddY'], FK_Y_Undis_tech1['ddvddY']
    # FK_family = FK_Dis_tech1['dvdL'], FK_ClimateUndis_tech1['dvdL'], FK_TechUndis_tech1['dvdL'], FK_DamageUndis_tech1['dvdL'], FK_BasleineUndis_tech1['dvdL'], HJB_Dis_tech1['dvdL'], HJB_Undis_tech1['dvdL'],  FK_Y_Dis_tech1['dvdY'], FK_Y_Undis_tech1['dvdY'], FK_Y_Dis_tech1['ddvddY'], FK_Y_Undis_tech1['ddvddY']
    # FK_family = FK_Dis_tech1['dvdL'], FK_BasleineUndis_tech1['dvdL'], HJB_Dis_tech1['dvdL'], HJB_Undis_tech1['dvdL'],  FK_Y_Dis_tech1['dvdY'], FK_Y_Undis_tech1['dvdY'], FK_Y_Dis_tech1['ddvddY'], FK_Y_Undis_tech1['ddvddY']
    FK_family = FK_Dis_tech1['dvdL'], FK_BasleineUndis_tech1['dvdL'], FK_Y_Dis_tech1['dvdY'], FK_Y_Undis_tech1['dvdY'], FK_Y_Dis_tech1['ddvddY'], FK_Y_Undis_tech1['ddvddY']

    for SimEpisodeNum in range(SimEpisode):
        
        print("simulate={}".format(SimEpochNum*SimEpisode + SimEpisodeNum),flush=True)
        
        res = simulate_pre(grid = (K, Y_short, L), 
                        model_args = model_args, 
                        controls = (i,e,x, g_tech, g_damage, pi_c, h, v),
                        ME = ME_family,
                        FK = FK_family,
                        n_bar = n_bar,
                        postdamage = postdamage_pretech,
                        FK_postdamageDistorted = FK_model_tech1_post_damage_Distorted,
                        FK_postdamageUndistorted = FK_model_tech1_post_damage_BaselineUndistorted,  
                        FK_postdamage_Y_Distorted = FK_model_tech1_post_damage_Y_Distorted,
                        FK_postdamage_Y_Undistorted = FK_model_tech1_post_damage_Y_Undistorted,
                        model_tech2_post_damage = model_tech2_post_damage,
                        Y_long = Y,
                        T0=0, 
                        T=IntPeriod, 
                        dt=timespan,printing=True)

        with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_path_{}".format(IntPeriod, SimEpochNum*SimEpisode + SimEpisodeNum)+ scheme + "_" +HJB_solution, "wb") as f:
            pickle.dump(res,f)

        # print(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_FK_simul_{}_path_{}".format(IntPeriod, SimEpochNum*SimEpisode + SimEpisodeNum)+ scheme + "_" +HJB_solution)
        # print(res.keys())
    return res


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_varrho in range(len(varrhoarr)):
                    
                    res = model_simulation_generate(xiaarr[id_xiag],xicarr[id_xiag],xidarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],varrhoarr[id_varrho])


