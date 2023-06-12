import os
import sys
sys.path.append("../../src/")
from src.Utility import finiteDiff_3D
import SolveLinSys
import numpy as np
import petsc4py
from petsc4py import PETSc
import petsclinearsystem
import petsclinearsystem_new
# import time
# from datetime import datetime
# import matplotlib.pyplot as plt


def PDESolver(stateSpace, A, B_r, B_f, B_k, C_rr, C_ff, C_kk, D, v0, ε = 1, tol = -10, smartguess = False, solverType = 'False Transient'):

    if solverType == 'False Transient':
        A = A.reshape(-1,1,order = 'F')
        B = np.hstack([B_r.reshape(-1,1,order = 'F'),B_f.reshape(-1,1,order = 'F'),B_k.reshape(-1,1,order = 'F')])
        C = np.hstack([C_rr.reshape(-1,1,order = 'F'), C_ff.reshape(-1,1,order = 'F'), C_kk.reshape(-1,1,order = 'F')])
        D = D.reshape(-1,1,order = 'F')
        v0 = v0.reshape(-1,1,order = 'F')
        out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

        return out

    elif solverType == 'Feyman Kac':
        
        if smartguess:
            iters = 1
        else:
            iters = 4000000
            
        A = A.reshape(-1, 1, order='F')
        B = np.hstack([B_r.reshape(-1, 1, order='F'), B_f.reshape(-1, 1, order='F'), B_k.reshape(-1, 1, order='F')])
        C = np.hstack([C_rr.reshape(-1, 1, order='F'), C_ff.reshape(-1, 1, order='F'), C_kk.reshape(-1, 1, order='F')])
        D = D.reshape(-1, 1, order='F')
        v0 = v0.reshape(-1, 1, order='F')
        out = SolveLinSys.solveFK(stateSpace, A, B, C, D, v0, iters)

        return out



def fk_pre_tech(
        state_grid=(), 
        model_args=(), 
        control=(),
        VF=(),
        ):

    K, Y, L = state_grid
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_k, xi_c, xi_j, xi_d, xi_g,rho, varrho = model_args
    

    epsilon=0.005
    hL = L[1]-L[0]
    ######## post jump, 3 states
    (X1_mat, X2_mat, X3_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
    stateSpace = np.hstack([X1_mat.reshape(-1,1,order = 'F'), X2_mat.reshape(-1,1,order = 'F'), X3_mat.reshape(-1, 1, order='F')])
    
    K_mat = X1_mat
    Y_mat = X2_mat
    L_mat = X3_mat
    # For PETSc
    X1_mat_1d = X1_mat.ravel(order='F')
    X2_mat_1d = X2_mat.ravel(order='F')
    X3_mat_1d = X3_mat.ravel(order='F')
    
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        # v_i = V_post_damage
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        i,e,x,pi_c,g_tech,g_damage,h, h_k, h_j = controls
        
        Phi_II, Phi, F_II, F_m = VF


        dv0dL = finiteDiff_3D(Phi, 2, 1, hL)       
        
        c = alpha-i-x-alpha*vartheta_bar*(1-e/(lambda_bar * alpha * np.exp(K_mat)))**theta
        C = c*np.exp(K_mat)

        A = -delta * (C/np.exp(Phi))**(1-rho) - psi_0 * psi_1 * (x * np.exp(K_mat-L_mat) )**psi_1 - np.exp(L_mat - np.log(varrho)) * g_tech - Intensity*np.sum(pi_d_o*g_damage,axis=0)
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_1 += sigma_k*h_k
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_2 += sigma_y * h * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 += sigma_g*h_j

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)

        D = np.exp(L_mat - np.log(varrho)) * g_tech * (Phi_II - Phi)  + np.exp(L_mat - np.log(varrho)) * g_tech * F_II + Intensity * np.sum(pi_d_o*g_damage* F_m,axis=0)  
        D += xi_g * np.exp(L_mat - np.log(varrho)) * (1-g_tech +g_tech *np.log(g_tech))


        out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, dv0dL, epsilon, solverType="Feyman Kac")
        F  = out[2].reshape(dv0dL.shape, order="F")
            
        dvdL_orig = finiteDiff_3D(Phi, 2, 1, hL)    
        print("sanity check: {}".format(np.max(abs(F-dvdL_orig))))

        
    else:
        model = "Post damage"
        i,e,x,pi_c,g_tech,h, h_k, h_j = control
        
        Phi_m_II, Phi_m, F_m_II = VF

        dv0dL = finiteDiff_3D(Phi_m, 2, 1, hL)       

        c = alpha-i-x-alpha*vartheta_bar*(1-e/(lambda_bar * alpha * np.exp(K_mat)))**theta
        C = c*np.exp(K_mat)

        A = -delta *  (C/np.exp(Phi_m))**(1-rho) - psi_0 * psi_1 * (x * np.exp(K_mat-L_mat) )**psi_1 - np.exp(L_mat - np.log(varrho)) * g_tech
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2 
        B_1 += sigma_k*h_k
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_2 += sigma_y * h * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 += sigma_g * h_j

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)

        D = np.exp(L_mat - np.log(varrho)) * g_tech * (Phi_m_II - Phi_m)+ np.exp(L_mat - np.log(varrho)) * g_tech * F_m_II
        D += xi_g * np.exp(L_mat - np.log(varrho)) * (1-g_tech +g_tech *np.log(g_tech))


        out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, dv0dL, epsilon, solverType="Feyman Kac")
        F_m  = out[2].reshape(dv0dL.shape, order="F")
            
        dvdL_orig = finiteDiff_3D(Phi_m, 2, 1, hL)    
        
        print("sanity check: {}".format(np.max(abs(F_m-dvdL_orig))))
        
    res = {
            "v0"    : F_m,
            "i_star": i,
            "e_star": e,
            "x_star": x,
            "pi_c"  : pi_c,
            "g_tech": g_tech,
            "h": h,
            "h_k": h_k,
            "h_j": h_j,
            "dvdL": F_m,
            }
    if model == "Pre damage":
        res = {
                "v0"    : F_m,
                "i_star": i,
                "e_star": e,
                "x_star": x,
                "pi_c"  : pi_c,
                "g_tech": g_tech,
                "g_damage": g_damage,
                "h": h,
                "h_k": h_k,
                "h_j": h_j,
                "dvdL": F_m,
                }
    return res
