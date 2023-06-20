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
import time
# from datetime import datetime
# import matplotlib.pyplot as plt

def pde_one_interation_noFT(ksp, petsc_mat, X1_mat_1d, X2_mat_1d, X3_mat_1d, lowerLims, upperLims, dVec, increVec, v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, tol, epsilon):

    bpoint1 = time.time()
    A_1d   = A.ravel(order = 'F')
    C_1_1d = C_1.ravel(order = 'F')
    C_2_1d = C_2.ravel(order = 'F')
    C_3_1d = C_3.ravel(order = 'F')
    B_1_1d = B_1.ravel(order = 'F')
    B_2_1d = B_2.ravel(order = 'F')
    B_3_1d = B_3.ravel(order = 'F')
    D_1d   = D.ravel(order = 'F')
    v0_1d  = v0.ravel(order = 'F')
    petsclinearsystem_new.formLinearSystem_noFT(X1_mat_1d, X2_mat_1d, X3_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    b = D_1d 
    petsc_rhs = PETSc.Vec().createWithArray(b)
    x = petsc_mat.createVecRight()


    # create linear solver
    start_ksp = time.time()
    ksp.setOperators(petsc_mat)
    ksp.setTolerances(rtol=tol)
    ksp.solve(petsc_rhs, x)
    petsc_rhs.destroy()
    x.destroy()
    out_comp = np.array(ksp.getSolution()).reshape(A.shape,order = "F")
    end_ksp = time.time()
    num_iter = ksp.getIterationNumber()
    # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
    return out_comp,end_ksp,bpoint1



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
            iters = 40000000
            
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
        print(model)
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        # v_i = V_post_damage
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        i,e,x,pi_c,g_tech,g_damage,h, h_k, h_j = control
        
        Phi_II, Phi, F_II, F_m = VF


        dv0dL = finiteDiff_3D(Phi, 2, 1, hL)       
        
        j = alpha*vartheta_bar*(1-e/(lambda_bar * alpha * np.exp(K_mat)))**theta
        j[j<=1e-16] = 1e-16
        c = alpha-i-x-j
        c[c<=1e-16] = 1e-16
        C = c*np.exp(K_mat)
        print(c.min(),c.max(),flush=True)

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
        print(F.shape)
        dvdL_orig = finiteDiff_3D(Phi, 2, 1, hL)    
        print("sanity check: {},{}".format(np.max(abs(F-dvdL_orig)),np.min(abs(F-dvdL_orig))))

        
    else:
        model = "Post damage"
        i,e,x,pi_c,g_tech,h, h_k, h_j = control
        
        Phi_m_II, Phi_m, F_m_II = VF

        dv0dL = finiteDiff_3D(Phi_m, 2, 1, hL)       

        c = alpha-i-x-alpha*vartheta_bar*(1-e/(lambda_bar * alpha * np.exp(K_mat)))**theta
        C = c*np.exp(K_mat)
        print(c.min(),c.max(),flush=True)

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
                "v0"    : F,
                "i_star": i,
                "e_star": e,
                "x_star": x,
                "pi_c"  : pi_c,
                "g_tech": g_tech,
                "g_damage": g_damage,
                "h": h,
                "h_k": h_k,
                "h_j": h_j,
                "dvdL": F,
                }
    return res



def fk_pre_tech_petsc(
        state_grid=(), 
        model_args=(), 
        control=(),
        VF=(),
        ):

    K, Y, L = state_grid
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_k, xi_c, xi_j, xi_d, xi_g,rho, varrho = model_args
    
    epsilon=0.005
    tol=1e-10
    K_min, K_max, Y_min, Y_max, L_min, L_max = K.min(), K.max(), Y.min(), Y.max(), L.min(), L.max()
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    
    ######## post jump, 3 states
    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)
    
    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)
    
    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nK * nY * nL, nK * nY * nL])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()
    
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
        i,e,x,pi_c,g_tech,g_damage,h, h_k, h_j = control
        
        Phi_II, Phi, F_II, F_m = VF


        dv0dL = finiteDiff_3D(Phi, 2, 1, hL)       
        
        j = alpha*vartheta_bar*(1-e/(lambda_bar * alpha * np.exp(K_mat)))**theta
        # j[j<=1e-16] = 1e-16
        c = alpha-i-x-j
        # c[c<=1e-16] = 1e-16
        C = c*np.exp(K_mat)
        print(c.min(),c.max(),flush=True)

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

        out_comp,end_ksp, bpoint1 = pde_one_interation_noFT(
                ksp,
                petsc_mat,K_mat_1d, Y_mat_1d, L_mat_1d, 
                lowerLims, upperLims, dVec, increVec,
                Phi, A, B_1, B_2, B_3, C_1, C_2, C_3, D, 1e-13, epsilon)
        

        F = out_comp
        dvdL_orig = finiteDiff_3D(Phi, 2, 1, hL)    
        print("sanity check: {}".format(np.max(abs(F-dvdL_orig))))

        
    else:
        model = "Post damage"
        print(model)
        i,e,x,pi_c,g_tech,h, h_k, h_j = control
        
        Phi_m_II, Phi_m, F_m_II = VF

        dv0dL = finiteDiff_3D(Phi_m, 2, 1, hL)       

        c = alpha-i-x-alpha*vartheta_bar*(1-e/(lambda_bar * alpha * np.exp(K_mat)))**theta
        C = c*np.exp(K_mat)
        print(c.min(),c.max(),flush=True)

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


        out_comp,end_ksp, bpoint1 = pde_one_interation_noFT(
                ksp,
                petsc_mat,K_mat_1d, Y_mat_1d, L_mat_1d, 
                lowerLims, upperLims, dVec, increVec,
                Phi_m, A, B_1, B_2, B_3, C_1, C_2, C_3, D, 1e-13, epsilon)
        
        F_m = out_comp

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
                "v0"    : F,
                "i_star": i,
                "e_star": e,
                "x_star": x,
                "pi_c"  : pi_c,
                "g_tech": g_tech,
                "g_damage": g_damage,
                "h": h,
                "h_k": h_k,
                "h_j": h_j,
                "dvdL": F,
                }
    return res
