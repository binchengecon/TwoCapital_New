"""
file for post HJB with k and y
"""
import os
import sys
sys.path.append("../src/")
import numpy as np
import pandas as pd
# from numba import njit
from supportfunctions import finiteDiff
import SolveLinSys



"""
solver.py
For 3D abatement solver
"""
import os
import sys
sys.path.append("../../src/")
from src.Utility import finiteDiff_3D
import SolveLinSys
import numpy as np
import petsc4py
from petsc4py import PETSc
import petsclinearsystem
import time
from datetime import datetime


def pde_one_interation(ksp, petsc_mat, X1_mat_1d, X2_mat_1d, X3_mat_1d, lowerLims, upperLims, dVec, increVec, v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, tol, epsilon):

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
    petsclinearsystem.formLinearSystem(X1_mat_1d, X2_mat_1d, X3_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    b = v0_1d + D_1d * epsilon
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
    return out_comp,end_ksp,bpoint1

def _FOC_update(v0, steps= (), states = (), args=(), controls=(), fraction=0.5):

    hX1, hX2, hX3 = steps
    K_mat, Y_mat, L_mat = states
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, xi_a, xi_k, xi_c, xi_j, xi_d, xi_g, rho, varrho = args

    i_star = controls
    # First order derivative
    dX1  = finiteDiff_3D(v0,0,1,hX1)
    dX1[dX1 <= 1e-16] = 1e-16
    dK = dX1
    dX2  = finiteDiff_3D(v0,1,1,hX2)
    dY = dX2
    dX3  = finiteDiff_3D(v0,2,1,hX3)
    dX3[dX3 <= 1e-16] = 1e-16
    dL = dX3
    ######## second order
    ddX1 = finiteDiff_3D(v0,0,2,hX1)
    ddX2 = finiteDiff_3D(v0,1,2,hX2)
    ddY = ddX2
    ddX3 = finiteDiff_3D(v0,2,2,hX3)


    temp = delta * ( (alpha - i_star) * np.exp(K_mat) / np.exp(v0)    )**(-rho) * (np.exp(K_mat)/np.exp(v0))

    i_new = (1- temp/dK)/kappa



    ii = i_new * fraction + i_star * (1 - fraction)


    
    
    h_k = -1/xi_k *sigma_k * dK 
    
    h_k[h_k>=-1e-16]=-1e-16
    

    consumption = alpha - ii 
    
    temp_recursive = (consumption *np.exp(K_mat)/np.exp(v0))**(1-rho) - 1
    
    A   = np.zeros_like(K_mat)
    
    B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
    B_1 += sigma_k*h_k
    B_2 = np.zeros_like(K_mat)
    B_3 = np.zeros_like(K_mat)
    

    C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
    C_2 = np.zeros_like(K_mat)
    C_3 = np.zeros_like(K_mat)
    
    D = delta / (1-rho) *temp_recursive
     
    D += 1/2 * xi_k * h_k**2

    
    
    return A, B_1, B_2, B_3, C_1, C_2, C_3, D, dX1, ddX1, ii, h_k


def hjb_post_tech(
        state_grid=(), model_args=(), V_post_damage=None, 
        tol=1e-8, epsilon=0.1, fraction=0.5, max_iter=10000,
        v0=None,
        smart_guess=None,
        ):

    now = datetime.now()
    current_time = now.strftime("%d-%H:%M")
    K, Y, L = state_grid

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_k, xi_c, xi_j, xi_d, xi_g, rho, varrho = model_args


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
    #### Model type


    # Initial setup of HJB
    FC_Err   = 1
    epoch    = 0

    if v0 is None:
        # v0 = K_mat + L_mat - np.average(pi_c_o, axis=0) * Y_mat
        v0 = K_mat
        
        
    i_star = np.zeros(K_mat.shape)

    if smart_guess:
        v0     = smart_guess["v0"]
        i_star = smart_guess["i_star"]

            
    

    dVec = np.array([hX1, hX2, hX3])
    increVec = np.array([1, nX1, nX1 * nX2],dtype=np.int32)

    # FOC_args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, V_post_tech, dG, ddG, xi_a, xi_g )

    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nX1 * nX2 * nX3, nX1 * nX2 * nX3])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()

    # Enter the optimization
    while FC_Err > tol and epoch < max_iter:
        
        FOC_args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, xi_a, xi_k, xi_c, xi_j, xi_d, xi_g, rho, varrho)

        start_ep = time.time()
        A, B_1, B_2, B_3, C_1, C_2, C_3, D, dX1, ddX1, ii, h_k = _FOC_update(v0, steps= (hX1, hX2, hX3), states = (K_mat, Y_mat, L_mat), args=FOC_args, controls=(i_star), fraction=fraction)



        out_comp,end_ksp, bpoint1 = pde_one_interation(
                ksp,
                petsc_mat,X1_mat_1d, X2_mat_1d, X3_mat_1d, 
                lowerLims, upperLims, dVec, increVec,
                v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, 1e-13, epsilon)
        # if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
        
        PDE_rhs = A * v0 + B_1 * dX1 + C_1 * ddX1  + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        if FC_Err < 1.2*tol:
            
            if epoch%100==0:
                print("-----------------------------------")
                print("---------Epoch {}---------------".format(epoch))
                print("-----------------------------------")
                print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
                # print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
                # print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
                # print("min h: {},\t max h: {}\t".format(h.min(), h.max()))
                print("min hk: {},\t max hk: {}\t".format(h_k.min(), h_k.max()))
                # print("min hj: {},\t max hj: {}\t".format(h_j.min(), h_j.max()))
                print("petsc total: {:.3f}s, Residual Norm is {:g}".format((end_ksp - bpoint1),ksp.getResidualNorm()))
                print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
                print("Epoch time: {:.4f}".format(time.time() - start_ep))
        elif epoch%10000==0:
            
            print("-----------------------------------")
            print("---------Epoch {}---------------".format(epoch))
            print("-----------------------------------")
            print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
            # print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
            # print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
            # print("min h: {},\t max h: {}\t".format(h.min(), h.max()))
            print("min hk: {},\t max hk: {}\t".format(h_k.min(), h_k.max()))
            # print("min hj: {},\t max hj: {}\t".format(h_j.min(), h_j.max()))
            print("petsc total: {:.3f}s, Residual Norm is {:g}".format((end_ksp - bpoint1),ksp.getResidualNorm()))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
            print("Epoch time: {:.4f}".format(time.time() - start_ep))
        

        v0     = out_comp
        i_star = ii
        # e_star = ee
        # x_star = xx
        epoch += 1

    dX1  = finiteDiff_3D(v0,0,1,hX1)
    dX1[dX1 <= 1e-16] = 1e-16
    dK = dX1
    ######## second order
    ddX1 = finiteDiff_3D(v0,0,2,hX1)


    
    res = {
            "v0"    : v0,
            "i_star": i_star,
            "h_k": h_k,
            "FC_Err": FC_Err,
            }

    return res



