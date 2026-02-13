import numpy as np
from time import perf_counter
import scipy.sparse.linalg as sp
from util import *

def _CG(A, b, x0 = None, M_inv = None, tol = 1e-10, normal_eq = False, max_iter = None):
    # if not normal_eq:
    #     if np.min(np.linalg.eigvals(A)) < 0: #check if A is positive definite
    #         raise Exception("Matrix not positive definite", np.min(np.linalg.eigvals(A)))
        
    # check if the function has been given a preconditioner
    is_precond = False if M_inv is None else True


    r_norm = []
    
    # dim of the system
    size = np.size(b)

    b = adj(A) @ b if normal_eq else b
    if normal_eq:
        A_tilde = adj(A) @ A

    # creates the starting point if none where given, else starting point is the zero vector
    x = np.zeros((size, 1), dtype = complex) if x0 is None else x0

    r = b.copy() if x0 is None else b - A@x # first residual
    if np.linalg.norm(r) < tol: # check if initial point works
        return x, r, 0
    
    p = (M_inv @ r).copy() if is_precond else r.copy() # first search direction
    
    rho_prev = inner(r, p)




    max_iter = np.inf if max_iter is None else max_iter
    # iteration counter
    k = 0 
    while k < max_iter:
        Ap_prod = A@p # saves one matrix vector prod
        denorm = inner(Ap_prod, Ap_prod) if normal_eq else inner(p, Ap_prod)
        alpha = rho_prev / denorm

        x += alpha * p # next point

        r -= alpha * A_tilde@p if normal_eq else alpha * Ap_prod # next residual
        r_norm.append(np.linalg.norm(r))
        if r_norm[-1] < tol: # stopping criteria
            break
        
        # calc to make the next search direction conjugate (A-orthogonal) to the previous
        Mr = M_inv @ r if is_precond else r
        rho_next = inner(r, Mr)
        
        beta = rho_next / rho_prev
        p = Mr + beta * p # next search direction

        rho_prev = rho_next

        if (k+1) % 5000 == 0:
            print(k, np.linalg.norm(r), end="\r")

        if size * 100 == k:
            break
        k += 1
    return x, r_norm, k

def CG(A, b, x0 = None, M_inv = None, tol = 1e-10, verbose = False, normal_eq = False, max_iter = None):
    """
    function to run the CG function, 
    with possibility of printing extra info to the terminal with the verbose bool
    """
    if verbose:
        print("\nMethod: CG\nSystem dim:", np.size(b))
        print(f"Is Preconditioned: {'False' if M_inv is None else 'True'}")
        start = perf_counter()
    x, r_norm, k = _CG(A=A, b=b, x0=x0, M_inv=M_inv, tol=tol, normal_eq=normal_eq, max_iter= max_iter)
    if verbose:
        print("Run time:", perf_counter() - start)
        print("Iter count:", k, "\nResidual norm:",  r_norm[-1])
        print("Sol norm:", np.linalg.norm(A@x - b),"\nAll close:", "true!!!!!!!!!!!!!" if np.allclose(A@x, b) else "False", "\n")
    return x, r_norm, k


def _BiCGSTAB(A, b, x0 = None, M_inv = None, tol = 1e-10, max_iter = None):

    # check if the function has been given a preconditioner
    is_precond = False if M_inv is None else True

    
    
    # dim of the system
    size = np.size(b)

    # list to norms of residuals for comparison
    r_norm = []
    s_norm = []

    # creates the starting point if none where given, else starting point is the zero vector
    x = np.zeros((size, 1), dtype = complex) if x0 is None else x0
    
    r = b - A@x # first residual
    
    r_tilde = r.copy()
    rho_prev = inner(r_tilde, r)
    p = r.copy()

    max_iter = np.inf if max_iter is None else max_iter

    k = 0 # iteration counter
    while k < max_iter:

        p_hat = M_inv@p if is_precond else p

        Ap_prod = A@p_hat

        temp = inner(r_tilde, Ap_prod)
        if temp == 0:
            flag = 2
            break
        alpha = rho_prev / temp
        x = x + alpha * p_hat # h
        r = r - alpha * Ap_prod # residual 2 (s)
        s_norm.append(np.linalg.norm(r))
        if s_norm[-1] < tol: # stopping criteria
            # x = x
            flag = 0
            break

        s_hat = M_inv @ r if is_precond else r

        t = A @ s_hat

        omega = inner(t, r) / inner(t, t)

        x = x + omega * s_hat
        r = r - omega * t # residual 1 (r)
        r_norm.append(np.linalg.norm(r))
        if r_norm[-1] < tol: # stopping criteria
            flag = 0
            break
            # return x, r_norm, k

        rho_next = inner(r_tilde, r)
        if np.abs(rho_next) == 0 or omega == 0:
            flag = 2
            # print(rho_next)
            break
        beta = (rho_next / rho_prev) * (alpha / omega)
        p = r + beta * (p - omega * Ap_prod)

        rho_prev = rho_next

        k += 1
    else:
        flag = 1
    return x, (r_norm, s_norm), k, flag

def BiCGSTAB(A, b, x0 = None, M_inv = None, tol = 1e-10, verbose = False, max_iter = None):
    """function to run the BiCGSTAB function, 
    with possibility of printing extra info to the terminal with the verbose bool"""
    if verbose:
        print("Method: BiCGSTAB\nSystem dim:", np.size(b))
        print(f"Is Preconditioned: {'False' if M_inv is None else 'True'}")
        start = perf_counter()
    x, r_norm, k, flag = _BiCGSTAB(A, b, x0, M_inv = M_inv, tol = tol, max_iter = max_iter)
    if verbose:
        print("Run time:", perf_counter() - start)
        print("Iter count:", k, "\nResidual norm:",  r_norm[0][-1],"\nResidual norm:",  r_norm[1][-1])
        print("Sol norm:", np.linalg.norm(A@x- b),"\nAll close:", np.allclose(A@x, b))
        print("Flag:", flag, "\n")
    return x, r_norm, k, flag


