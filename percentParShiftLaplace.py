import linearSolver as ls
import Precondition as prec
import numpy as np
import util
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import genLaplace
import multiprocessing as mp
from datetime import datetime
from itertools import product
from sys import argv
from configparser import ConfigParser

ebutton = mp.Event()

def runBiCGStab(A,b,M_inv, config:ConfigParser, stop_futher = True):
    np.seterr(all = 'raise')

    try:
        if ebutton.is_set() and stop_futher:
            return -1, 4
        
        precond_type = config.get('Precondition', 'type')
        if M_inv is None:
            pass
        elif precond_type.lower() == 'par_shift':
            M_inv = sparse.eye(config.getint('Data', 'dim')) + M_inv
        elif precond_type.lower() == 'par_shift_jacobi':
            M_inv = prec.Jacobi(A) + M_inv

        if b is None:
            b = np.ones((config.getint('Data', 'dim'),1 ))


        _,_,k,flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = config.getfloat('Solver', 'tol'), max_iter=config.getintOrNone('Solver', 'max_iteration'))


        if flag == 2:
            ebutton.set()
        return k, flag
    
    except:
        ebutton.set()
        return -1, 3






def laplaceDataML(data, file, config: ConfigParser):

    # precond_function = config.getprecondFunction('Precondition', 'type')

    b = np.ones((config.getint('Data', 'dim'), 1))


    rng = np.random.default_rng(config.getint('Learn', 'seed'))
    coef_list = rng.normal(scale = 0.05, size=config.getint('Precondition', 'num_coef'))
    # coef_list_initial = coef_list.copy()
    found_initial = False

    best_median_k = np.inf
    best_mean_k = np.inf


    count = 0
    last_improvement = (0, 0)

    # M_inv = prec.parShift(config.getint('Data', 'dim'), coef_list)


    for ii in range(config.getint('Learn', 'iteration_count')):
        index = rng.choice(config.getint('Precondition', 'num_coef')) # choose index to change
        change = rng.uniform(low = -config.getfloat('Learn', 'step_range'), high = config.getfloat('Learn', 'step_range'))

        found_better = False
        for scale, pm in product(config.getfloatList('Learn', 'scale'), [1, -1]):
            count += 1
            print(best_mean_k, ii, count, last_improvement, end = '\r')

            temp_coef = coef_list.copy()
            temp_coef[index] += change * pm * scale
            # M_inv = prec.parShift(config.getint('Data', 'dim'), temp_coef)
            M_inv = prec.parShiftOff(config.getint('Data', 'dim'), temp_coef)
            # M_inv = precond_function(config.getint('Data', 'dim'), temp_coef)

            pool = mp.Pool()
            new_k_list, flag_list = zip(*pool.starmap(runBiCGStab, [(A, b, M_inv, config) for A in data]))


            new_median = np.median(new_k_list)
            new_mean = np.mean(new_k_list)

            if found_initial:
                sign = util.betterWorse(new_k_list, k_list)
            else:
                sign = {1: config.getint('Data', 'amount'), -1: 0}
            

            if np.sum(flag_list) < 1: # if all converged  
                if new_median <= best_median_k and config.get('Learn', 'method') == 'median': # is it better or as good
                    found_better = True
                elif sign[1] > sign[-1] and config.get('Learn', 'method') == 'sign': # is it better or as good
                    found_better = True
                elif new_mean <= best_mean_k and config.get('Learn', 'method')== 'mean': # is it better or as good
                    found_better = True

            if found_better:
                if not found_initial:
                    coef_list_initial = temp_coef.copy()
                    found_initial = True
                coef_list = temp_coef.copy()
                best_median_k = new_median
                best_mean_k = new_mean
                last_improvement = (ii, count)
                file.write(f'({ii}, {count}): {util.statStr(new_k_list)}, B: {sign[1]}, W: {sign[-1]}, \n\t{new_k_list}\n')
                k_list = new_k_list
                break

    pool.close()
    pool.join()

    print()
    return coef_list, coef_list_initial







if __name__ == '__main__':

    if len(argv) == 1:
        file_str = 'test_config.ini'
    else:
        file_str = argv[1]


    config = util.getConfig(file_str)




    rng_data = np.random.default_rng(config.getint('Learn', 'seed'))
    training_data = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)


    

    with open(f'Data/shift_laplace_{config.get("Learn", "method")}.txt', mode = 'a') as txt_file:
        txt_file.write(f'\n{datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\n')
        txt_file.write(f'Config file: {file_str}\n')
        
        
        for key in config.sections():
            txt_file.write(f'{key}:\n')
            for op in config.options(key):
                txt_file.write(f'\t{op}: {config.get(key, op)}\n')


            

        txt_file.write(f'\n')
        txt_file.write('Train:\n')
        coef_list, coef_list_initial = laplaceDataML(training_data, file = txt_file, config = config)

        txt_file.write('\n')


        test_data = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)

        txt_file.write('Test:\n')

        pool = mp.Pool()
        non_precond_k_list, non_pre_flag_list = zip(*pool.starmap(runBiCGStab,[(A, None, None, config, False) for A in test_data]))
        txt_file.write(f'No precond: {util.statStr(non_precond_k_list)}, flag: {np.sum(non_pre_flag_list)} \n\t{non_precond_k_list}\n\n')


        pool = mp.Pool()
        M_inv = prec.parShift(config.getint('Data', 'dim'), coef_list)
        final_k_list, final_flag_list = zip(*pool.starmap(runBiCGStab, [(A, None, M_inv, config, False) for A in test_data]))

        sign = util.betterWorse(final_k_list, non_precond_k_list)

        txt_file.write(f'Last: {util.statStr(final_k_list)}, B: {sign[1]}, W: {sign[-1]}, flag: {np.sum(final_flag_list)} \n\t{final_k_list}\n')

            
            
        txt_file.write(f'\n{coef_list}\n')


        pool.close()
        pool.join()

    n1 ,_,_ = plt.hist([non_precond_k_list,final_k_list],bins=40, alpha = 1, label=['Non', 'precond', 'jacobi'], color=['c', 'm', 'y'])
    plt.legend()
    plt.show()
