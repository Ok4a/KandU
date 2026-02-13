import linearSolver as ls
import Precondition as prec
import numpy as np
import util
from scipy.io import mmread
import matplotlib.pyplot as plt
import time
import genLaplace
from multiprocessing import Pool
from datetime import datetime
from itertools import product
from sys import argv
from configparser import ConfigParser

def runBiCGStab(A,b,M_inv, config:ConfigParser):
    if b is None:
        b = np.ones((config.getint('Data', 'dim'),1 ))
    _,_,k,flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = config.getfloat('Solver', 'tol'), max_iter=config.getintOrNone('Solver', 'max_iteration'))
    return k, flag





def laplaceDataML(data, file, config: ConfigParser):

    # step_range = config['step_range']
    # iteration_count = config['train_iteration_count']
    # num_coef = config['num_coef']
    # method = config['learn_method']
    # seed = config['learn_seed']
    # size = config['dim']

    b = np.ones((config.getint('Data', 'dim'), 1))


    rng = np.random.default_rng(config.getint('Learn', 'seed'))
    coef_list = rng.normal(scale = 0.05, size=config.getint('Precondition', 'num_coef'))
    # coef_list_initial = coef_list.copy()
    found_initial = False

    best_median_k = np.inf
    best_mean_k = np.inf


    count = 0
    last_improvement = (0, 0)

    M_inv = prec.parShift(config.getint('Data', 'dim'), coef_list)

    with Pool() as pool:

        for ii in range(config.getint('Learn', 'iteration_count')):
            index = rng.choice(config.getint('Precondition', 'num_coef')) # choose index to change
            change = rng.uniform(low = -config.getfloat('Learn', 'step_range'), high = config.getfloat('Learn', 'step_range'))

            found_better = False
            for scale, pm in product(config.getfloatList('Learn', 'scale'), [1, -1]):
                count += 1
                print(best_mean_k, ii, count, last_improvement, end = '\r')

                temp_coef = coef_list.copy()
                temp_coef[index] += change * pm * scale
                M_inv = prec.parShift(config.getint('Data', 'dim'), temp_coef)

                new_k_list, flag_list = zip(*pool.starmap(runBiCGStab, [(A, b, M_inv, config) for A in data]))


                new_median = np.median(new_k_list)
                new_mean = np.mean(new_k_list)

                # diff = np.sum(np.array(k_list) - np.array(new_k_list))
                if found_initial:
                    # sign = np.sum(np.sign(np.array(k_list) - np.array(new_k_list)))
                    sign = betterWorse(new_k_list, k_list)
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
                    file.write(f'({ii}, {count}): {statStr(new_k_list)}, B: {sign[1]}, W: {sign[-1]}, \n\t{new_k_list}\n')
                    k_list = new_k_list
                    break

    pool.close()

    print()
    return coef_list, coef_list_initial





def statStr(k_list):
    mean = np.mean(k_list, axis=0)
    median = np.median(k_list, axis=0)
    std = np.std(k_list, axis=0)
    prt1 = np.quantile(k_list,q=(1-0.68)/2, axis=0)
    prt2 = np.quantile(k_list,q=1-(1-0.68)/2, axis=0)

    return f'Mean: {np.round(mean,3)}, median: {np.round(median,3)}, std: {np.round(std,3)}, percent: {np.round((prt1+prt2)/2,3)}'

def betterWorse(new, old):
    sign = np.sign(np.array(old) - np.array(new))
    unique, count = np.unique(sign, return_counts = True)
    counts = dict(zip(unique, count))
    for ii in [-1,0,1]:
        if ii not in counts.keys():
            counts[ii] = 0
    return counts

def betterWorseStr(new, old):
    sign = betterWorse(new,old)
    return f'B: {sign[1]}, W: {sign[-1]}'





if __name__ == '__main__':

    if len(argv) == 1:
        file_str = 'test_config.ini'
    else:
        file_str = argv[1]


    config = util.getConfig(file_str)



    rng_data = np.random.default_rng(config.getint('Learn', 'seed'))
    training_data = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)




    with open(f'testData/shift_laplace_{config.get("Learn", "method")}.txt', mode = 'a') as txt_file:
        txt_file.write(f'\n{datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\n')
        txt_file.write(f'Config file: {file_str}')
        
        # for key in config:
        #     txt_file.write(f'{key}: {config[key]}\n')

        txt_file.write(f'\n')
        txt_file.write('Train:\n')
        coef_list, coef_list_initial = laplaceDataML(training_data, file = txt_file, config = config)

        txt_file.write('\n')


        test_data = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)

        txt_file.write('Test:\n')
        with Pool() as pool:
            non_precond_k_list, non_pre_flag_list = zip(*pool.starmap(runBiCGStab,[(A, None ,None, config) for A in test_data]))
            txt_file.write(f'No precond: {statStr(non_precond_k_list)}, flag: {np.sum(non_pre_flag_list)} \n\t{non_precond_k_list}\n\n')

            # M_inv = prec.parShift(config.getint('Data', 'dim'), coef_list_initial)
            # initial_k_list, initial_flag_list = zip(*pool.starmap(runBiCGStab, [(A, None, M_inv,config) for A in test_data]))
            # txt_file.write(f'Initial: {statStr(initial_k_list)}, flag: {np.sum(initial_flag_list)} \n\t{initial_k_list}\n')

            M_inv = prec.parShift(config.getint('Data', 'dim'), coef_list)
            final_k_list, final_flag_list = zip(*pool.starmap(runBiCGStab, [(A, None, M_inv, config) for A in test_data]))

            sign = betterWorse(final_k_list, non_precond_k_list)

            txt_file.write(f'Last: {statStr(final_k_list)}, B: {sign[1]}, W: {sign[-1]}, flag: {np.sum(final_flag_list)} \n\t{final_k_list}\n')

            
            
            
            # diffVInitial {np.sum(np.array(initial_k_list) - np.array(final_k_list))}, diffVNon {np.sum(np.array(non_precond_k_list) - np.array(final_k_list))},
        txt_file.write(f'\n{coef_list}\n')
        pool.close()

    plt.hist(non_precond_k_list, alpha=0.5, label='Non', bins=20)
    plt.hist(final_k_list, alpha=0.5, label='Final', bins=20)
    plt.legend()
    plt.show()
