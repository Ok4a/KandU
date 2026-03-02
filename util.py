import numpy as np
from configparser import ConfigParser
import Precondition as prec
rng = np.random.default_rng()

def inner(x,y):
    return np.conj(x.T) @ y

def adj(x):
    return np.conj(x.T)

def randAb(size, l = 0, u = 1, normal = False):
    A = rng.random((size, size)) * (u - l) + l
    b = np.ones((size, 1))
    if normal:
        return np.conj(A.T) @ A, np.conj(A.T) @ np.ones((size, 1))
    else:
        return A, b

def GenAb(size):
    A = np.diag(rng.integers(10, size = size)) - np.eye(size, k = -1) - np.eye(size, k = 1)
    b = rng.random((size,1))
    return A, b




def getConfig(file_name: str):
    config_file = ConfigParser(converters={'intOrNone': intOrNone, 'floatList': floatList, 'precondFunction': PrecondFunction})
    config_file.read(file_name)

    config_file['Data']['dim'] = str(config_file.getint('Data', '1D_laplace_size')*config_file.getint('Data', '1D_laplace_size'))

    return config_file

def intOrNone(s:str):
    return None if s == 'None' else int(s)
    
def floatList(s:str):
    return [float(x) for x in s.split(' ')]



def PrecondFunction(precond_type: str):
    print("dawd")
    
    if precond_type.lower() == 'par_shift':
        return prec.parShift
    elif precond_type.lower() == 'par_shift_jacobi':
        return prec.parShiftJacobi

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