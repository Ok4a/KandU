import numpy as np
from configparser import ConfigParser
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
    config_file = ConfigParser(converters={'intOrNone': intOrNone, 'floatList': floatList})
    config_file.read(file_name)

    config_file['Data']['dim'] = str(config_file.getint('Data', '1D_laplace_size')*config_file.getint('Data', '1D_laplace_size'))

    return config_file

def intOrNone(s:str):
    return None if s == 'None' else int(s)
    
def floatList(s:str):
    return [float(x) for x in s.split(' ')]

