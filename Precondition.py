import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg._interface import LinearOperator
import scipy.sparse as spar
import util
from itertools import islice, cycle
from configparser import ConfigParser
import matplotlib.pyplot as plt

def Jacobi(A, invert = True, normalEq = False):
     
    A = np.conj(A.T) @ A  if normalEq else A
    if invert:
        return diags(1/A.diagonal())
    else:
        return diags(np.diag(A))
    

class Jacobi_class():
    def __init__(self, A):
        self.diag = 1/np.diag(A)
        self.size = np.size(A,0)

    def mv(self,v):
        return np.array([self.diag[i] * v[i] for i in range(self.size)])
  

def shuffle(size, seed = None):
    rng = np.random.default_rng(seed = seed)
    row = np.arange(size)
    col = np.arange(size)
    rng.shuffle(col)
    data = np.array([1]*size)

    coo = spar.coo_array((data, (row, col)), shape = (size, size))
    
    return coo, col


class shift_precondition():
    def __init__(self, A, rng=None):
        self.size = np.size(A,0)
        if rng is None or type(rng) == int:
            rng = np.random.default_rng(seed = rng)
        self.alpha = rng.normal(size=(self.size, 1))

    def mv(self, v):
        return np.array([v[i]+ v[(i+1)%self.size]*self.alpha[i] for i in range(self.size)])
    
    def Linear(self):
        return LinearOperator(shape=(self.size,self.size), matvec = self.mv)
    
def randShift(size, numOffDiag = 1, rng = None, upper = True):
    if rng is None or type(rng) == int:
        rng = np.random.default_rng(seed = rng)

    if type(numOffDiag) == int:
        if numOffDiag < size:
            numOffDiag = range(1, numOffDiag + 1)
    
    bound = 0.05
    scale = 0.01
    prob = 0.5
    M = np.eye(size)

    for i in numOffDiag:
        offDiag = rng.normal(scale = scale, size = size - i)# * rng.binomial(1, prob, size = size - i)
        # offDiag = rng.uniform(low=-bound, high=bound,size=size-i)*rng.binomial(1,prob, size=size-i)
        if upper:
            M += np.diag(offDiag, k = i)
            # M += np.diag(offDiag, k = -i)
        else:
            M += np.diag(offDiag, k = -i)

    return spar.csr_array(M)

def randMShift(size, numShift, numOffDiag = 1, rng = None):


    M = spar.csr_array(np.eye(size))
    for i in range(numShift):
        M = randShift(size, numOffDiag, rng, upper = True) @ M

    return M



def parShift(size:int, upper_coef:list[int] = []):

    numUCoef = len(upper_coef)

    if numUCoef > size - 1:
        raise Exception("More coefficients than off diagonal elements")


    return spar.csr_array(np.eye(size) + np.diag(repCoef(size - 1, upper_coef, numUCoef), 1))

def parShiftOff(size:int, upper_coef:list[int] = []):

    numUCoef = len(upper_coef)

    if numUCoef > size - 1:
        raise Exception("More coefficients than off diagonal elements")


    # return spar.csr_array(np.diag(repCoef(size - 1, upper_coef, numUCoef), 1))
    return spar.diags(repCoef(size - 1, upper_coef, numUCoef), 1)

def repCoef(len, coef, numCoef):

    if numCoef == 1:
        temp = np.ones(shape=len)*coef[0]
        return temp


    split = np.array_split(np.ones((len, 1)), numCoef) 
    offDiag = split[0] * coef[0]
    for i in range(1, numCoef):
        offDiag = np.append(offDiag, split[i] * coef[i])
        
    return offDiag


def cycleCoef(size, coef):

    return spar.csr_array(np.eye(size) + np.diag(list(islice(cycle(coef), size-1)), 1))




def genInitalParShift(size, numCoef, seed = None, scale = 0.05):

    if type(seed) == 'int' or seed is None:
        rng = np.random.default_rng(seed)
    else:
        rng = seed

    coefList = rng.normal(scale=scale, size=numCoef)

    return coefList, parShift(size, coefList)



def parShiftJacobi(A, size:int, upper_coef:list[int] = []):

    numUCoef = len(upper_coef)

    if numUCoef > size - 1:
        raise Exception("More coefficients than off diagonal elements")


    return spar.csr_array(Jacobi(A) + np.diag(repCoef(size - 1, upper_coef, numUCoef), 1))


def superShift(size, coef_list):
    
    if len(coef_list) > size-1:
        raise Exception("More coefficients than super diagonals")


    diagonals=[[coef_list[ii]]*(size-ii-1) for ii in range(len(coef_list))]
    M_inv = spar.diags_array(diagonals, offsets= range(1,len(coef_list)+1))
    

    return M_inv


def superParShift(size, coef_list):
    
    if len(coef_list) > size-1:
        raise Exception("More coefficients than super diagonals")


    diagonals=[repCoef((size-ii-1), coef_list[ii], len(coef_list[ii])) for ii in range(len(coef_list))]
    M_inv = spar.diags_array(diagonals, offsets = range(1,len(coef_list)+1))
    

    return M_inv




class shiftPrecond():
    def __init__(self, config:ConfigParser, rng:np.random = None, zero_diag = True):
        self.config = config
        self.zero_diag = zero_diag
        # self.rng = np.random.default_rng(config.getint('Learn', 'seed'))
        self.rng = np.random.default_rng()

        self.coef_dict = self._genInitialCoef()
        self.temp_coef = self.coef_dict.copy()
        self.best_coef = self.coef_dict.copy()

        self.diag_list_index = None
        self.par_index = None
        self.coef_index = None


    def _genInitialCoef(self):
        par_list = self.config.getintList('Precondition', 'par_list')
        diag_list = self.config.getintList('Precondition', 'diag_list')

        coef_dict = {}
        for ii in range(len(diag_list)):
            coef_dict[diag_list[ii]] = self.rng.normal(scale = 0.05, size=par_list[ii])
        return coef_dict
    
    def makePrecond(self, coef_type = 'last', scale = None):
        size = self.config.getint('Data', 'dim')
        # par_list = config.getintList('Precondition', 'par_list')
        diag_list = self.config.getintList('Precondition', 'diag_list')

        if coef_type == 'last':
            self.temp_coef = self.coef_dict.copy()
        elif coef_type == 'best':
            self.temp_coef = self.best_coef.copy()


        if scale is not None:
            self.temp_coef[diag_list[self.diag_list_index]][self.par_index] += self.change*scale
        else:
            pass
            


        diagonals=[repCoef((size-ii), self.temp_coef[ii], len(self.temp_coef[ii])) for ii in diag_list]
        M_inv = spar.diags_array(diagonals, offsets = diag_list)

        return M_inv
    

    def newChange(self):

        par_list = config.getintList('Precondition', 'par_list')

        self.coef_index = self.rng.choice(self.config.getint('Precondition', 'num_coef'))
        par_sum = 0
        for ii in range(len(par_list)):
            if par_sum + par_list[ii] > self.coef_index:
                self.diag_list_index = ii
                self.par_index = self.coef_index-par_sum
                break
            else:
                par_sum += par_list[ii]


        self.change = self.rng.uniform(low = -self.config.getfloat('Learn', 'step_range'), high = self.config.getfloat('Learn', 'step_range'))



    
    def keep(self):
        self.coef_dict = self.temp_coef.copy()








if __name__ == '__main__':
    config = util.getConfig('test_config.ini')




    precond = shiftPrecond(config)


    M_inv = precond.makePrecond()
    dense2 = M_inv.toarray()
    dense2[dense2 == 0.0] = np.nan

    plt.figure(2)
    plt.imshow(dense2)

    precond.newChange()

    M_inv = precond.makePrecond(scale = 10000)
    dense1 = M_inv.toarray()
    dense1[dense1 == 0.0] = np.nan

    plt.figure(1)
    plt.imshow(dense1)
    plt.show()
    