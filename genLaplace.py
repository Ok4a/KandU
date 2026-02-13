import numpy as np
import scipy.sparse as spar
import scipy.sparse.linalg as sLG
import linearSolver as LS
# import matplotlib.pyplot as plt

class dist():
    def __init__(self, param = [0, 1], rng = None, dist_type = 'n'):
        self.param = param

        if dist_type.lower() in ['uni', 'uniform', 'n', 'normal','n']:
            self.dist_type = dist_type.lower()
        else:
            raise Exception(dist_type, ' not defined')
        if rng is None or type(rng) == int:
            self.rng = np.random.default_rng(seed = rng)

    def rand(self, size= None):
        if self.dist_type in  ['uni', 'uniform', 'u']:
            return self.rng.uniform(self.param[0], self.param[1], size = size)
        elif self.dist_type in ['normal', 'n']:
            return self.rng.normal(loc = self.param[0], scale = self.param[1], size = size)



def gen2dLaplace(N, param = [1, 0.4], seed = None, dist_type = 'n'):


    distribution = dist(param = param, rng = seed, dist_type = dist_type)

    L = -2 * spar.diags_array(distribution.rand(size = N)) + spar.diags_array(distribution.rand(size = N - 1), offsets = 1) + spar.diags_array(distribution.rand(size = N - 1), offsets = -1) + spar.eye(N, k = -N + 1) * distribution.rand() + spar.eye(N, k = N - 1) * distribution.rand()

    I1 = spar.diags_array(distribution.rand(size = N))
    I2 = spar.diags_array(distribution.rand(size = N))

    return spar.kron(I1, L) + spar.kron(L, I2)


def genLaplaceData(N, param = [1, 0.4], data_count=10, seed = None, dist_type = 'n'):

    if type(seed) == 'int' or seed is None:
        rng = np.random.default_rng(seed)
    else:
        rng = seed

    data = []

    for i in rng.integers(1e10, size = data_count):
        data.append(gen2dLaplace(N, param, seed=int(i), dist_type = dist_type))

    return data






# if __name__ == '__main__':

#     N = 30
#     # laplace = gen2dLaplace(N)

#     # np.set_printoptions(linewidth = 150, precision = 4)
#     # print(laplace.toarray())

#     # dense = laplace.toarray()
#     # print('Cond:', np.linalg.cond(dense))
#     # _, __, k, _ = LS.BiCGSTAB(laplace, np.ones(shape = (N * N, 1)), verbose = True)
#     # dense[dense == 0.0] = np.nan

#     # plt.figure(1)
#     # plt.spy(dense)
#     # # plt.spy(laplace, color = 'black')

#     # plt.figure(2)
#     # plt.imshow(dense)
#     # plt.show()


#     k_list = []
#     cond_list = []
#     for i in range(500):
#         print(i, end = '\r')
#         laplace = gen2dLaplace(N)
#         _, __, k, _ = LS.BiCGSTAB(laplace, np.ones(shape = (N * N, 1)), verbose = False)
#         cond_list.append(np.linalg.cond(laplace.toarray()))

#         k_list.append(k)



#     print('Dim:', N * N)
#     print('Iter')
#     print('Mean:', np.mean(k_list))
#     print('Median:', np.median(k_list))
#     print('Var:', np.var(k_list))
#     print('Std:', np.std(k_list))
#     print('Min:', np.min(k_list))
#     print('Max:', np.max(k_list))


#     print('Cond')
#     print('Mean:', np.mean(cond_list))
#     print('Median:', np.median(cond_list))
#     print('Var:', np.var(cond_list))
#     print('Std:', np.std(cond_list))
#     print('Min:', np.min(cond_list))
#     print('Max:', np.max(cond_list))

#     plt.figure(1)
#     plt.hist(k_list, bins = 40)
#     plt.figure(2)
#     plt.hist(cond_list, bins = 40)
#     plt.show()

