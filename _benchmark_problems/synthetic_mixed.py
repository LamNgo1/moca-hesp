# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np
import torch


def myrosenbrock(X):
    X = np.asarray(X)
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:  # one observation
        x1 = X[0]
        x2 = X[1]
    else:  # multiple observations
        x1 = X[:, 0]
        x2 = X[:, 1]
    fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return fx.reshape(-1, 1) / 300

def mysixhumpcamp(X):
    X = np.asarray(X)
    X = np.reshape(X, (-1, 2))
    if len(X.shape) == 1:
        x1 = X[0]
        x2 = X[1]
    else:
        x1 = X[:, 0]
        x2 = X[:, 1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    fval = term1 + term2 + term3
    return fval.reshape(-1, 1) / 10

def mybeale(X):
    X = np.asarray(X) / 2
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:
        x1 = X[0] * 2
        x2 = X[1] * 2
    else:
        x1 = X[:, 0] * 2
        x2 = X[:, 1] * 2
    fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
            2.625 - x1 + x1 * x2 ** 3) ** 2
    return fval.reshape(-1, 1) / 50



def func2C(ht_list, X):
    # ht is a categorical index
    # X is a continuous variable
    X = X * 2

    assert len(ht_list) == 2
    ht1 = ht_list[0]
    ht2 = ht_list[1]

    if ht1 == 0:  # rosenbrock
        f = myrosenbrock(X)
    elif ht1 == 1:  # six hump
        f = mysixhumpcamp(X)
    elif ht1 == 2:  # beale
        f = mybeale(X)

    if ht2 == 0:  # rosenbrock
        f = f + myrosenbrock(X)
    elif ht2 == 1:  # six hump
        f = f + mysixhumpcamp(X)
    else:
        f = f + mybeale(X)

    y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])
    return y.astype(float)

class BaseFunc2C():
    def __init__(self, lamda=1e-6, **kwargs):
        super(BaseFunc2C, self).__init__(**kwargs)
        self.name = 'func2c'
        self.categorical_idx_m = [0, 1]
        self.discrete_idx_m = []
        self.continuous_idx_m = [2, 3]
        self.bounds_m = np.array([
            [0, 2],     # cat
            [0, 4],     # cat
            [-1, 1],    # cont
            [-1, 1],    # cont
        ])

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)
        self.categorical_vertices_m = [self.bounds_m[cat_idx][1] - self.bounds_m[cat_idx][0] + 1 
                                       for cat_idx in self.categorical_idx_m]

        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.lamda = lamda

    def eval(self, ht_list, X):
        # ht is a categorical index
        # X is a continuous variable
        X = X * 2

        assert len(ht_list) == 2
        ht1 = ht_list[0]
        ht2 = ht_list[1]

        if ht1 == 0:  # rosenbrock
            f = myrosenbrock(X)
        elif ht1 == 1:  # six hump
            f = mysixhumpcamp(X)
        elif ht1 == 2:  # beale
            f = mybeale(X)

        if ht2 == 0:  # rosenbrock
            f = f + myrosenbrock(X)
        elif ht2 == 1:  # six hump
            f = f + mysixhumpcamp(X)
        else:
            f = f + mybeale(X)

        y = f + self.lamda * np.random.rand(f.shape[0], f.shape[1])
        return y.astype(float)
    
    def func_core(self, all_inputs):
        # The first 2 indices are for categorical
        if isinstance(all_inputs, np.ndarray):
            if all_inputs.ndim == 1:
                X = deepcopy(all_inputs)
            elif all_inputs.ndim == 2:
                X = all_inputs.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(all_inputs, list):  
            X = np.array(all_inputs) 
        else:
            raise NotImplementedError()  
         
        assert len(X) == self.dim_m
        
        for cat_idx in self.categorical_idx_m:
            X[cat_idx] = np.round(X[cat_idx]) # to make sure all are 0 or 1

        for i in range(self.dim_m):
            assert X[i] <= self.bounds_m[i][1] and X[i] >= self.bounds_m[i][0]
          
        return self.eval(X[0: 2], X[2:])


class BaseFunc3C():
    def __init__(self, lamda=1e-6, shifted=False, **kwargs):
        super(BaseFunc3C, self).__init__(**kwargs)
        self.name = 'func3c'
        self.categorical_idx_m = [0, 1, 2]
        self.discrete_idx_m = []
        self.continuous_idx_m = [3, 4]
        self.bounds_m = np.array([
            [0, 2],     # cat
            [0, 4],     # cat
            [0, 3],     # cat
            [-1, 1],    # cont
            [-1, 1],    # cont
        ])

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)
        self.categorical_vertices_m = [self.bounds_m[cat_idx][1] - self.bounds_m[cat_idx][0] + 1 
                                       for cat_idx in self.categorical_idx_m]

        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.lamda = lamda
        assert shifted == False, 'shifted version not supported'

    def eval(self, ht_list, X):
        # ht is a categorical index
        # X is a continuous variable
        X = np.atleast_2d(X)
        assert len(ht_list) == 3
        ht1 = ht_list[0]
        ht2 = ht_list[1]
        ht3 = ht_list[2]

        X = X * 2
        if ht1 == 0:  # rosenbrock
            f = myrosenbrock(X)
        elif ht1 == 1:  # six hump
            f = mysixhumpcamp(X)
        elif ht1 == 2:  # beale
            f = mybeale(X)

        if ht2 == 0:  # rosenbrock
            f = f + myrosenbrock(X)
        elif ht2 == 1:  # six hump
            f = f + mysixhumpcamp(X)
        else:
            f = f + mybeale(X)

        if ht3 == 0:  # rosenbrock
            f = f + 5 * mysixhumpcamp(X)
        elif ht3 == 1:  # six hump
            f = f + 2 * myrosenbrock(X)
        else:
            f = f + ht3 * mybeale(X)

        y = f + self.lamda * np.random.rand(f.shape[0], f.shape[1])

        return y.astype(float)
    
    def func_core(self, all_inputs):
        # The first 2 indices are for categorical
        if isinstance(all_inputs, np.ndarray):
            if all_inputs.ndim == 1:
                X = deepcopy(all_inputs)
            elif all_inputs.ndim == 2:
                X = all_inputs.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(all_inputs, list):  
            X = np.array(all_inputs) 
        else:
            raise NotImplementedError()  
         
        assert len(X) == self.dim_m
        
        for cat_idx in self.categorical_idx_m:
            X[cat_idx] = np.round(X[cat_idx]) # to make sure all are 0 or 1

        for i in range(self.dim_m):
            assert X[i] <= self.bounds_m[i][1] and X[i] >= self.bounds_m[i][0]
          
        return self.eval(X[0: 3], X[3:])



class BaseRosenbrock:  
    def __init__(self, discrete_dim=7, continuous_dim=3, **kwargs):
        super(BaseRosenbrock, self).__init__(**kwargs)
        self.name = f'rosenbrock-d{discrete_dim}'
        self.categorical_idx_m = []
        self.discrete_idx_m = list(range(discrete_dim))
        self.continuous_idx_m = [(discrete_dim + i) for i  in range(continuous_dim)]

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)

        self.bounds_m = np.vstack(
            (
                np.array([[-2,2] for _ in range(self.discrete_dim_m)]),
                np.array([[-2,2] for _ in range(self.continuous_dim_m)]),
            )
        )

        self.categorical_vertices_m = [self.bounds_m[cat_idx][1] - self.bounds_m[cat_idx][0] + 1 
                                       for cat_idx in self.categorical_idx_m]
        

        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
    
    def eval(self, X):
        fval = 0
        for i in range(self.dim_m-1):
            fval += (100*(X[i+1]-X[i]**2)**2 + (X[i]-1)**2)
        
        return fval
    
    def func(self, all_inputs):
        # The first 2 indices are for categorical
        if isinstance(all_inputs, np.ndarray):
            if all_inputs.ndim == 1:
                X = deepcopy(all_inputs)
            elif all_inputs.ndim == 2:
                X = all_inputs.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(all_inputs, list):  
            X = np.array(all_inputs) 
        else:
            raise NotImplementedError()  
         
        assert len(X) == self.dim_m
        for dis_idx in self.discrete_idx_m:
            X[dis_idx] = np.round(X[dis_idx]) # to make sure all integers

        for i in range(self.dim_m):
            assert X[i] <= self.bounds_m[i][1] and X[i] >= self.bounds_m[i][0]
            
        return np.array([self.eval(X)])
    

class BaseAckley:  
    def __init__(self, discrete_dim=7, continuous_dim=3):
        self.name = f'ackley-d{discrete_dim}'
        self.categorical_idx_m = []
        self.discrete_idx_m = list(range(discrete_dim))
        self.continuous_idx_m = [(discrete_dim + i) for i  in range(continuous_dim)]

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)

        self.bounds_m = np.vstack(
            (
                np.array([[0,1] for _ in range(self.discrete_dim_m)]),
                np.array([[-1,1] for _ in range(self.continuous_dim_m)]),
            )
        )

        self.categorical_vertices_m = [self.bounds_m[cat_idx][1] - self.bounds_m[cat_idx][0] + 1 
                                       for cat_idx in self.categorical_idx_m]
        

        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
    
    def eval(self, X):
        fval = 0
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(-1)/self.dim_m))-np.exp(np.cos(2*np.pi*X).sum(-1)/self.dim_m))

        return fval
    
    def func_core(self, all_inputs):
        # The first 2 indices are for categorical
        if isinstance(all_inputs, np.ndarray):
            if all_inputs.ndim == 1:
                X = deepcopy(all_inputs)
            elif all_inputs.ndim == 2:
                X = all_inputs.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(all_inputs, list):  
            X = np.array(all_inputs) 
        else:
            raise NotImplementedError()  
         
        assert len(X) == self.dim_m
        for dis_idx in self.discrete_idx_m:
            X[dis_idx] = np.round(X[dis_idx]) # to make sure all integers

        for i in range(self.dim_m):
            assert X[i] <= self.bounds_m[i][1] and X[i] >= self.bounds_m[i][0]
            
        return self.eval(X)
    

class BaseAckleyMixed:  
    def __init__(self, lamda=1e-6, categorical_dim=50, continuous_dim=3, shifted=False, **kwargs):
        super(BaseAckleyMixed, self).__init__(**kwargs)
        self.name = f'ackley{categorical_dim+continuous_dim}m'
        self.categorical_idx_m = list(range(categorical_dim))
        self.discrete_idx_m = []
        self.continuous_idx_m = [(categorical_dim + i) for i  in range(continuous_dim)]

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)

        # each categorical variable has 2 vertices, mapping from [0,1] to [0.0, 1.0]
        self.bounds_m = np.vstack(
            (
                np.array([[0,1] for _ in range(self.categorical_dim_m)]),
                np.array([[-1,1] for _ in range(self.continuous_dim_m)]),
            )
        )

        self.categorical_vertices_m = [self.bounds_m[cat_idx][1] - self.bounds_m[cat_idx][0] + 1 
                                       for cat_idx in self.categorical_idx_m]
        

        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.lamda = lamda
        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.shifted = shifted
        if self.shifted:
            self.offset_cat = np.random.RandomState(2024).choice([0, 1], size=self.categorical_dim_m)
            self.offset_cont = np.random.RandomState(2024).uniform(-1, 1, size=self.continuous_dim_m)
            self.name = f'shifted-{self.name}'
            print(f'offset={np.around(self.offset_cat, 0).tolist() + np.around(self.offset_cont, 4).tolist()}')
        else:
            self.offset_cat = np.zeros(self.categorical_dim_m)
            self.offset_cont = np.zeros(self.continuous_dim_m)
    
    def eval(self, X):
        fval = 0
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(-1)/self.dim_m))-np.exp(np.cos(2*np.pi*X).sum(-1)/self.dim_m))

        return fval + self.lamda * np.random.rand()
    
    def func_core(self, all_inputs):
        # The first 2 indices are for categorical
        if isinstance(all_inputs, np.ndarray):
            if all_inputs.ndim == 1:
                X = deepcopy(all_inputs)
            elif all_inputs.ndim == 2:
                X = all_inputs.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(all_inputs, list):  
            X = np.array(all_inputs) 
        else:
            raise NotImplementedError()  
         
        assert len(X) == self.dim_m
        for dis_idx in self.discrete_idx_m:
            X[dis_idx] = np.round(X[dis_idx]) # to make sure all integers

        for i in range(self.dim_m):
            assert X[i] <= self.bounds_m[i][1] and X[i] >= self.bounds_m[i][0]
        
        if self.shifted:
            X[self.categorical_idx_m] = (X[self.categorical_idx_m] + self.offset_cat) % 2 # binary
            X[self.continuous_idx_m] = X[self.continuous_idx_m] + self.offset_cont
            
        return np.array([self.eval(X)])

class BaseAckleyCat:
    def __init__(self, lamda=1e-6, categorical_dim=20, shifted=False, **kwargs):
        super(BaseAckleyCat, self).__init__(**kwargs)
        self.name = f'ackley{categorical_dim}c'
        self.categorical_idx_m = list(range(categorical_dim))
        self.discrete_idx_m = []
        self.continuous_idx_m = []

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)

        # each categorical variable has 11 vertices, mapping from [0,10] to [-32.768, 32.768]
        self.bounds_m = np.vstack(
            (
                np.array([[0, 10] for _ in range(self.categorical_dim_m)]),
            )
        )

        self.categorical_vertices_m = [self.bounds_m[cat_idx][1] - self.bounds_m[cat_idx][0] + 1 
                                       for cat_idx in self.categorical_idx_m]
        

        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.lamda = lamda
        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.shifted = shifted
        if self.shifted:
            self.offset = np.random.RandomState(2024).choice(11, size=self.categorical_dim_m)
            self.name = f'shifted-{self.name}'
            print(f'offset={np.around(self.offset, 0).tolist()}')
        else:
            self.offset = np.zeros(self.categorical_dim_m)
    
    def eval(self, X):
        fval = 0
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(-1)/self.dim_m))-np.exp(np.cos(2*np.pi*X).sum(-1)/self.dim_m))

        return fval + self.lamda * np.random.rand()
    
    def func_core(self, all_inputs):
        # The first 2 indices are for categorical
        if isinstance(all_inputs, np.ndarray):
            if all_inputs.ndim == 1:
                X = deepcopy(all_inputs)
            elif all_inputs.ndim == 2:
                X = all_inputs.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(all_inputs, list):  
            X = np.array(all_inputs) 
        else:
            raise NotImplementedError()  
         
        assert len(X) == self.dim_m
        # for dis_idx in self.discrete_idx_m:
        #     X[dis_idx] = np.round(X[dis_idx]) # to make sure all integers

        for i in range(self.dim_m):
            assert X[i] <= self.bounds_m[i][1] and X[i] >= self.bounds_m[i][0]
        
        if self.shifted:
            X[self.categorical_idx_m] = (X[self.categorical_idx_m] + self.offset) % self.categorical_vertices_m

        a = 2 * 32.768 / 10
        b = -32.768
        X_transformed = a * X + b
        return np.array([self.eval(X_transformed)])

class BaseLabs:
    """
    Low auto-correlation binaary 
    """
    def __init__(self, dim=50, shifted=False, **kwargs):
        super(BaseLabs, self).__init__(**kwargs)
        self.n_vertices = np.array([2] * dim)
        # Customized params
        self.name = f'labs{dim}'
        self.categorical_idx_m = list(range(dim))
        self.discrete_idx_m = []
        self.continuous_idx_m = []

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)

        self.bounds_m = np.vstack(
            (
                np.array([[0, 1] for _ in range(self.categorical_dim_m)]),
            )
        )
        self.categorical_vertices_m = [self.bounds_m[cat_idx][1] - self.bounds_m[cat_idx][0] + 1 
                                        for cat_idx in self.categorical_idx_m]

        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.shifted = shifted
        if self.shifted:
            self.offset = np.random.RandomState(2024).choice([-1, 1], size=self.dim_m, replace=True)
            self.name = f'shifted-{self.name}'
            print(f'offset={np.around(self.offset)}')
        else:
            self.offset = np.ones(self.dim_m)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return -torch.tensor([self._evaluate_single(xx) for xx in x])

    def _evaluate_single(self, x_eval: torch.Tensor) -> torch.Tensor:
        # from Z[0,1]^50 to Z{-1, 1}^50
        x = x_eval.detach().clone()
        assert x.dim() == 1
        if x.dim() == 2:
            x = x.squeeze(0)
        assert x.shape[0] == self.dim_m
        x = x.cpu().numpy()
        N = x.shape[0]
        x[x == 0] = -1
        if self.shifted:
            x = x * self.offset
        # print(f'x transformed {x}')
        E = 0  # energy
        for k in range(1, N):
            C_k = 0
            for j in range(0, N - k):
                C_k += (x[j] * x[j + k])
            E += C_k ** 2
        if E == 0:
            print("found zero")
        return (N**2)/ (2 * E)
    
    def func_core(self, x):
        '''
		Override the func method in base class
		'''
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                X = deepcopy(x)
            elif x.ndim == 2:
                X = x.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(x, list):  
            X = np.array(x) 
        elif isinstance(x, torch.Tensor):
            X = x.clone().detach()
        else:
            raise NotImplementedError() 
        if isinstance(X, dict):
            X = np.array([val for val in X.values()])
        if not isinstance(X, torch.Tensor):
            try:
                X = torch.tensor(X.astype(int))
            except:
                raise Exception('Unable to convert x to a pytorch tensor!')
        return self.evaluate(X)
    
class BaseAlpine:
    def __init__(self, lamda=0, continuous_dim=100, shifted=False, **kwargs):
        super(BaseAlpine, self).__init__(**kwargs)
        self.name = f'alpine{continuous_dim}'
        self.categorical_idx_m = []
        self.discrete_idx_m = []
        self.continuous_idx_m = list(range(continuous_dim))

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)

        self.bounds_m = np.vstack(
            (
                np.array([[-10.,10.] for _ in range(self.continuous_dim_m)]),
            )
        )

        self.categorical_vertices_m = []
        
        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.lamda = lamda
        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.shifted = shifted
        if self.shifted:
            self.offset_cont = np.random.RandomState(2024).uniform(-1, 1, size=self.continuous_dim_m)
            self.name = f'shifted-{self.name}'
            print(f'offset={np.around(self.offset_cont, 4).tolist()}')
        else:
            self.offset_cont = np.zeros(self.continuous_dim_m)
    
    def eval(self, X):
        # X = reshape(X, self.input_dim)
        temp = abs(X*np.sin(X) + 0.1*X)
        if len(temp.shape) <= 1:
            fval = np.sum(temp)
        else:
            fval = np.sum(temp, axis=1)

        return fval + self.lamda * np.random.rand()
    
    def func_core(self, all_inputs):
        # The first 2 indices are for categorical
        if isinstance(all_inputs, np.ndarray):
            if all_inputs.ndim == 1:
                X = deepcopy(all_inputs)
            elif all_inputs.ndim == 2:
                X = all_inputs.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(all_inputs, list):  
            X = np.array(all_inputs) 
        else:
            raise NotImplementedError()  
         
        assert len(X) == self.dim_m

        for i in range(self.dim_m):
            assert X[i] <= self.bounds_m[i][1] and X[i] >= self.bounds_m[i][0]
        
        if self.shifted:
            X[self.continuous_idx_m] = X[self.continuous_idx_m] + self.offset_cont
            
        return np.array([self.eval(X)])

class BaseLevy:
    def __init__(self, lamda=0, continuous_dim=100, shifted=False, **kwargs):
        super(BaseLevy, self).__init__(**kwargs)
        self.name = f'levy{continuous_dim}'
        self.categorical_idx_m = []
        self.discrete_idx_m = []
        self.continuous_idx_m = list(range(continuous_dim))

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)

        self.bounds_m = np.vstack(
            (
                np.array([[-10.,10.] for _ in range(self.continuous_dim_m)]),
            )
        )

        self.categorical_vertices_m = []
        
        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.lamda = lamda
        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.shifted = shifted
        if self.shifted:
            self.offset_cont = np.random.RandomState(2024).uniform(-1, 1, size=self.continuous_dim_m)
            self.name = f'shifted-{self.name}'
            print(f'offset={np.around(self.offset_cont, 4).tolist()}')
        else:
            self.offset_cont = np.zeros(self.continuous_dim_m)
    
    def eval(self, X):
        X = X.reshape(-1, self.dim_m)
        temp = abs(X*np.sin(X) + 0.1*X)
        w = np.zeros((X.shape[0], self.dim_m))
        for i in range(1, self.dim_m+1):
            w[:, i-1] = 1 + 1/4*(X[:, i-1]-1)

        fval = (np.sin(np.pi*w[:, 0]))**2 + ((w[:, self.dim_m-1]-1)**2)*(1+(np.sin(2*np.pi*w[:, self.dim_m-1]))**2)
        for i in range(1, self.dim_m):
            fval += ((w[:, i]-1)**2)*(1+10*(np.sin(np.pi*w[:, i]))**2) 

        return fval + self.lamda * np.random.rand()
    
    def func_core(self, all_inputs):
        # The first 2 indices are for categorical
        if isinstance(all_inputs, np.ndarray):
            if all_inputs.ndim == 1:
                X = deepcopy(all_inputs)
            elif all_inputs.ndim == 2:
                X = all_inputs.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(all_inputs, list):  
            X = np.array(all_inputs) 
        else:
            raise NotImplementedError()  
         
        assert len(X) == self.dim_m

        for i in range(self.dim_m):
            assert X[i] <= self.bounds_m[i][1] and X[i] >= self.bounds_m[i][0]
        
        if self.shifted:
            X[self.continuous_idx_m] = X[self.continuous_idx_m] + self.offset_cont
            
        return np.array([self.eval(X)])