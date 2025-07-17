from _benchmark_problems.antibody_design.cdrh3_design import BaseAntibodyDesgin
from _benchmark_problems.cco.cco import BaseCCO
from _benchmark_problems.MaxSAT.maximum_satisfiability import *
from _benchmark_problems.pest import BasePestControl
from _benchmark_problems.SVM.svm_mixed import BaseSVMMixed
from _benchmark_problems.synthetic_mixed import *
from mocahesp.utils import from_unit_cube



class Ackley53m(BaseAckleyMixed):
    def __init__(self, **kwargs):
        super(Ackley53m, self).__init__(**kwargs)
        self.input_dim = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        
        # create ohe map        
        self.cat_idx = self.categorical_idx_m
        self.disc_idx = []
        self.cont_idx = self.continuous_idx_m

        assert len(self.cat_idx) == self.categorical_dim_m
        assert len(self.disc_idx) == self.discrete_dim_m
        assert len(self.cont_idx) == self.continuous_dim_m

        self.bounds = [(0, 1)] * self.input_dim

    def func(self, x):    
        X = self.decode_input(x)
        return self.func_core(X)
    
    def decode_input(self, x):
        X = np.empty(self.dim_m)

        if len(self.cat_idx):
            # scale from R[[0,1]^50] to Z[[0,1]^50]
            lb, ub = self.bounds_m[self.categorical_idx_m,0], self.bounds_m[self.categorical_idx_m,1]
            temp = from_unit_cube(x[self.categorical_idx_m].reshape(1, -1), lb, ub).flatten()
            temp = np.round(temp)
            temp = np.clip(temp, lb, ub)
            X[self.categorical_idx_m] = temp

        if len(self.disc_idx):
            # scale from Z[0,1] to R[-2,2]
            # lb, ub = self.bounds_m[self.disc_idx,0], self.bounds_m[self.disc_idx,1]
            # temp = from_unit_cube(x[self.disc_idx].reshape(1, -1), lb, ub).flatten()
            # temp = np.round(temp)
            # temp = np.clip(temp, lb, ub)
            # X[self.discrete_idx_m] = temp
            pass


        if len(self.cont_idx):
            # scale from R[0,1] to R[-2,2]
            lb, ub = self.bounds_m[self.cont_idx,0], self.bounds_m[self.cont_idx,1]
            X[self.continuous_idx_m] = from_unit_cube(x[self.cont_idx].reshape(1, -1), lb, ub).flatten()

        return X
    
class Ackley20c(BaseAckleyCat):
    def __init__(self, **kwargs):
        super(Ackley20c, self).__init__(**kwargs)
        self.input_dim = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        
        # create ohe map        
        self.cat_idx = self.categorical_idx_m
        self.disc_idx = []
        self.cont_idx = []

        assert len(self.cat_idx) == self.categorical_dim_m
        assert len(self.disc_idx) == self.discrete_dim_m
        assert len(self.cont_idx) == self.continuous_dim_m

        self.bounds = [(0, 1)] * self.input_dim

    def func(self, x):    
        X = self.decode_input(x)
        return self.func_core(X)
    
    def decode_input(self, x):
        X = np.empty(self.dim_m)

        if len(self.cat_idx):
            # scale from R[[0,1]^50] to Z[[0,10]^50]
            lb, ub = self.bounds_m[self.categorical_idx_m,0], self.bounds_m[self.categorical_idx_m,1]
            temp = from_unit_cube(x[self.categorical_idx_m].reshape(1, -1), lb, ub).flatten()
            temp = np.round(temp)
            temp = np.clip(temp, lb, ub)
            X[self.categorical_idx_m] = temp

        return X

class MaxSAT28(BaseMaxSAT28):
    def __init__(self, **kwargs):
        super(MaxSAT28, self).__init__(**kwargs)
        self.input_dim = self.dim_m       
        self.bounds = [(0, 1)] * self.input_dim

    def func(self, x):    
        X = self.decode_input(x)
        return self.func_core(X).numpy()
    
    def decode_input(self, x):
        X = np.empty(self.dim_m)

        if len(self.categorical_idx_m):
            # scale from R[[0,1]^28] to Z[[0,1]^28]
            lb, ub = self.bounds_m[self.categorical_idx_m,0], self.bounds_m[self.categorical_idx_m,1]
            temp = from_unit_cube(x[self.categorical_idx_m].reshape(1, -1), lb, ub).flatten()
            temp = np.round(temp)
            temp = np.clip(temp, lb, ub)
            X[self.categorical_idx_m] = temp

        return X
    
class MaxSAT125(BaseMaxSAT125):
    def __init__(self, **kwargs):
        super(MaxSAT125, self).__init__(**kwargs)
        self.input_dim = self.dim_m       
        self.bounds = [(0, 1)] * self.input_dim

    def func(self, x):    
        X = self.decode_input(x)
        return self.func_core(X).numpy()
    
    def decode_input(self, x):
        X = np.empty(self.dim_m)

        if len(self.categorical_idx_m):
            # scale from R[[0,1]^125] to Z[[0,1]^125]
            lb, ub = self.bounds_m[self.categorical_idx_m,0], self.bounds_m[self.categorical_idx_m,1]
            temp = from_unit_cube(x[self.categorical_idx_m].reshape(1, -1), lb, ub).flatten()
            temp = np.round(temp)
            temp = np.clip(temp, lb, ub)
            X[self.categorical_idx_m] = temp

        return X  

class PestControl(BasePestControl):
    def __init__(self, **kwargs):
        super(PestControl, self).__init__(**kwargs)
        self.input_dim = self.dim_m  
        self.bounds = [(0, 1)] * self.input_dim

    def func(self, x):    
        X = self.decode_input(x)
        return self.func_core(X)
    
    def decode_input(self, x):
        X = np.empty(self.dim_m)

        if len(self.categorical_idx_m):
            # scale from R[[0,1]^25] to Z[[0,4]^25]
            lb, ub = self.bounds_m[self.categorical_idx_m,0], self.bounds_m[self.categorical_idx_m,1]
            temp = from_unit_cube(x[self.categorical_idx_m].reshape(1, -1), lb, ub).flatten()
            temp = np.round(temp)
            temp = np.clip(temp, lb, ub)
            X[self.categorical_idx_m] = temp

        return X
    
class CellularNetworkOpt(BaseCCO):
    def __init__(self, **kwargs):
        super(CellularNetworkOpt, self).__init__(**kwargs)
        self.input_dim = self.dim_m
        self.bounds = [(0, 1)] * self.input_dim

    def func(self, x):    
        X = self.decode_input(x)
        return self.func_core(X)
    
    def decode_input(self, x):
        X = np.empty(self.dim_m)

        if len(self.categorical_idx_m):
            # scale from Z[0,1] to Z[0,5]
            lb, ub = self.bounds_m[self.categorical_idx_m, 0], self.bounds_m[self.categorical_idx_m, 1]
            temp = from_unit_cube(x[self.categorical_idx_m].reshape(1, -1), lb, ub).flatten()
            temp = np.round(temp)
            temp = np.clip(temp, lb, ub)
            X[self.categorical_idx_m] = temp


        if len(self.continuous_idx_m):
            # scale from R[0,1] to R[30,50]
            lb, ub = self.bounds_m[self.continuous_idx_m, 0], self.bounds_m[self.continuous_idx_m, 1]
            X[self.continuous_idx_m] = from_unit_cube(x[self.continuous_idx_m].reshape(1, -1), lb, ub).flatten()


        return X
    
class LABS(BaseLabs):
    def __init__(self, **kwargs):
        super(LABS, self).__init__(**kwargs)
        self.input_dim = self.dim_m       
        self.bounds = [(0, 1)] * self.input_dim

    def func(self, x):    
        X = self.decode_input(x)
        return self.func_core(X).numpy()
    
    def decode_input(self, x):
        X = np.empty(self.dim_m)

        if len(self.categorical_idx_m):
            # scale from R[[0,1]^50] to Z[[0,1]^50]
            lb, ub = self.bounds_m[self.categorical_idx_m,0], self.bounds_m[self.categorical_idx_m,1]
            temp = from_unit_cube(x[self.categorical_idx_m].reshape(1, -1), lb, ub).flatten()
            temp = np.round(temp)
            temp = np.clip(temp, lb, ub)
            X[self.categorical_idx_m] = temp

        return X
    
class AntiBodyDesign(BaseAntibodyDesgin):
    def __init__(self, **kwargs):
        super(AntiBodyDesign, self).__init__(**kwargs)
        self.input_dim = self.dim_m  
        self.bounds = [(0, 1)] * self.input_dim

    def func(self, x):    
        X = self.decode_input(x)
        return self.func_core(X)
    
    def decode_input(self, x):
        X = np.empty(self.dim_m)

        if len(self.categorical_idx_m):
            # scale from R[[0,19]^25] to Z[[0,19]^25]
            lb, ub = self.bounds_m[self.categorical_idx_m,0], self.bounds_m[self.categorical_idx_m,1]
            temp = from_unit_cube(x[self.categorical_idx_m].reshape(1, -1), lb, ub).flatten()
            temp = np.round(temp)
            temp = np.clip(temp, lb, ub)
            X[self.categorical_idx_m] = temp

        return X
     
class SVMMixed(BaseSVMMixed):
    def __init__(self, **kwargs):
        super(SVMMixed, self).__init__(**kwargs)
        self.input_dim = self.dim_m       
        self.bounds = [(0, 1)] * self.input_dim

    def func(self, x):    
        X = self.decode_input(x)
        return self.func_core(X)
    
    def decode_input(self, x):
        X = np.empty(self.dim_m)

        if len(self.categorical_idx_m):
            # scale from R[[0,1]^50] to Z[[0,1]^50]
            lb, ub = self.bounds_m[self.categorical_idx_m,0], self.bounds_m[self.categorical_idx_m,1]
            temp = from_unit_cube(x[self.categorical_idx_m].reshape(1, -1), lb, ub).flatten()
            temp = np.round(temp)
            temp = np.clip(temp, lb, ub)
            X[self.categorical_idx_m] = temp
        
        if len(self.continuous_idx_m):
            # scale from R[0,1] to R[0,1]
            lb, ub = self.bounds_m[self.continuous_idx_m,0], self.bounds_m[self.continuous_idx_m,1]
            X[self.continuous_idx_m] = from_unit_cube(x[self.continuous_idx_m].reshape(1, -1), lb, ub).flatten()

        return X
