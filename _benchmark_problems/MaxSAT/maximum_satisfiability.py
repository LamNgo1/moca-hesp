import os
import numpy as np

import torch

from _benchmark_problems.MaxSAT.exp_utils import sample_init_points

MAXSAT_DIR_NAME = os.path.join(os.path.split(__file__)[0], 'maxsat2018_data')


class BaseMaxSAT():
	def __init__(self, data_filename, dims, normalize_weights=True, shifted=False, **kwargs):
		super(BaseMaxSAT, self).__init__(**kwargs)
		f = open(os.path.join(MAXSAT_DIR_NAME, data_filename), 'rt')
		line_str = f.readline()
		while line_str[:2] != 'p ':
			line_str = f.readline()
		self.n_variables = int(line_str.split(' ')[2])
		self.n_clauses = int(line_str.split(' ')[3])
		self.n_vertices = np.array([2] * self.n_variables)
		clauses = [(float(clause_str.split(' ')[0]), clause_str.split(' ')[1:-1]) for clause_str in f.readlines()]
		f.close()
		weights = np.array([elm[0] for elm in clauses]).astype(np.float32)
		if normalize_weights:
			weight_mean = np.mean(weights)
			weight_std = np.std(weights)
			self.weights = (weights - weight_mean) / weight_std
		else:
			self.weights = weights
		self.clauses = [([abs(int(elm)) - 1 for elm in clause], [int(elm) > 0 for elm in clause]) for _, clause in clauses]

		# Customized params
		self.name = f'maxsat{dims}'
		self.categorical_idx_m = list(range(dims))
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
			self.offset = np.random.RandomState(2024).choice([False, True], size=self.dim_m)
			self.name = f'shifted-{self.name}'
			print(f'offset={self.offset}')
		else:
			self.offset = np.zeros(self.dim_m, dtype=bool)
    
	
	def func_core(self, x):
		'''
		Override the func method in base class
		'''
		if isinstance(x, np.ndarray):
			if x.ndim == 1:
				X = np.copy(x)
			elif x.ndim == 2:
				X = x.flatten()
			else: 
				raise NotImplementedError()
		elif isinstance(x, list):  
			X = np.array(x) 
		else:
			raise NotImplementedError() 
		if isinstance(X, dict):
			X = np.array([val for val in x.values()])
		if not isinstance(X, torch.Tensor):
			try:
				X = torch.tensor(X)
			except:
				raise Exception('Unable to convert x to a pytorch tensor!')
		return self.evaluate(X)

	def evaluate(self, x):
		assert x.numel() == self.n_variables
		if x.dim() == 2:
			x = x.squeeze(0)
		x_np = (x.cpu() if x.is_cuda else x).numpy().astype(np.bool_)
		if self.offset is not None:
			x_np = np.logical_xor(x_np, self.offset)
		satisfied = np.array([(x_np[clause[0]] == clause[1]).any() for clause in self.clauses])
		return -np.sum(self.weights * satisfied) * x.float().new_ones(1, 1)


class BaseMaxSAT28(BaseMaxSAT):
	def __init__(self, **kwargs):
		super(BaseMaxSAT28, self).__init__(data_filename='maxcut-johnson8-2-4.clq.wcnf', dims=28, **kwargs)


class BaseMaxSAT43(BaseMaxSAT):
	def __init__(self, **kwargs):
		super(BaseMaxSAT43, self).__init__(data_filename='maxcut-hamming8-2.clq.wcnf', dims=43, **kwargs)


class BaseMaxSAT60(BaseMaxSAT):
	def __init__(self, **kwargs):
		super(BaseMaxSAT60, self).__init__(data_filename='frb-frb10-6-4.wcnf', dims=60, **kwargs)
		

class BaseMaxSAT125(BaseMaxSAT):
	def __init__(self, **kwargs):
		super(BaseMaxSAT125, self).__init__(data_filename='cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf', 
									  dims=125, normalize_weights=False, **kwargs)



# if __name__ == '__main__':
# 	import torch as torch_
# 	maxsat_ = BaseMaxSAT60()
# 	x_ = torch_.from_numpy(np.random.randint(0, 2, maxsat_.dim_m))
# 	eval_ = maxsat_.evaluate(x_)
# 	weight_sum_ = np.sum(maxsat_.weights)
# 	print(weight_sum_, eval_, weight_sum_ - eval_)
