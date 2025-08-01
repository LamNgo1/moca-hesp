import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from casmopolitan_mocahesp.bo.localbo_utils import train_gp
from casmopolitan_mocahesp.bo.localbo_utils import from_unit_cube, latin_hypercube, to_unit_cube
from casmopolitan_mocahesp.bo.localbo_utils import onehot2ordinal
from casmopolitan_mocahesp.bo.localbo_utils import random_sample_within_discrete_tr_ordinal
import logging

class CASMOPOLITANCat:
    """

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Data types that require special treatments
    cat_dims: list of lists. e.g. [[1, 2], [3, 4, 5]], which denotes that indices 1,2,3,4,5 are categorical, and [1, 2]
        belong to the same variable (a categorical variable with 2 possible values) and [3, 4, 5] belong to another,
        with 3 possible values.
    int_dims: list. [2, 3, 4]. Denotes the indices of the dimensions that are of integer types

    true_dim: The actual dimension of the problem. When there is no categorical variables, this value would be the same
        as the dimensionality inferred from the data. When there are categorical variable(s), due to the one-hot
        transformation. If not supplied, the dimension inferred from the data will be used.

    """

    def __init__(
            self,
            dim,
            n_init,
            max_evals,
            config,
            batch_size=1,
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            device="cpu",
            dtype="float32",
            kernel_type='transformed_overlap',
            **kwargs
    ):

        # Very basic input checks
        # assert lb.ndim == 1 and ub.ndim == 1
        # assert len(lb) == len(ub)
        # assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # Save function information
        self.dim = dim
        self.config = config
        self.kwargs = kwargs
        # self.lb = lb
        # self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        self.kernel_type = kernel_type

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = kwargs['n_cand'] if 'n_cand' in kwargs.keys() else min(100 * self.dim, 5000)
        self.tr_multiplier = kwargs['multiplier'] if 'multiplier' in kwargs.keys() else 1.5
        self.failtol = kwargs['failtol'] if 'failtol' in kwargs.keys() else 40
        self.succtol = kwargs['succtol'] if 'succtol' in kwargs.keys() else 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = kwargs['length_min'] if 'length_min' in kwargs.keys() else 0.5 ** 7
        self.length_max = kwargs['length_max'] if 'length_max' in kwargs.keys() else 1.6
        self.length_init = kwargs['length_init'] if 'length_init' in kwargs.keys() else 0.8

        # Trust region sizes (in terms of Hamming distance) of the discrete variables.
        self.length_min_discrete = kwargs['length_min_discrete'] if 'length_min_discrete' in kwargs.keys() else 1
        self.length_max_discrete = kwargs['length_max_discrete'] if 'length_max_discrete' in kwargs.keys() else 30
        self.length_init_discrete = kwargs['length_init_discrete'] if 'length_init_discrete' in kwargs.keys() else 20

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        self._restart()
        self.X_cand = None # for logging

        self.create_candidates = kwargs['create_candidates']
        self.output_file = kwargs.get('output_file', None)
        
        if self.output_file is not None:
            log_file = f'{self.output_file}.txt'
            logging.root.setLevel(logging.DEBUG)
            logging.getLogger('').addHandler(logging.FileHandler(log_file))
            logging.getLogger('').addHandler(logging.StreamHandler(sys.stdout))

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init
        self.length_discrete = self.length_init_discrete

    def _adjust_length(self, fX_next):
        if np.min(fX_next) <= np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([self.tr_multiplier * self.length, self.length_max])
            # For the Hamming distance-bounded trust region, we additively (instead of multiplicatively) adjust.
            self.length_discrete = int(min(self.length_discrete * self.tr_multiplier, self.length_max_discrete))
            # self.length = min(self.length * 1.5, self.length_max)
            self.succcount = 0
            print("expand", self.length, self.length_discrete)
        elif self.failcount == self.failtol:  # Shrink trust region
            # self.length = max([self.length_min, self.length / 2.0])
            self.failcount = 0
            # Ditto for shrinking.
            self.length_discrete = int(self.length_discrete / self.tr_multiplier)
            # self.length = max(self.length / 1.5, self.length_min)
            print("Shrink", self.length, self.length_discrete)

    def _create_and_select_candidates(self, X, fX, length, n_training_steps, hypers):
        # assert X.min() >= 0.0 and X.max() <= 1.0
        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.device, self.dtype
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers,
                kern=self.kernel_type,
                noise_variance=self.kwargs['noise_variance'] if
                'noise_variance' in self.kwargs else None
            )
            # Save state dict
            hypers = gp.state_dict()
        # Standardize function values.
        # mu, sigma = np.median(fX), fX.std()
        # sigma = 1.0 if sigma < 1e-6 else sigma
        # fX = (deepcopy(fX) - mu) / sigma

        kwargs = {}
        kwargs["length_discrete"] = self.length_discrete
        kwargs["length_init_discrete"] = self.length_init_discrete
        kwargs["tr"] = {
            'length': self.length,
            'length_discrete': self.length_discrete,
            'failcount': self.failcount,
            'succcount': self.succcount,
        }
        # Use moca-hesp candidate generation
        X_cand = self.create_candidates(n_cand=5000, length=self.length, **kwargs) 
        # Generate n_cand candidates for the continuous variables, in their trust region
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand, dtype=torch.float32)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Select the best candidates
        X_next = np.ones((self.batch_size, self.dim))
        y_next = np.ones((self.batch_size, 1))
        for i in range(self.batch_size):
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_next[i, :] = deepcopy(y_cand[indbest, i])
            y_cand[indbest, :] = np.inf

        # Remove the torch tensors
        del X_torch, y_torch
        X_next = np.array(X_next)
        return X_next
