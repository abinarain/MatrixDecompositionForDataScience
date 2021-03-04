import numpy as np
from scipy.sparse import coo_matrix, find
import matplotlib.pyplot as plt
from collections import namedtuple

# Return data type
Result = namedtuple('Result', ['L', 'R', 'global_bias', 'row_bias', 'col_bias', 'err'])

def normalize(L, R):
    "Normalizes L and R"
    if L is None or R is None:
        return (L, R)
    K = L.shape[1]
    for k in range(K):
        sL = sum(L[:,k])
        sR = sum(R[:,k])
        L[:,k] *= np.sqrt(sL*sR)/sL
        R[:,k] *= np.sqrt(sL*sR)/sR
    return (L, R)

def squared_error(D,
                      L=None, R=None,
                      g_bias = None, r_bias=None, c_bias=None,
                      res = None):
    """Computes the squared error 
If only data D is given, computes its squared Frobenius.
If also L and R are given, computes error between data and L times R.
If also bias terms are given (and are not None), adds them to the error.
If L and R are None or not given, but bias vectors are given, uses only them.
Input can also be a Result named tuple in res, which will overtake all other parameters.
"""
    nz_data = find(D)
    rows = nz_data[0]
    cols = nz_data[1]
    vals = nz_data[2]
    nnz = len(rows)
    # If res is given, it takes presendence
    if res is not None:
        L = res.L
        R = res.R
        g_bias = res.global_bias
        r_bias = res.row_bias
        c_bias = res.col_bias
    LR_present = L is not None and R is not None
    bias_present = g_bias is not None and r_bias is not None and c_bias is not None
    if not LR_present and not bias_present:
        return sum(vals**2)
    elif LR_present and not bias_present:
        return sum((vals[i] - L[rows[i],:].dot(R[cols[i],:]))**2
                    for i in range(nnz))
    elif not LR_present and bias_present:
        return sum((vals[i] - g_bias - r_bias[rows[i]] - c_bias[cols[i]])**2
                       for i in range(nnz))
    else:
        return sum((vals[i] - L[rows[i],:].dot(R[cols[i],:])
                        - g_bias - r_bias[rows[i]] - c_bias[cols[i]])**2
                        for i in range(nnz))
def sgd_epoch(D, L, R,
                  g_bias, r_bias, c_bias,
                  factor_learning_rate,
                  bias_learning_rate,
                  factor_regu,
                  bias_regu):
    "Computes a single epoch of sgd" 
    nz_data = find(D)
    rows = nz_data[0]
    cols = nz_data[1]
    vals = nz_data[2]
    pidx = np.random.permutation(len(rows)) # permute the indices
    for i in range(len(pidx)):
        row = rows[pidx[i]]
        col = cols[pidx[i]]
        val = vals[pidx[i]]
        err = val
        if L is not None:
            err -= L[row,:].dot(R[col,:])
        if g_bias is not None: # use also bias
            err -= g_bias + r_bias[row] + c_bias[col]
        if err > 1000: print(f'Error is {err}, you should probably use slower learning rate')
        # Update factors
        if factor_learning_rate is not None:
            L[row,:] += (factor_learning_rate * 2
                             * (err * R[col,:] - factor_regu * L[row,:]))
            R[col,:] += (factor_learning_rate * 2
                             * (err * L[row,:] - factor_regu * R[col,:]))
        # Update biases, if present
        if bias_learning_rate is not None:
            r_bias[row] += (bias_learning_rate * 2
                                * (err - bias_regu * r_bias[row]))
            c_bias[col] += (bias_learning_rate * 2
                                * (err - bias_regu * c_bias[col]))

    return (L, R, r_bias, c_bias)

def sgd(D, k,
            factor_learning_rate=None,
            bias_learning_rate=None,
            factor_regu=0.0,
            bias_regu=0.0,
            bold_driver=1,
            epochs=100,
            init_LR = None,
            init_biases = None,
            verbose=True):
    max_learning_rate = 0.01 # too fast learning is bad
    base_factor_learning_rate = factor_learning_rate
    base_bias_learning_rate = bias_learning_rate
    n = D.get_shape()[0] 
    m = D.get_shape()[1]
    # If both factor and bias learning rates are None, raise error
    if factor_learning_rate is None and bias_learning_rate is None:
        raise RuntimeError('Either factor_learning_rate or bias_learning_rate must be not None')
    # If factor_learning_rate is not give, use None for factors
    if factor_learning_rate is None:
        L = None
        R = None
    # If no initial solution, use uniform distribution and normalize
    elif init_LR is None:
        L = np.random.uniform(size=(n,k))
        R = np.random.uniform(size=(m,k))
        L, R = normalize(L, R)
    else:
        L = init_LR[0]
        R = init_LR[1]
    # If bias_learning_rate is not given, use None for biases 
    if bias_learning_rate is None:
        global_bias = None
        row_bias = None
        col_bias = None
    # If no initial solution, use normal distribution
    elif init_biases is None:
        global_bias = D.sum()/D.count_nonzero() # average rating
        row_bias = np.random.normal(size=n, scale=0.1)
        col_bias = np.random.normal(size=m, scale=0.1)
    else:
        global_bias = init_biases[0]
        row_bias = init_biases[1]
        col_bias = init_biases[2]
        
    err = np.full(epochs+1, np.nan)
    err[0] = squared_error(D, L, R, global_bias, row_bias, col_bias) 
    for epo in range(epochs):
        if verbose:
            print(f'Epoch {epo:3d}: error {err[epo]:.2F}'
                      f'\tfactor_learning rate = {factor_learning_rate}')
        L, R, row_bias, col_bias = sgd_epoch(D, L, R, global_bias, row_bias, col_bias, factor_learning_rate, bias_learning_rate, factor_regu, bias_regu)
        L, R = normalize(L, R)
        err[epo+1] = squared_error(D, L, R, global_bias, row_bias, col_bias)
        if err[epo] > err[epo+1]:
            if factor_learning_rate is not None:
                factor_learning_rate = min(factor_learning_rate * bold_driver,
                                               max_learning_rate)
            if bias_learning_rate is not None:
                bias_learning_rate = min(bias_learning_rate * bold_driver,
                                             max_learning_rate)
        else:
            factor_learning_rate = base_factor_learning_rate
            bias_learning_rate = base_bias_learning_rate

    return Result(L=L, R=R, global_bias=global_bias, row_bias=row_bias, col_bias=col_bias, err=err)
    

def print_factors(M, names, k=10, bottom=True):
    "Prints top-k (and bottom-k) factors using names"
    n = M.shape[0]
    if M.ndim == 1:
        M.shape = (n, 1) # Make M 2D
    m = M.shape[1]
    for i in range(m):
        vals = M[:,i]
        ids = np.argsort(vals)
        print(f'Factor {i} top-{k}')
        for j in range(k):
            print(f'{j+1:3d}\t{names[ids[-(j+1)]]}\t{vals[ids[-(j+1)]]:.3F}')
        if bottom:
            print('')
            print(f'Factor {i} bottom-{k}')
            for j in range(k):
                print(f'{j+1:3d}\t{names[ids[j]]}\t{vals[ids[j]]:.3F}')
        print()
    


def cv(D, folds=5, func=sgd, cv_verbose=True, **kwargs):
    "Computes cross validation and returns avg error"
    n = D.shape[0]
    m = D.shape[1]
    nz_data = find(D)
    rows = nz_data[0]
    cols = nz_data[1]
    vals = nz_data[2]
    data_len = len(rows)
    pidx = np.random.permutation(data_len) # permute the indices
    fold_len = len(rows)//folds
    leave_out_err = []
    for i in range(folds):
        if cv_verbose:
            print(f'Fold {i+1}/{folds}')
        if i < folds:
            lo = tuple(j for j in range(data_len) if j >= i*fold_len and j < (i+1)*fold_len)
        else: # last one has all remaining
            lo = tuple(j for j in range(data_len) if j >= i*fold_len)
        leave_out_idx = pidx[lo,]
        train_idx = pidx[tuple(np.setdiff1d(np.arange(data_len), lo, assume_unique=True)),]

        train_data = coo_matrix((vals[train_idx], (rows[train_idx], cols[train_idx])), shape=(n, m))
        leave_out_data = coo_matrix((vals[leave_out_idx], (rows[leave_out_idx], cols[leave_out_idx])), shape=(n, m))

        res = func(train_data, **kwargs)
        leave_out_err.append(squared_error(leave_out_data, res=res))
        if cv_verbose:
            print(f'\tError = {leave_out_err[-1]}')
    return np.mean(leave_out_err)
