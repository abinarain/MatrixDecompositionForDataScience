from util import *
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

tmp = np.genfromtxt('ratings.csv', delimiter=',', dtype='int32')
val = np.array(tmp[:,2]+1, dtype='float')/6
# ratings are from 0 to 5, but we can't have 0 entries, as they'd be confused
# with missing ratings.
data = coo_matrix( (val, (tmp[:,0]-1, tmp[:,1]-1)) )
# data is 1-indexed, so we need to reduce 1 to get it 0-indexed

tmp = np.genfromtxt('ratings_validation.csv', delimiter=',', dtype='int32')
val = np.array(tmp[:,2]+1, dtype='float')/6
validation = coo_matrix( (val, (tmp[:,0]-1, tmp[:,1]-1)) )

n = data.get_shape()[0]
m = data.get_shape()[1]
avg = data.sum()/data.count_nonzero()
print(f'Data has {n} rows and {m} columns')
print(f'Average rating is {avg:.4F}')

# Read movies
movies = []
with open('movies.txt') as f:
    for row in f:
        movies.append(row.strip())

# Compute just factors
res = sgd(data, 10, factor_learning_rate=0.001)
val_err = squared_error(validation, res.L, res.R)
print('Only factors, fixed rate, no regularization')
print(f'Final training error = {res.err[-1]:5.3f}'
          f'\tRelative error = {res.err[-1]/squared_error(data)}')
print(f'    Validation error = {val_err:5.3f}'
          f'\tRelative error = {val_err/squared_error(validation)}')

# Plot errors over epocs
plt.plot(res.err)
plt.gca().set_yscale('log') # log scale makes these plots easier to read
plt.show()
plt.close()

# Print the top-10 and bottom-10 movies
print_factors(res.R, movies)

# Compute just factors, but use bold driver and slower learning
res = sgd(data, 10, bold_driver=2, factor_learning_rate=0.0001)
val_err = squared_error(validation, res.L, res.R)
print('Only factors, bold driver, no regularization')
print(f'Final training error = {res.err[-1]:5.3f}'
          f'\tRelative error = {res.err[-1]/squared_error(data)}')
print(f'    Validation error = {val_err:5.3f}'
          f'\tRelative error = {val_err/squared_error(validation)}')

# Plot errors over epocs
plt.plot(res.err)
plt.gca().set_yscale('log') # log scale makes these plots easier to read
plt.show()
plt.close()

#####
## YOUR PART STARTS HERE
## Play around with different learning rates & bold driver multipliers
#####

#####
## YOUR PART ENDS HERE
#####

# Compute only bias; factor_learning_rate=None means don't compute factors 
res = sgd(data, 10, bold_driver=2, factor_learning_rate=None, bias_learning_rate=0.0001, verbose=False)
val_err = squared_error(validation, res.L, res.R, res.global_bias, res.row_bias, res.col_bias)
print('Bias only, no regularization')
print(f'Final training error = {res.err[-1]:5.3f}'
          f'\tRelative error = {res.err[-1]/squared_error(data)}')
print(f'    Validation error = {val_err:5.3f}'
          f'\tRelative error = {val_err/squared_error(validation)}')


# Compute factors and bias
res = sgd(data, 10, bold_driver=2, factor_learning_rate=0.0001, bias_learning_rate=0.0001, verbose=False)
val_err = squared_error(validation, res.L, res.R, res.global_bias, res.row_bias, res.col_bias)
print('Factors and bias, no regularization')
print(f'Final training error = {res.err[-1]:5.3f}'
          f'\tRelative error = {res.err[-1]/squared_error(data)}')
print(f'    Validation error = {val_err:5.3f}'
          f'\tRelative error = {val_err/squared_error(validation)}')

# Print the movies based on their bias
print_factors(res.col_bias, movies)

# Add regularization
res = sgd(data, 10, bold_driver=2, factor_learning_rate=0.0001, bias_learning_rate=0.0001, factor_regu=0.001, bias_regu=0.001, verbose=False)
val_err = squared_error(validation, res.L, res.R, res.global_bias, res.row_bias, res.col_bias)
print('Factors and bias, regularized')
print(f'Final training error = {res.err[-1]:5.3f}'
          f'\tRelative error = {res.err[-1]/squared_error(data)}')
print(f'    Validation error = {val_err:5.3f}'
          f'\tRelative error = {val_err/squared_error(validation)}')

# To be able to compute the SVD, you can convert the data into full array with
#data.toarray()
# To use squared_error(), you must multiply Sigma to U or V and use a transpose of V.
#####
## YOUR PART STARTS HERE
## Use the parameters from earlier and compare your errors (training and validation)
## to those you get with rank-10 truncated SVD
#####

#####
## YOUR PART ENDS HERE
#####

############
## TASK 2 ##
############

# Initial factors are given as a tuple (L, R); R is transposed
init_LR = (np.ones(shape=(n,10)), np.ones(shape=(m,10)))
# Initial biases are a tuple (global_bias, row_bias, col_bias
# Give the true global bias.
init_biases = (avg, np.ones(shape=n), np.ones(shape=m))

# You can change the learning rates and bold driver factor if you want
res = sgd(data, 10, bold_driver=2, factor_learning_rate=0.0001, bias_learning_rate=0.0001, verbose=True, init_LR=init_LR, init_biases=init_biases)
val_err = squared_error(validation, res=res)
print('Initial solutions as all-1s vectors')
print(f'Final training error = {res.err[-1]:5.3f}'
          f'\tRelative error = {res.err[-1]/squared_error(data)}')
print(f'    Validation error = {val_err:5.3f}'
          f'\tRelative error = {val_err/squared_error(validation)}')

#####
## YOUR PART STARTS HERE
## Try the other initial solutions
#####


#####
## YOUR PART ENDS HERE
#####


# try two different values for learning rates using 5-fold cross validation
learning_rates = [1e-5, 1e-3]
err = []
for rate in learning_rates:
    err.append( cv(data, folds=5, k=10, bold_driver=2, factor_learning_rate=rate, bias_learning_rate=rate, verbose=False) )
print('Rate\terr')
for i in range(len(learning_rates)):
    print(f'{learning_rates[i]}\t{err[i]}')

#####
## YOUR PART STARTS HERE
## Implement the grid search. After you've found the optimum combination of
## hyperparameters, run sgd with full training data using those parameters and
## compute the validation error.
#####


#####
## YOUR PART ENDS HERE
#####


############
## TASK 3 ##
############    

from sklearn.impute import KNNImputer
from sklearn.decomposition import FastICA

# Read data
data = np.genfromtxt('housing_prices.csv', delimiter=',', skip_header=1, missing_values="NA", filling_values=np.nan)
# Read locations
loc=[]
with open('housing_prices_locations.txt', 'r') as f:
    for row in f:
        loc.append(row.strip())
# Read times
times=[]
with open('housing_prices_times.txt', 'r') as f:
    for row in f:
        times.append(row.strip())

# Plot data
for i in range(data.shape[0]):
    plt.subplot(4, 5, i+1)
    plt.title(loc[i])
    plt.plot(data[i,:])
plt.tight_layout()
plt.show()

# We'd like to have cities as columns and time stamps as rows, so transpose
data = data.T

# Remove missing values using k-nearest neighbour imputer
imputer = KNNImputer(n_neighbors=2)
imp_data=imputer.fit_transform(data)

#Plot data again

# Compute ICA
ica = FastICA(n_components=4)
all_comps = ica.fit_transform(imp_data)

# Plot components
for i in range(all_comps.shape[1]):
    ax = plt.subplot(2, 2, i+1)
    ax.set_xticks(np.arange(0,len(times),36))
    ax.set_xticklabels(np.arange(1987, 2015, 3), rotation='vertical')
    plt.title(f'Component {i+1}')
    plt.plot(all_comps[:,i])
plt.tight_layout()
plt.show()

# To impute mean of columns, we use SimpleImputer
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imp_data = mean_imputer.fit_transform(data)

#####
## YOUR PART STARTS HERE
## Do further analysis
#####


#####
## YOUR PART ENDS HERE
#####
