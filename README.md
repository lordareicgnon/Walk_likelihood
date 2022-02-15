# Walk-likelihood methods 
We present two novel algorithms for partitioning a network with symmetric edges (both weighted and unweighted) into non-overlapping communities. 
1. Walk-likelihood algorithm (WLA): WLA produces an optimal partition of network nodes into a given number of communities.
2. Walk-likelihood community finder (WLCF): WLCF predicts the optimal number of network communities mopt using global moves that involve bifurcation and merging of communities and employs WLA to refine node community assignments at each step.

# Overview

Here is an overview of the files included in the repository:
1. ```walk_likelihood.py```: File that defines the class walk_likelihood and the methods WLA and WLCF are implemented as functions of the class.
2. ```other_functions.py```:
3. ```Example_networks```

# class walk_likelihood

```
class walk_likelihood:
```
## Initialization
```
__init__(self, X)
```

### Parameters:

#### X: _{array-like, sparse matrix} of shape (N, N)_
The symmetric transition matrix of a network of size N.

## Walk-likelihood algorithm (WLA):

```
WLA(self, m=None, U=None, init='NMF',lm=8,max_iter_WLA=20,thr_WLA=0.99,eps=0.00000001)
```

### Parameters:
 
 __m:__ ___int, default= None___ The number of communities for the partition of the network. Not required if initialization is custom.

__U:__ ___ndarray of shape (N, m), default= None___
The matrix U refers to the initialization of the partition of the network of size N into m communities, only required to be specified if the intialization is custom.

##### init: _{'NMF','random', 'custom' }, default= 'NMF'_
The method to initialize U: the partition of the network of size N into m communities for WLA. If U is provided, then the initialization is set to custom.
##### l_max: _int, default= 8_
The length of random-walks
##### max_iter_WLA: _int, default= 20_
The maximum number of interations for WLA
##### thr_WLA: _float, default= 0.99_
The halting threshold for WLA
##### eps: _float, default= 0.00000001_
The lowest accepted non-zero value

## Walk-likelihood Community Finder (WLCF):

```
WLCF(self, U=[], max_iter_WLCF=50, thr_WLCF=0.99, bifuraction_type='random', **WLA_params)
```

### Parameters:

##### U: _ndarray of shape (N, m), default= None_
The matrix U refers to the initialization of the partition of the network of size N into m communities, not required generally. 
##### max_iter_WLCF: _int, default= 50_
The maximum number of interations for WLCF
##### thr_WLCF: _float, default= 0.99_
The halting threshold for WLCF
##### bifuraction_type: _{'random', 'NMF'}, default= random_
The method used for initilizing bifurcation.
##### **WLA_params 
The parameters that need to be specified to the Walk-likelihood Algorithm that will be used by WLCF

## Attributes:

Both WLA and WLCF have the following attributes

__N__: The number of nodes in the network

__m:__ The number of communities the network is partitioned into

__w__: The outward rate of each node specified in a 1-dimensional array of size N 

__comm_id:__ The community identity of each node specified in a 1-dimensional array of size N

__U:__ The partition of the network of size N into m communities specified in a two-dimensional array of size N X m


# Example

1. Walk-likelihood Algorithm

```
model=walk_likelihood(X)
model.WLA(m=5)
print(model.clusters)
```
