# Walk-likelihood methods 
We present two novel algorithms for partitioning a network with symmetric edges (both weighted and unweighted) into non-overlapping communities. 
1. Walk-likelihood algorithm (WLA): WLA produces an optimal partition of network nodes into a given number of communities.
2. Walk-likelihood community finder (WLCF): WLCF predicts the optimal number of network communities mopt using global moves that involve bifurcation and merging of communities and employs WLA to refine node community assignments at each step.

# class walk_likelihood

```
class walk_likelihood:
```
The methods WLA and WLCF are implemented as functions in this class defined in the file walk_likelihood.py.

## Initialization
```
__init__(self, X)
```

### Parameters:

#### X: {array-like, sparse matrix} of shape (N, N)
The symmetric transition matrix of a network of size N.

## Functions:

Walk-likelihood algorithm
```
WLA(self,U=[],clusters=[],init='NMF',m=0,lm=8,max_iter_WLA=20,thr_WLA=0.99,eps=0.00000001)
```

### Parameters:

#### self

#### U

#### clusters

#### init

#### m

#### lm

#### max_iter_WLA

#### thr_WLA
#### eps
