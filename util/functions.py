import numpy as np 

def logistic(x, A, k, x0):
    return A / (1.0 + np.exp(-k * (x - x0)))
    