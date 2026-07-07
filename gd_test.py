import numpy as np
from gradient_descent import gradient_descent_backward_AD as gdbad
from gradient_descent import gradient_descent_forward_AD as gdfad


def f(x0, x1):
    return (x0-3)**2 + (x1+4)**2

def g(x0, x1, x2):
    return (x0-3)**2 + (x1+4)**2 + x2**2

def h(x0): 
    return x0**4 - 2*x0**3 + 2

print("Testing function f")
print(gdbad(f, np.array([1,2]), 0.001, 10000))
print(gdfad(f, np.array([1,2]), 0.001, 10000))

print("Test function g")
print(gdbad(g, np.array([5,6,7]), 0.001, 100000))
print(gdfad(g, np.array([5,6,7]), 0.001, 100000))

#function with a saddle point around 0
#finds the optimum point if after 
print("Test function h")
print(gdbad(h, np.array([-0.5]), 0.001, 100000))
print(gdfad(h, np.array([-0.5]), 0.001, 100000))
