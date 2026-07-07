import math
import numpy as np
import inspect

class Node:
    def __init__(self, value, parents=[], op=""):
        self.value = value
        self.parents = parents 
        self.grad = 0.0
        self.op = op

    def _convert_to_node(self, other):
        if isinstance(other, (float,int)):
            other = Node(other)
        elif not isinstance(other, Node):
            raise Exception("Node must be of type Node, Float or Int")
        return other
    
    def __repr__(self):
        return f"Node(val={self.value}, grad={self.grad}, op={self.op})"

    def __add__(self, other):
        other = self._convert_to_node(other)
        out = Node(self.value + other.value, 
                   parents=[(self, 1.0), (other, 1.0)], 
                   op="+")
        
        return out

    def __radd__(self, other):
        return other + self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        other = self._convert_to_node(other)
        out = Node(self.value * other.value,
                   parents=[(self, other.value), (other, self.value)],
                   op="*")
        
        return out
    
    def __neg__(self):
        return -1 * self
    
    def __rmul__(self, other):
        return self * other 

    def __pow__(self, other):
        other = self._convert_to_node(other)
        out = Node(self.value ** other.value,
                   parents=[(self, other.value * self.value ** (other.value - 1))],
                   op="^")
        
        return out
    
    def __rpow__(self, other):
        return self ** other 
    
    def __truediv__(self, other):
        return self * other ** -1
    
    def ln(self):
        out = Node(math.log(self.value),
                   parents=[(self, 1/self.value)],
                   op="ln")
        return out
    
    def log(self, base):
        out = Node(math.log(self.value)/math.log(base),
                   parents=[(self, 1/(self.value * math.log(base)))],
                   op=f"log_{base}")
        return out
    
    def sin(self):
        out = Node(math.sin(self.value),
                   parents=[(self, math.cos(self.value))],
                   op=f"sin")
        return out

    def cos(self):
        out = Node(math.sin(self.value),
                   parents=[(self, -math.sin(self.value))],
                   op=f"cos")
        return out
    
    def tan(self):
        cos_val = math.cos(self.value)
        if abs(cos_val) < 1e-12:  
            raise ValueError("tan undefined at odd multiples of π/2")

        out = Node(math.tan(self.value),
                parents=[(self, 1 / (cos_val ** 2))],
                op="tan")
        return out
        

    def topo_sort(self):
        visited, order = set(), []
        def dfs(n):
            if n not in visited:
                visited.add(n)
                for p, _ in n.parents:
                    dfs(p)
                order.append(n)
        dfs(self)
        return order
    
    def reset_grads(self, order):
        for n in order:
            n.grad = 0
    
    def backward(self):
        """
        Must use topological sort to make sure that 
        all parents are accumulated before continuing.
        """

        order = self.topo_sort()
        self.reset_grads(order)
        self.grad = 1.0

        for node in reversed(order): 
            for parent, local_deriv in node.parents:
                parent.grad += node.grad * local_deriv


def grad(f, **inputs):

    sig = inspect.signature(f)
    arg_names = list(sig.parameters.keys())
    
    n = len(arg_names)
    
    vars = dict()
    for i, name in enumerate(sorted(arg_names)):
        if name not in inputs:
            raise ValueError(f"Missing input value for {name}")
        value = inputs[name]
        
        if not isinstance(value, (float,int)):
            raise TypeError(f"Input {name} must be of numerical (of type Int or Float)")
        vars[name] = Node(value)
    
    result = f(**vars)
    result.backward()
    
    grads = dict()
    for k,n in vars.items():
        grads[k] = n.grad
    return grads

    
