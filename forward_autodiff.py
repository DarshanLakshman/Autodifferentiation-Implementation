import math
import numpy as np
import inspect

class Dual:
    def __init__(self, real, dual):
        self.real = real
        self.dual = np.array(dual, dtype=float)

    def _to_dual(self, other):
        if isinstance(other, Dual):
            return other
        elif isinstance(other, (int, float)):
            return Dual(other, [0]*len(self.dual))
        else:
            raise TypeError(f"Unsupported type: {type(other)}")
    
    def __add__(self, n):
        n = self._to_dual(n)
        real = self.real + n.real
        dual = self.dual + n.dual
        return Dual(real, dual)
    
    def __radd__(self,n): return self + n 

    def __sub__(self, n):
        n = self._to_dual(n)
        real = self.real - n.real
        dual = self.dual - n.dual
        return Dual(real, dual)
    
    def __rsub__(self, n):
        return self - n
    
    def __mul__(self, n):
        """Product Rule"""
        n = self._to_dual(n)
        real = self.real * n.real
        dual = self.dual * n.real + self.real * n.dual
        return Dual(real, dual)
    
    def __rmul__(self,n): return self * n 
    
    def __truediv__(self, n):
        """Quotient Rule"""
        n = self._to_dual(n)
        if n.real == 0:
            raise ZeroDivisionError("Division by zero in Dual numbers")
        real = self.real / n.real
        dual = (self.dual * n.real - self.real * n.dual) / (n.real ** 2)
        return Dual(real, dual)
    
    def __rtruediv__(self,n): return self._to_dual(n)/self

    def __pow__(self, n):
        """ 
        Chain Rule 
        Case 1: Integer Exponent uses polynomial rule
        Case 2: Negative Real and potential for -ve sqrt

        Dual Exponent uses the following derivation:
        (a + bε) ** (c + dε) 
        = e^((c + dε) * ln(a + bε)) 
        ≈ e^(c * ln(a) + ε * (c*b/a + d*ln(a))) [using ln(a+bε) ≈ ln(a) + bε/a] 
        = a^c * (1 + ε * (c*b/a + d*ln(a))) 
        """
        n = self._to_dual(n)

        # TODO: currently fails for things like -2 ** 1/2
        if np.all(n.dual == 0):
            power = int(n.real)
            real = self.real ** power
            dual = power * (self.real ** (power - 1)) * self.dual
            return Dual(real, dual)

        if self.real <= 0 and 1:
            raise ValueError("Base must be positive for non-integer exponents")

        real = self.real ** n.real
        dual = real * ((self.dual * n.real) / self.real + n.dual * math.log(self.real))
        return Dual(real, dual)
    
    def __rpow__(self, base):
        """
        Chain Rule for real base: base ** (a + bε)
        = base^a * e^(bε * ln(base))
        ≈ base^a + bε * ln(base) * base^a
        """
        if not isinstance(base, (int, float)):
            raise TypeError("Base must be int or float")
        if base <= 0:
            raise ValueError("Base must be positive")
        real = base ** self.real
        dual = real * self.dual * math.log(base)
        return Dual(real, dual)
    
    def ln(self):
        """
        Chain Rule
        Using Taylor Series: ln(a+bε) ≈ ln(a) + bε/a
        """
        if self.real <= 0:
            raise ValueError("Log undefined for non-positive Dual reals")
        real = math.log(self.real)
        dual = self.dual / self.real
        return Dual(real, dual)
    
    def log(self, base):
        """
        Using Taylor Series: ln(a+bε) ≈ ln(a) + bε/a
        """
        if self.real <= 0:
            raise ValueError("Log undefined for non-positive Dual reals")
        if base <= 0 or base == 1:
            raise ValueError("Base must be positive and not equal to 1")
        real = math.log(self.real) / math.log(base)
        dual = self.dual / (self.real * math.log(base))
        return Dual(real, dual)

    def sin(self):
        """
        Chain Rule
        sin(a+bε)
        = sin(a) * cos(bε) + sin(bε) * cos(a) [By compound angle formula]
        ≈ sin(a) + bε * cos(a) [as bε -> 0 so cos(bε) -> 1 and sin(bε) = bε by small angle approx]
        """
        real = math.sin(self.real)
        dual = math.cos(self.real) * self.dual
        return Dual(real, dual)

    def cos(self):
        """
        Chain Rule
        cos(a+bε)
        = cos(a) * cos(bε) + sin(a) * sin(bε) [By compound angle formula]
        ≈ cos(a) - bε * sin(a) [as bε -> 0 so cos(bε) -> 1 and sin(bε) = bε by small angle approx]
        """
        real = math.cos(self.real)
        dual = -math.sin(self.real) * self.dual
        return Dual(real, dual)
    
    def exp(self):
        return Dual(math.e, 0) ** self

    def tan(self):
        if abs(math.cos(self.real)) < 1e-12:
            raise ValueError("tan undefined for real = (π/2 + kπ)")
        return self.sin() / self.cos()
    
    def sqrt(self):
        if self.real < 0:
            raise Exception("Real must be greater than 0")
        return self ** Dual(0.5, 0)

    def max(self, n):
        """
        Max function for Dual numbers using the convention:
        Boolean expressions like (a > b) are 1 if true and 0 if false.
        Can be used as ReLu = Max(0, x)
        """

        if self.real > n.real:
            return Dual(self.real, self.dual)
        elif self.real < n.real:
            return Dual(n.real, n.dual)
        
        raise Exception("Both are of the same value")

    def __repr__(self):
        return f"Dual(real = {self.real}, dual = {self.dual})"
    
    def __str__(self):
        return f"{self.real} + {self.dual}ε"

def grad(f, **inputs):

    sig = inspect.signature(f)
    arg_names = list(sig.parameters.keys())
    
    n = len(arg_names)
    
    dual_inputs = dict()
    for i, name in enumerate(arg_names):
        if name not in inputs:
            raise ValueError(f"Missing input value for {name}")
        value = inputs[name]
        if not isinstance(value, (float,int)):
            raise TypeError(f"Input {name} must be of numerical (of type Int or Float)")
        seed = [1 if j == i else 0 for j in range(n)]
        dual_inputs[name] = Dual(value, seed)
    
    result = f(**dual_inputs)

    if isinstance(result, (list, tuple)):
        jacobian = np.array([r.dual for r in result])
        reals = np.array([r.real for r in result])
        return reals, jacobian
    
    return result.real, result.dual
