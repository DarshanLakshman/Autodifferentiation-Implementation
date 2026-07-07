import time, forward_autodiff, reverse_autodiff
import numpy as np

def gradient_descent_forward_AD(f, x0, lr=0.01, steps=100):

    x = np.array(x0, dtype=float)

    for _ in range(steps):

        real, grad = forward_autodiff.grad(f, **{f"x{i}": val for i, val in enumerate(x)})

        x -= lr * grad

        if np.any(np.isnan(x)):
            raise ValueError("NaN encountered in gradient descent!\n\
                             \tPlease check (1) Learning Rate (lr)\n\
                             \t(2)Initial Values (x0)")
    return x

def gradient_descent_backward_AD(f, x0, lr=0.01, steps=100):

    x = np.array(x0, dtype=float)

    for _ in range(steps):

        grad = reverse_autodiff.grad(f, **{f"x{i}": val for i, val in enumerate(x)})
        x -= lr * np.array(list(grad.values()))

        if np.any(np.isnan(x)):
            raise ValueError("NaN encountered in gradient descent!\n\
                             \tPlease check (1) Learning Rate (lr)\n\
                             \t(2)Initial Values (x0)")

    return x
