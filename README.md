# Autodifferentiation Project

A small from-scratch project exploring automatic differentiation and optimization algorithms in Python — built as a learning exercise, not a production library.

## What's in here

| File | Description |
|---|---|
| `forward_autodiff.py` | Forward-mode automatic differentiation using dual numbers. |
| `reverse_autodiff.py` | Reverse-mode automatic differentiation using a computation graph (backpropagation). |
| `gradient_descent.py` | Gradient descent implemented on top of both forward-mode and reverse-mode autodiff. |
| `gd_tests.py` | A few smoke tests / demos running gradient descent on simple functions. |

## Forward-mode autodiff (`forward_autodiff.py`)

The core idea: represent every number as a **dual number** `a + bε`, where `ε² = 0`. If you seed the "dual" part correctly and push it through your normal arithmetic, the dual part comes out the other end holding the derivative — no symbolic differentiation or finite differences required.

### `Dual` class

Wraps a real value and a dual value (a NumPy array, so it can carry partial derivatives with respect to multiple inputs at once). Implements:

- Arithmetic: `+`, `-`, `*`, `/`, `**` (including dual-exponent and real-base cases), with the corresponding right-hand versions (`__radd__`, etc.)
- Transcendental functions: `sin`, `cos`, `tan`, `exp`, `ln`, `log(base)`, `sqrt`
- `max(self, n)` — a basic max/ReLU-style operator
- Each method's docstring shows the derivative rule (product rule, quotient rule, chain rule, small-angle approximations, etc.) it's implementing

### `grad(f, **inputs)`

Takes a plain Python function `f(x0, x1, ...)` and a set of keyword values, and:

1. Inspects `f`'s signature to find its argument names
2. Wraps each input as a `Dual`, seeding the dual part as a one-hot vector (so each input's "direction" is tracked independently)
3. Calls `f` with these dual inputs
4. Returns `(value, gradient)` — or, if `f` returns multiple outputs, `(values, jacobian)`

Example:

```python
from forward_autodiff import grad

def f(x0, x1):
    return (x0 - 3) ** 2 + (x1 + 4) ** 2

value, gradient = grad(f, x0=1.0, x1=2.0)
```

**Known limitations:**
- Non-integer powers require a positive base (`self.real <= 0` raises), so expressions like `(-2) ** 0.5` aren't supported.
- `max()` raises an exception if both values are exactly equal.

## Reverse-mode autodiff (`reverse_autodiff.py`)

The complementary approach to forward mode: build a **computation graph** as the function executes, then walk it backwards (backpropagation) to accumulate gradients. This is generally more efficient than forward mode when there are many inputs and few outputs (which is the common case in optimization and machine learning).

### `Node` class

Each `Node` wraps a scalar `value` and keeps a list of `parents` — the nodes that combined to produce it, along with the *local derivative* of the output with respect to each parent.

**Operator overloading:** the class overloads Python's arithmetic dunder methods so that ordinary-looking expressions (`x0 - 3`, `x1 + 4`, `x0 ** 2`, etc.) automatically build up the computation graph behind the scenes, node by node, as the function runs:

- `__add__` / `__radd__` — addition (`Node + Node`, `Node + number`, `number + Node`)
- `__sub__` / `__rsub__` — subtraction, built on top of `__add__` and `__neg__`
- `__mul__` / `__rmul__` — multiplication (product rule: local derivative w.r.t. each parent is the *other* parent's value)
- `__neg__` — negation (`-node`), implemented as `-1 * self`
- `__truediv__` — division, built on top of `__mul__` and `__pow__` (`a / b == a * b ** -1`)
- `__pow__` / `__rpow__` — exponentiation (power rule for the local derivative)

Each of these methods doesn't just compute the resulting value — it also records `parents=[(input_node, local_derivative), ...]` on the new `Node` it returns, which is what makes the backward pass possible later.

Beyond the operators, there are plain methods for functions Python has no operator for:

- `sin()`, `cos()`, `tan()`, `ln()`, `log(base)` — each returns a new `Node` whose value is the function applied, and whose `parents` list records the corresponding derivative rule (e.g. `d/dx sin(x) = cos(x)`, `d/dx ln(x) = 1/x`)

Graph-traversal explanation:

- **`topo_sort()`** — short for **topological sort**. This orders all the nodes in the computation graph so that every node's parents (the nodes it was built from) appear *before* it in the list. It does this with a depth-first search: for the current node, it first recurses into all of its parents, and only appends the current node to the result *after* all of its parents have been added. The practical effect is a valid dependency ordering — you can walk it in reverse and be sure that by the time you reach any given node, everything that depends on it has already had a chance to pass its gradient back.
- **`reset_grads(order)`** — zeroes out `.grad` on every node in the graph, so re-running `backward()` doesn't accidentally accumulate onto stale gradients from a previous call.
- **`backward()`** — this is backpropagation itself. It topologically sorts the graph, resets gradients, seeds the output node's own gradient as `1.0` (since `d(output)/d(output) = 1`), then walks the sorted list **in reverse** (from output back to inputs). At each node it applies the chain rule: for every `(parent, local_derivative)` pair, it adds `node.grad * local_derivative` onto `parent.grad`. By the time the walk reaches the original input nodes, each one's `.grad` holds `d(output)/d(input)` — the full chain rule product/sum has been accumulated across every path from that input to the output.

### `grad(f, **inputs)`

Same interface as the forward-mode version: inspects `f`'s signature, wraps each input as a `Node`, evaluates `f`, calls `.backward()` on the result, and returns a `{name: gradient}` dict.

```python
from reverse_autodiff import grad

def f(x0, x1):
    return (x0 - 3) ** 2 + (x1 + 4) ** 2

gradients = grad(f, x0=1.0, x1=2.0)
```

## Gradient descent (`gradient_descent.py`)

Two versions of vanilla gradient descent, differing only in how the gradient is computed:

- `gradient_descent_forward_AD(f, x0, lr=0.01, steps=100)` — uses `forward_autodiff.grad`
- `gradient_descent_backward_AD(f, x0, lr=0.01, steps=100)` — uses `reverse_autodiff.grad` (module not included, see note above)

Both take an initial point `x0` (a list/array), a learning rate, and a step count, and raise a `ValueError` if the iterates diverge to `NaN` (usually a sign the learning rate is too high).

## Tests / demos (`gd_tests.py`)

Runs both gradient descent variants on three toy functions:

- `f(x0, x1) = (x0-3)² + (x1+4)²` — a simple convex bowl, minimum at `(3, -4)`
- `g(x0, x1, x2) = (x0-3)² + (x1+4)² + x2²` — same idea in 3 variables
- `h(x0) = x0⁴ - 2x0³ + 2` — a non-convex function with a saddle-ish flat region near 0, included to show gradient descent can stall depending on the starting point

Run with:

```bash
python gd_tests.py
```

Note: because of the `Node.cos()` bug described below, results will be wrong for any test function that uses `cos()` when run through `gradient_descent_backward_AD` — worth fixing before trusting reverse-mode results on trig-based functions.

## Possible next steps

- Fix the negative-base fractional power case in `Dual.__pow__` (forward mode)
- Add a comparison of convergence speed/accuracy between forward-mode GD and reverse-mode GD on the same test functions
