import numpy as np

def rosenbrock(x, y):
    return (1-x)**2 + 100*(y-x**2)**2

def d_rosenbrock(x, y):
    return np.array(2*(x-1) - 400*x*(y-x**2), 200*(y-x**2))


# f = rosenbrock
# d_f = d_rosenbrock
#
def f(x, y):
    return (1-x)**2 + (1-y)**2

def d_f(x, y):
    return np.array(-2*(1-x), -2*(1-y))


alpha = 1e-2
x0 = np.array([10, 10]) # (1, 1)
x1 = x0 - alpha * d_f(x0[0], x0[1])
i = 0
max_iter = 10000
while np.abs(f(x1[0], x1[1])) > 1e-16 and i < max_iter:
    x0 = x1
    x1 = x0 - alpha * d_f(x0[0], x0[1])
    i += 1