import numpy as np

# def rosenbrock(x, y):
#     return (1-x)**2 + 100*(y-x**2)**2

# def f(y):
#     return rosenbrock(1, y)

def f(x):
    return np.cos(x)

def f_prime(x):
    return -np.sin(x)


x0 = 2
x1 = x0 - f(x0)/f_prime(x0)
i = 0
while abs(x1-x0)>1e-6:
    x0 = x1
    x1 = x0 - f(x0)/f_prime(x0)
    i += 1

print(x1) # Debería dar 1.5707963267948966
print(i)