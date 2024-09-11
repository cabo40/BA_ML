import numpy as np

def f(x):
    return np.cos(x)

x0 = 3
x1 = 2
x2 = x0 - f(x0)*(x1-x0)/(f(x1)-f(x0))
i = 0
while abs(x1-x0)>1e-6:
    x0 = x1
    x1 = x2
    x2 = x0 - f(x0) * (x1 - x0) / (f(x1) - f(x0))
    i += 1

print(x2) # debería dar -1
print(i)