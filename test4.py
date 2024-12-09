import numpy as np

lam = lambda x: x[0] + x[1] + x[2] + x[3]
lam1 = lambda x: 2 * x[0] + x[1] + x[2] + x[3]

x = np.random.randint(5, size=(10, 4))
print(x)
a, b = lam(x.T), lam1(x.T)
print(a, b)
print(np.stack([a, b], axis=1))
