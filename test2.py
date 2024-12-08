import re

import numpy as np
import sympy as sp
import test3

from benchmarks.Examplers import get_example_by_name
from utils.Config import CegisConfig
from learn.generate_data import Data
from verify.SosVerify import SOS

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(0)

name = 'C1'
example = get_example_by_name(name)
print(name)

opts = {"example": example, 'batch_size': 100, "DEG_continuous": [2, 2, 1, 2]}
Config = CegisConfig(**opts)

data = Data(Config).generate_data_for_continuous()

data = np.array(data[0])

print(data.shape)

s = test3.s1

x = sp.symbols(['x0', 'x1'])

f = sp.lambdify(x, s)

Y = [f(*e) for e in data]
print(len(Y))

P = PolynomialFeatures(2)
X = P.fit_transform(data)
model = Ridge()
model.fit(X, Y)

print(P.get_feature_names_out(), model.coef_)
s = ''
for k, v in zip(P.get_feature_names_out(), model.coef_):
    k = re.sub(r' ', r'*', k)
    k = k.replace('^', '**')
    if v < 0:
        s += f'- {-v} * {k} '
    else:
        s += f'+ {v} * {k} '

x_ = sp.symbols(['x1', 'x2'])
temp = sp.sympify(s[1:])
b = sp.lambdify(x, temp)(*x_)
barrier = [b, None]
print(barrier)
sos = SOS(Config, barrier)
vis, state = sos.verify_continuous()
print(vis)
