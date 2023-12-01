#!/usr/bin/env python

import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return((x - 0.3)**2 * np.exp(x))

x = -10. + 20. * np.arange(100000) / np.float32(100000)
y = func(x)
plt.plot(x, y)
plt.xlim([-2., 2])
plt.ylim([-2., 2])
plt.show()

bracket = np.array([0.01, 0.75, 1.])

print(func(bracket[0]))
print(func(bracket[1]))
print(func(bracket[2]))

xmin = scipy.optimize.brent(func, brack=bracket)

print(xmin)
