#!/usr/bin/env python

import numpy as np
import numpy.polynomial.hermite


def func(x):
    f = x**2 / np.sqrt(np.pi)
    return(f)

x, w = numpy.polynomial.hermite.hermgauss(7)

print(x)

ii = (w * func(x)).sum()
print(ii)
