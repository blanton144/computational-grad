#!/usr/bin/env python

import astropy.table
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

def logit_model(x, beta0, beta1):
    px = 1. / (1. + jnp.exp(- (beta0 + beta1 * x)))
    return(px)

np.random.seed(100)

beta0 = - 5.
beta1 = 0.1
nx = 1000
xmin = 5.
xmax = 80.
x = xmin + (xmax - xmin) * (np.arange(nx, dtype=np.float32) + 0.5) / np.float32(nx)

size = 100
xran = np.random.random(size) * (xmax - xmin) + xmin
pxran = logit_model(xran, beta0, beta1)
y = jnp.float32(np.random.random(size) < pxran)

outstr_dtype = np.dtype([('age', np.float32),
                         ('recognized_it', np.float32)])
outstr = np.zeros(size, dtype=outstr_dtype)
outstr['age'] = xran
outstr['recognized_it'] = y

t = astropy.table.Table(outstr)
t.write('survey.csv', overwrite=True)

def negloglike(params, *args):
    x = args[0]
    y = args[1]
    beta0 = params[0]
    beta1 = params[1]
    i0 = np.where(y == 0.)[0]
    i1 = np.where(y == 1.)[0]
    b01 = (beta0 + beta1 * x)
    lnpx0 = (- b01[i0] - jnp.log(1. + jnp.exp(- b01[i0])))
    lnpx1 = - jnp.log(1. + jnp.exp(- b01[i1]))
    nll = - lnpx0.sum() - lnpx1.sum()
    return(nll)
    
negloglike_grad = jax.grad(negloglike)

def hessian(f):
  return jax.jacfwd(jax.grad(f))

hess = hessian(negloglike)

beta0_start = - 2.
beta1_start = 0.5
xst = [beta0_start, beta1_start]
res = scipy.optimize.minimize(negloglike, xst, args=(xran, y),
                              jac=negloglike_grad, # hess=hess,
                              method='BFGS', tol=1.e-10)
print(res.x)

covar = np.linalg.inv(hess(res.x, xran, y))
print(covar)
print(np.sqrt(np.diag(covar)))

plt.plot(x, logit_model(x, beta0, beta1))
plt.plot(x, logit_model(x, res.x[0], res.x[1]))
plt.scatter(xran, y, s=2)
plt.show()

