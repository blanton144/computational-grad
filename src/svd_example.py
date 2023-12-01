#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import astropy.table

def func(t, a1=1., f1=15., a2=2., f2=15., c0=0., c1=1., c2=-0.2, c3=0.02):
    dt = (t - t.mean())
    dt = dt / dt.std()
    dtf = (t - 1e+9) / 1.e+9
    y = (a1 * np.sin(np.pi * dtf * f1) +
         a2 * np.cos(np.pi * dtf * f2) +
         c0 + c1 * dt + c2 * dt**2 + c3 * dt**3)
    return(y)

np.random.seed(20)

nt = 1000
t = 1.e9 * np.random.random(size=nt)
y = func(t) + 2.0 * np.random.normal(size=len(t))

outstr_dtype = np.dtype([('time', np.float64), ('signal', np.float64)])
outstr = np.zeros(nt, dtype=outstr_dtype)
outstr['time'] = t
outstr['signal'] = y

outtable = astropy.table.Table(outstr)
outtable.write('signal.dat', format='ascii.fixed_width', overwrite=True)

plt.scatter(t, y, s=1)
plt.show()

dt = (t - t.min()) / (t.max() - t.min()) - 0.5
dt = (t - 1.e+9) / 1.e+9 - 0.5

A = np.zeros((len(t), 4))
A[:, 0] = 1.
A[:, 1] = dt
A[:, 2] = dt**2
A[:, 3] = dt**3

(u, w, vt) = np.linalg.svd(A, full_matrices=False)

ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())

c = ainv.dot(y)

ym = A.dot(c)

plt.scatter(t, y, label='data')
plt.scatter(t, ym, label='model')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()

print((y-ym).std())

nump = 50
A = np.zeros((len(t), nump))
for i in np.arange(nump):
    A[:, i] = dt**(i)

(u, w, vt) = np.linalg.svd(A, full_matrices=False)

print(w.max() / w.min())

ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())

c = ainv.dot(y)

ym = A.dot(c)

print((y-ym).std())

plt.scatter(t, y, label='data')
plt.scatter(t, ym, label='model')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()


nperiod = 20

A = np.zeros((len(t), 1 + nperiod * 2 + 3))
A[:, 0] = 1.
nn = np.arange(nperiod, dtype=np.int32) + 1
freqs = np.zeros(1 + nperiod * 2)
for n in nn:
    A[:, n] = np.cos(np.pi * n * (dt + 0.5))
    A[:, n + nperiod] = np.sin(np.pi * n * (dt + 0.5))
    freqs[n] = n
    freqs[n + nperiod] = n
A[:, nperiod * 2 + 1] = dt
A[:, nperiod * 2 + 2] = dt**2
A[:, nperiod * 2 + 3] = dt**3

(u, w, vt) = np.linalg.svd(A, full_matrices=False)

invw = 1. / w
wmax = w.max()
invw[w <= 1.e-7 * wmax] = 0.

ainv = vt.transpose().dot(np.diag(invw)).dot(u.transpose())

c = ainv.dot(y)

ym = A.dot(c)

print((y-ym).std())

plt.scatter(t, y, label='data')
plt.scatter(t, ym, label='model')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()

plt.scatter(freqs, c[0:len(freqs)]**2, label='power')
plt.xlabel('frequency')
plt.ylabel('coeff')
plt.legend()
plt.show()
