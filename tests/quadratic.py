import numpy as np

def quadratic_normal(a=None, b=None, c=None):
    """Return quadratic eqn solutions using common formula

    Parameters
    ----------

    a : np.float32 or np.float64
        x^2 coefficient

    b : np.float32 or np.float64
        x^1 coefficient

    c : np.float32 or np.float64
        x^0 coefficient
 
    Returns
    -------

    x1 : np.float32 or np.float64
        + solution

    x2 : np.float32 or np.float64
        - solution
"""
    a = np.float64(a)
    b = np.float64(b)
    c = np.float64(c)
    sqrt_value  = np.sqrt(b**2 - 4. * a *c)
    x1 = (- b + sqrt_value) / (2. * a)
    x2 = (- b - sqrt_value) / (2. * a) 
    return(x1, x2)


def quadratic_inverted(a=None, b=None, c=None):
    """Return quadratic eqn solutions using inverted formula

    Parameters
    ----------

    a : np.float32 or np.float64
        x^2 coefficient

    b : np.float32 or np.float64
        x^1 coefficient

    c : np.float32 or np.float64
        x^0 coefficient
 
    Returns
    -------

    x1 : np.float32 or np.float64
        + solution

    x2 : np.float32 or np.float64
        - solution
"""
    a = np.float64(a)
    b = np.float64(b)
    c = np.float64(c)
    sqrt_value  = np.sqrt(b**2 - 4. * a *c)
    x1 = 2. * c / (- b - sqrt_value)
    x2 = 2. * c / (- b + sqrt_value)
    return(x1, x2)


def quadratic(a=None, b=None, c=None):
    """Return always-stable quadratic eqn solutions

    Parameters
    ----------

    a : np.float32 or np.float64
        x^2 coefficient

    b : np.float32 or np.float64
        x^1 coefficient

    c : np.float32 or np.float64
        x^0 coefficient
 
    Returns
    -------

    x1 : np.float32 or np.float64
        + solution

    x2 : np.float32 or np.float64
        - solution
"""
    a = np.float64(a)
    b = np.float64(b)
    c = np.float64(c)
    sqrt_value  = np.sqrt(b**2 - 4. * a *c)
    if(b > 0):
        x1 = 2. * c / (- b - sqrt_value)
        x2 = (- b - sqrt_value) / (2. * a) 
    else:
        x1 = (- b + sqrt_value) / (2. * a) 
        x2 = 2. * c / (- b + sqrt_value)
    return(x1, x2)
