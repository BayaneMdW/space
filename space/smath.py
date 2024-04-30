import numpy as np
import pandas as pd

def norm(u, v, w):
    return np.sqrt(u**2 + v**2 + w**2)

def resolve_poly1(a, b, epsilon=1e-7):
    if isinstance(a, np.ndarray) | isinstance(a, pd.Series):
        b = np.ones_like(a) * b
        r = np.zeros_like(a)
        r[(abs(a) >= epsilon)] = -b[ (abs(a) >= epsilon)] / a[ (abs(a) >= epsilon)]
    else :
        if abs(a) >= epsilon :
            r=-b/a
        else :
            r=0
    return r

def resolve_poly2(a, b, c):
    if isinstance(a, np.ndarray )  | isinstance(a,pd.Series):
        r1 = np.zeros_like(a)
        r2 = np.zeros_like(a)
        a_null = np.where(np.abs(a) < 1e-6)[0]

        delta = b ** 2 - 4 * a * c
        np.testing.assert_array_less(-delta, 0)

        r1, r2 = (-b + np.sqrt(delta)) / (2 * a), (-b + np.sqrt(delta)) / (2 * a)
        if isinstance(c, np.ndarray )  | isinstance(c,pd.Series):
            c = c[a_null]
        if isinstance(b, np.ndarray )  | isinstance(b,pd.Series):
            b = b[a_null]
        r1[a_null] = r2[a_null] = -c / b
    else:
        delta = b ** 2 - 4 * a * c
        np.testing.assert_array_less(-delta, 0)
        if np.abs(a) < 1e-6:
            r1 = r2 = -c / b
        else :
            r1, r2 = (-b + np.sqrt(delta)) / (2 * a), (-b - np.sqrt(delta)) / (2 * a)

    return r1, r2
