import numpy as np
import pandas as pd
from scipy import constants as cst

import sys
sys.path.append('.')
from .. import utils
from ..coordinates import coordinates as coords
from ..smath import resolve_poly2
from ..smath import norm as mnorm
from ..utils import listify

def get_tilt(date):
    doy = pd.to_numeric(pd.DatetimeIndex(date).strftime('%j'))
    ut = pd.to_numeric(pd.DatetimeIndex(date).strftime('%H')) + pd.to_numeric(pd.DatetimeIndex(date).strftime('%M'))/60
    tilt_year = 23.4 * np.cos((doy-172)*2*np.pi/365.25)
    tilt_day = 11.2 * np.cos((ut-16.72)*2*np.pi/24)
    tilt = (tilt_year + tilt_day)*np.pi/180
    return tilt

def _checking_angles(theta, phi):

    if isinstance(theta, np.ndarray) and isinstance(theta, np.ndarray) and len(theta.shape) > 1 and len(phi.shape) > 1:
        return np.meshgrid(theta, phi)
    return theta, phi


def _formisano1979(theta, phi, **kwargs):
    a11,a22,a33,a12,a13,a23,a14,a24,a34,a44 = kwargs["coefs"]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    A = a11 * ct**2 + st**2 * (a22* cp**2 + a33 * sp**2) \
        + ct * st * (a12 * cp + a13 * sp) + a23 * st**2 * cp * sp
    B = a14*ct + st*(a24*cp + a34*sp)
    C = a44
    D = B**2 - 4*A*C
    return (-B + np.sqrt(D))/(2*A)


def formisano1979(theta, phi, **kwargs):
    '''
    Formisano 1979 magnetopause model. Give the default position of the magnetopause.
    function's arguments :
        - theta : angle in radiant, can be int, float or array (1D or 2D)
        - phi   : angle in radiant, can be int, float or array (1D or 2D)
    kwargs:
        - boundary  : "magnetopause", "bow_shock"
        - base  : can be "cartesian" (default) or "spherical"

    information : to get a particular point theta and phi must be an int or a float
    (ex : the nose of the boundary is given with the input theta=0 and phi=0). If a plan (2D) of
    the boundary is wanted one of the two angle must be an array and the other one must be
    an int or a float (ex : (XY) plan with Z=0 is given with the input
    theta=np.linespace(-np.pi/2,np.pi/2,100) and phi=0). To get the 3D boundary theta and phi
    must be given an array, if it is two 1D array a meshgrid will be performed to obtain a two 2D array.

    return : X,Y,Z (base="cartesian")or R,theta,phi (base="spherical") depending on the chosen base
    '''

    if kwargs["boundary"] == "magnetopause":
        coefs = [0.65,1,1.16,0.03,-0.28,-0.11,21.41,0.46,-0.36,-221]
    elif kwargs["boundary"] == "bow_shock":
        coefs = [0.52, 1, 1.05, 0.13, -0.16, -0.08, 47.53, -0.42, 0.67, -613]
    else:
        raise ValueError("boundary: {} not allowed".format(kwargs["boundary"]))

    theta, phi = _checking_angles(theta, phi)
    r          =  _formisano1979(theta, phi, coefs = coefs)
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)



def mp_formisano1979(theta, phi, **kwargs):
    return formisano1979(theta, phi, boundary="magnetopause", **kwargs)

def bs_formisano1979(theta, phi, **kwargs):
    return formisano1979(theta, phi, boundary="bow_shock", **kwargs)




def fairfield1971(x, args):

    '''
    Fairfield 1971 : Magnetopause and Bow shock models. Give positions of the boudaries in plans (XY) with Z=0 and (XZ) with Y=0.
    function's arguments :
        - x :  X axis (array) in Re (earth radii)
        - args : coefficients Aij are determined from many boundary crossings and they depend on upstream conditions.

        --> Default parameter for the bow shock and the magnetopause respectively are :
            default_bs_fairfield = [0.0296,-0.0381,-1.28,45.644,-652.1]
            default_mp_fairfield = [-0.0942,0.3818,0.498,17.992,-248.12]

     return : DataFrame (Pandas) with the position (X,Y,Z) in Re of the wanted boudary to plot (XY) and (XZ) plans.
    '''


    A,B,C,D,E = args[0],args[1],args[2],args[3],args[4]

    a = 1
    b = A*x + C
    c = B*x**2 + D*x + E

    delta = b**2-4*a*c

    ym = (-b - np.sqrt(delta))/(2*a)
    yp = (-b + np.sqrt(delta))/(2*a)

    pos=pd.DataFrame({'X' : np.concatenate([x, x[::-1]]),
                      'Y' : np.concatenate([yp, ym[::-1]]),
                      'Z' : np.concatenate([yp, ym[::-1]]),})

    return pos.dropna()


def bs_Jerab2005(theta, phi, **kwargs):
    '''
    Jerab 2005 Bow shock model. Give positions of the box shock in plans (XY) with Z=0 and (XZ) with Y=0 as a function of the upstream solar wind.
    function's arguments :
        - Np : Proton density of the upstream conditions
        - V  : Speed of the solar wind
        - B  : Intensity of interplanetary magnetic field
        - gamma : Polytropic index ( default gamma=2.15)


        --> mean parameters :  Np=7.35, V=425.5,  B=5.49

     return : DataFrame (Pandas) with the position (X,Y,Z) in Re of the bow shock to plot (XY) and (XZ) plans.
    '''

    def make_Rav(theta, phi):
        a11 = 0.45
        a22 = 1
        a33 = 0.8
        a12 = 0.18
        a14 = 46.6
        a24 = -2.2
        a34 = -0.6
        a44 = -618

        a = a11 * np.cos(theta) ** 2 + np.sin(theta) ** 2 * (a22 * np.cos(phi) ** 2 + a33 * np.sin(phi) ** 2)
        b = a14 * np.cos(theta) + np.sin(theta) * (a24 * np.cos(phi) + a34 * np.sin(phi))
        c = a44

        delta = b ** 2 - 4 * a * c

        R = (-b + np.sqrt(delta)) / (2 * a)
        return R

    Np = kwargs.get('Np', 6.025)
    V = kwargs.get('V', 427.496)
    B = kwargs.get('B', 5.554)
    gamma = kwargs.get('gamma', 2.15)
    Ma = V * 1e3 * np.sqrt(Np * 1e6 * cst.m_p * cst.mu_0) / (B * 1e-9)

    C = 91.55
    D = 0.937 * (0.846 + 0.042 * B)
    R0 = make_Rav(0, 0)



    Rav = make_Rav(theta, phi)
    K = ((gamma - 1) * Ma ** 2 + 2) / ((gamma + 1) * (Ma ** 2 - 1))
    r = (Rav / R0) * (C / (Np * V ** 2) ** (1 / 6)) * (1 + D * K)
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)

def mp_shue1997(theta, phi, **kwargs):
    Pd = kwargs.get("Pd", 2.056)
    Bz = kwargs.get("Bz", -0.001)

    if isinstance(Bz, float) | isinstance(Bz, int):
        if Bz >= 0:
            r0 = (11.4 + 0.13 * Bz) * Pd ** (-1 / 6.6)
        else:
            r0 = (11.4 + 0.14 * Bz) * Pd ** (-1 / 6.6)
    else:
        if isinstance(Pd, float) | isinstance(Pd, int):
            Pd = np.ones_like(Bz) * Pd
        r0 = (11.4 + 0.13 * Bz) * Pd ** (-1 / 6.6)
        r0[Bz < 0] = (11.4 + 0.14 * Bz[Bz < 0]) * Pd[Bz < 0] ** (-1 / 6.6)
    a = (0.58 - 0.010 * Bz) * (1 + 0.010 * Pd)
    r = r0 * (2. / (1 + np.cos(theta))) ** a
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)

def mp_shue1998(theta, phi, **kwargs):
    '''
    Shue 1998 Magnetopause model.

     Returns the MP distance for given theta,
     dynamic pressure (in nPa) and Bz (in nT).

    - theta : Angle from the x axis (model is cylindrical symmetry)
         example : theta = np.arange( -np.pi+0.01, np.pi-0.01, 0.001)
    - phi   : unused, azimuthal angle, to get the same interface as all models

    kwargs:
    - "pdyn": Dynamic Pressure in nPa
    - "Bz"  : z component of IMF in nT

    '''

    Pd = kwargs.get("pdyn", 2)
    Bz = kwargs.get("Bz", 1)

    r0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd**(-1./6.6)
    a = (0.58-0.007*Bz)*(1+0.024*np.log(Pd))
    r = r0*(2./(1+np.cos(theta)))**a
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)


def mp_lin2010(phi_in ,th_in, Pd, Pm, Bz, tilt=0.):
    ''' The Lin 2010 Magnetopause model. Returns the MP distance for a given
    azimuth (phi), zenith (th), solar wind dynamic and magnetic pressures (nPa)
    and Bz (in nT).
    * th: Zenith angle from positive x axis (zenith) (between 0 and pi)
    * phi: Azimuth angle from y axis, about x axis (between 0 abd 2 pi)
    * Pd: Solar wind dynamic pressure in nPa
    * Pm: Solar wind magnetic pressure in nPa
    * tilt: Dipole tilt
    '''
    a = [12.544,
         -0.194,
         0.305,
         0.0573,
         2.178,
         0.0571,
         -0.999,
         16.473,
         0.00152,
         0.382,
         0.0431,
         -0.00763,
         -0.210,
         0.0405,
         -4.430,
         -0.636,
         -2.600,
         0.832,
         -5.328,
         1.103,
         -0.907,
         1.450]

    arr = type(np.array([]))

    if(type(th_in) == arr):
        th = th_in.copy()
    else:
        th = th_in

    if(type(phi_in) == arr):
        phi = phi_in.copy()
    else:
        phi = phi_in

    el = th_in < 0.
    if(type(el) == arr):
        if(el.any()):
            th[el] = -th[el]

            if(type(phi) == type(arr)):
                phi[el] = phi[el]+np.pi
            else:
                phi = phi*np.ones(th.shape)+np.pi*el
    else:
        if(el):
            th = -th
            phi = phi+np.pi

    P = Pd+Pm

    def exp2(i):
        return a[i]*(np.exp(a[i+1]*Bz)-1)/(np.exp(a[i+2]*Bz)+1)

    def quad(i, s):
        return a[i]+s[0]*a[i+1]*tilt+s[1]*a[i+2]*tilt**2

    r0 = a[0]*P**a[1]*(1+exp2(2))

    beta = [a[6] + exp2(7),
            a[10],
            quad(11, [1, 0]),
            a[13]]

    f = np.cos(0.5*th)+a[5]*np.sin(2*th)*(1-np.exp(-th))
    s = beta[0]+beta[1]*np.sin(phi)+beta[2]*np.cos(phi)+beta[3]*np.cos(phi)**2
    f = f**(s)

    c = {}
    d = {}
    TH = {}
    PHI = {}
    e = {}
    for i, s in zip(['n', 's'], [1, -1]):
        c[i] = a[14]*P**a[15]
        d[i] = quad(16, [s, 1])
        TH[i] = quad(19, [s, 0])
        PHI[i] = np.cos(th)*np.cos(TH[i])
        PHI[i] = PHI[i] + np.sin(th)*np.sin(TH[i])*np.cos(phi-(1-s)*0.5*np.pi)
        PHI[i] = np.arccos(PHI[i])
        e[i] = a[21]
    r = f*r0

    Q = c['n']*np.exp(d['n']*PHI['n']**e['n'])
    Q = Q + c['s']*np.exp(d['s']*PHI['s']**e['s'])

    return r+Q


def mp_liu2015(theta, phi, **kwargs):
    if isinstance(theta, np.ndarray) | isinstance(theta, pd.Series):
        idx = np.where(theta < 0)[0]
        if isinstance(phi, np.ndarray) | isinstance(phi, pd.Series):
            phi[idx] = phi[idx] + np.pi
        else:
            phi = phi * np.ones(theta.shape)
            phi[idx] = phi[idx] + np.pi
    else:
        phi = phi + (theta < 0) * np.pi

    theta = np.sign(theta) * theta

    Pd = kwargs.get('Pd', 2.056)
    Bx = kwargs.get('Bx', 0.032)
    By = kwargs.get('By', -0.015)
    Bz = kwargs.get('Bz', -0.001)
    tilt = kwargs.get('tilt', 0)
    B_param = [('Bx' in kwargs), ('By' in kwargs), ('Bz' in kwargs)]
    if all(B_param):
        Pm = (Bx ** 2 + By ** 2 + Bz ** 2) * 1e-18 / (2 * cst.mu_0) * 1e9
    elif not any(B_param):
        Pm = 0.016
    else:
        raise ValueError('None or all Bx, By, Bz parameters must be set')

    P = Pd + Pm

    r0 = (10.56 + 0.956 * np.tanh(0.1795 * (Bz + 10.78))) * P ** (-0.1699)

    alpha_0 = (0.4935 + 0.1095 * np.tanh(0.07217 * (Bz + 6.882))) * (1 + 0.01182 * np.log(Pd))

    alpha_z = 0.06263 * np.tanh(0.0251 * tilt)

    alpha_phi = (0.06354 + 0.07764 * np.tanh(0.07217 * (abs(Bz) + 4.851))) * (1 + 0.01182 * np.log(Pd))

    delta_alpha = 0.02582 * np.tanh(0.0667 * Bx) * np.sign(Bx)

    if isinstance(Bz, np.ndarray) | isinstance(Bz, pd.Series):
        idx_zero = np.where(Bz != 0)[0]
        omega = np.zeros_like(np.shape(Bz))
        omega = omega + np.sign(By) * np.pi / 2
        omega[idx_zero] = np.arctan2(0.1718 * By[idx_zero] * (By[idx_zero] ** 2 + Bz[idx_zero] ** 2) ** 0.194,
                                     Bz[idx_zero])

    else:
        if Bz != 0:
            omega = np.arctan2(0.1718 * By * (By ** 2 + Bz ** 2) ** 0.194, Bz)
        else:
            omega = np.sign(By) * np.pi / 2

    alpha = alpha_0 + alpha_z * np.cos(phi) + (alpha_phi + delta_alpha * np.sign(np.cos(phi))) * np.cos(
        2 * (phi - omega))

    l_n = (0.822 + 0.2921 * np.tanh(0.08792 * (Bz + 10.12))) * (1 - 0.01278 * tilt)
    l_s = (0.822 + 0.2921 * np.tanh(0.08792 * (Bz + 10.12))) * (1 + 0.01278 * tilt)
    w = (0.2382 + 0.005806 * np.log(Pd)) * (1 + 0.0002335 * tilt ** 2)

    C = np.exp(-abs(theta - l_n) / w) * (1 + np.sign(np.cos(phi))) + np.exp(-abs(theta - l_s) / w) * (
        1 + np.sign(-np.cos(phi)))

    r = (r0 * (2 / (1 + np.cos(theta))) ** alpha) * (1 - 0.1 * C * np.cos(phi) ** 2)

    return coords.choice_coordinate_system(r, theta, phi, **kwargs)

def mp_jelinek2012(theta, phi, **kwargs):
    lamb = 1.54
    R = 12.82
    epsilon = 5.26
    Pd = kwargs.get('Pd', 2.056)

    R0 = 2 * R * Pd ** (-1 / epsilon)
    r = R0 / (np.cos(theta) + np.sqrt(np.cos(theta) ** 2 + np.sin(theta) * np.sin(theta) * lamb ** 2))
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)


def bs_jelinek2012(theta, phi, **kwargs):
    lamb = 1.17
    R = 15.02
    epsilon = 6.55
    Pd = kwargs.get('Pd', 2.056)

    R0 = 2 * R * Pd ** (-1 / epsilon)
    r = R0 / (np.cos(theta) + np.sqrt(np.cos(theta) ** 2 + np.sin(theta) * np.sin(theta) * lamb ** 2))
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)

def mp_nguyen2020(theta, phi, **kwargs):
    def inv_cos(t):
        return 2 / (1 + np.cos(t))

    ## return an array of phi and theta in the right ranges of values
    if isinstance(theta, np.ndarray) | isinstance(theta, pd.Series):
        t = np.array(theta).copy()
        idx = np.where(t < 0)[0]
        if isinstance(phi, np.ndarray) | isinstance(phi, pd.Series):
            p = np.array(phi).copy()
            p[idx] = p[idx] + np.pi
        else:
            p = phi * np.ones(t.shape)
            p[idx] = p[idx] + np.pi
    else:
        t = theta
        p = phi + (t < 0) * np.pi

    t = np.sign(t) * t

    # coefficients
    a = [10.61,
         -0.150,
         0.027,
         0.207,
         1.62,
         0.558,
         0.135,
         0.0150,
         -0.0839]

    Pd = kwargs.get('Pd', 2.056)
    Bx = kwargs.get('Bx', 0.032)
    By = kwargs.get('By', -0.015)
    Bz = kwargs.get('Bz', -0.001)
    gamma = kwargs.get('tilt', 0.)

    B_param = [('Bx' in kwargs), ('By' in kwargs), ('Bz' in kwargs)]
    if all(B_param):
        Pm = (Bx ** 2 + By ** 2 + Bz ** 2) * 1e-18 / (2 * cst.mu_0) * 1e9
    elif not any(B_param):
        Pm = 0.016
    else:
        raise ValueError('None or all Bx, By, Bz parameters must be set')

    omega = np.sign(By) * np.arccos(Bz / np.sqrt(By ** 2 + Bz ** 2))
    P = Pd + Pm
    r0 = a[0] * (1 + a[2] * np.tanh(a[3] * Bz + a[4])) * P ** (a[1])

    alpha0 = a[5]
    alpha1 = a[6] * gamma
    alpha2 = a[7] * np.cos(omega)
    alpha3 = a[8] * np.cos(omega)

    alpha = alpha0 + alpha1 * np.cos(p) + alpha2 * np.sin(p) ** 2 + alpha3 * np.cos(p) ** 2
    r = r0 * inv_cos(t) ** alpha
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)


def mp_sibeck1991(theta, phi, **kwargs):
    Pd = kwargs.get('Pd', 2.056)
    if (Pd >= 0.54) and (Pd < 0.87):
        a0 = 0.19
        b0 = 19.3
        c0 = -272.4
        p0 = 0.71
    elif (Pd >= 0.87) and (Pd < 1.47):
        a0 = 0.19
        b0 = 18.7
        c0 = -243.9
        p0 = 1.17
    elif (Pd >= 1.47) and (Pd < 2.60):
        a0 = 0.14
        b0 = 18.2
        c0 = -217.2
        p0 = 2.04
    elif (Pd >= 2.60) and (Pd < 4.90):
        a0 = 0.15
        b0 = 17.3
        c0 = -187.4
        p0 = 3.75
    elif (Pd >= 4.90) and (Pd < 9.90):
        a0 = 0.18
        b0 = 14.2
        c0 = -139.2
        p0 = 7.4

    else:
        raise ('Dynamic pressure none valid')

    a = a0 * np.cos(theta) ** 2 + np.sin(theta) ** 2
    b = b0 * np.cos(theta) * (p0 / Pd) ** (1 / 6)
    c = c0 * (p0 / Pd) ** (1 / 3)

    r = resolve_poly2_real_roots(a, b, c)[0]

    return coords.choice_coordinate_system(r, theta, phi, **kwargs)



def derivative_spherical_to_cartesian(r, theta, phi, drdt, drdp):
    dxdt = drdt * np.cos(theta) - r * np.sin(theta)
    dydt = drdt * np.sin(theta) * np.sin(phi) + r * np.cos(theta) * np.sin(phi)
    dzdt = drdt * np.sin(theta) * np.cos(phi) + r * np.cos(theta) * np.cos(phi)

    normt = mnorm(dxdt, dydt, dzdt)
    normt[normt == 0] = 1

    dxdp = 0
    dydp = drdp * np.sin(theta) * np.sin(phi) + r * np.sin(theta) * np.cos(phi)
    dzdp = drdp * np.sin(theta) * np.cos(phi) - r * np.sin(theta) * np.sin(phi)

    normp = mnorm(dxdp, dydp, dzdp)
    normp[normp == 0] = 1

    return [dxdt / normt, dydt / normt, dzdt / normt], [dxdp / normp, dydp / normp, dzdp / normp]


def find_normal_from_tangents(tth, tph):
    [dxdt, dydt, dzdt], [dxdp, dydp, dzdp] = tth, tph
    pvx = dzdt * dydp - dydt * dzdp
    pvy = dxdt * dzdp - dzdt * dxdp
    pvz = dydt * dxdp - dxdt * dydp

    norm = mnorm(pvx, pvy, pvz)

    pvx[norm == 0] = 1
    norm[norm == 0] = 1

    return (pvx / norm, pvy / norm, pvz / norm)




def mp_sibeck1991_tangents(theta, phi, **kwargs):
    theta = listify(theta)
    phi = listify(phi)
    Pd = kwargs.get("Pd", 2.056)
    if (Pd >= 0.54) and (Pd < 0.87):
        a0 = 0.19
        b0 = 19.3
        c0 = -272.4
        p0 = 0.71
    elif (Pd >= 0.87) and (Pd < 1.47):
        a0 = 0.19
        b0 = 18.7
        c0 = -243.9
        p0 = 1.17
    elif (Pd >= 1.47) and (Pd < 2.60):
        a0 = 0.14
        b0 = 18.2
        c0 = -217.2
        p0 = 2.04
    elif (Pd >= 2.60) and (Pd < 4.90):
        a0 = 0.15
        b0 = 17.3
        c0 = -187.4
        p0 = 3.75
    elif (Pd >= 4.90) and (Pd < 9.90):
        a0 = 0.18
        b0 = 14.2
        c0 = -139.2
        p0 = 7.4

    else:
        raise ('Dynamic pressure none valid')

    a = a0 * np.cos(theta) ** 2 + np.sin(theta) ** 2
    dadt = 2 * np.cos(theta) * np.sin(theta) * (1 - a0)

    b = b0 * np.cos(theta) * (p0 / Pd) ** (1 / 6)
    dbdt = -b0 * np.sin(theta) * (p0 / Pd) ** (1 / 6)

    c = c0 * (p0 / Pd) ** (1 / 3)
    dcdt = 0

    delta = b ** 2 - 4 * a * c
    ddeltadt = 2 * b * dbdt - 4 * dadt * c

    u = -b + np.sqrt(delta)
    dudt = -dbdt + ddeltadt / (2 * np.sqrt(delta))

    v = 2 * a
    dvdt = 2 * dadt

    r = resolve_poly2_real_roots(a, b, c)[0]
    drdt = (dudt * v - dvdt * u) / v ** 2
    drdp = 0

    return derivative_spherical_to_cartesian(r, theta, phi, drdt, drdp)


def mp_sibeck1991_normal(theta, phi, **kwargs):
    tth, tph = mp_sibeck1991_tangents(theta, phi, **kwargs)
    return find_normal_from_tangents(tth, tph)


def mp_shue1998_tangents(theta, phi, **kwargs):
    theta = listify(theta)
    phi = listify(phi)
    Pd = kwargs.get("Pd", 2.056)
    Bz = kwargs.get("Bz", -0.001)

    r0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd ** (-1. / 6.6)
    a = (0.58 - 0.007 * Bz) * (1 + 0.024 * np.log(Pd))
    r = r0 * (2. / (1 + np.cos(theta))) ** a
    drdt = r0 * a * (2 ** a) * np.sin(theta) / (1 + np.cos(theta)) ** (a + 1)
    drdp = 0
    return derivative_spherical_to_cartesian(r, theta, phi, drdt, drdp)


def bs_jelinek2012_tangents(theta, phi, **kwargs):
    theta = listify(theta)
    phi = listify(phi)
    lamb = 1.17
    R = 15.02
    epsilon = 6.55
    Pd = kwargs.get('Pd', 2.056)

    R0 = 2 * R * Pd ** (-1 / epsilon)
    r = R0 / (np.cos(theta) + np.sqrt(np.cos(theta) ** 2 + (lamb * np.sin(theta)) ** 2))
    test = R0 * np.sin(theta) * (
        np.cos(theta) * (1 - lamb ** 2) / np.sqrt(np.cos(theta) ** 2 + (lamb * np.sin(theta)) ** 2)) / (
               np.cos(theta) + np.sqrt(np.cos(theta) ** 2 + (lamb * np.sin(theta)) ** 2)) ** 2

    u0 = R0
    v0 = np.cos(theta) + np.sqrt(np.cos(theta) ** 2 + (lamb * np.sin(theta)) ** 2)
    v0p = -np.sin(theta) + (np.cos(theta) * (-np.sin(theta)) + lamb ** 2 * np.sin(theta) * np.cos(theta)) / np.sqrt(
        np.cos(theta) ** 2 + (lamb * np.sin(theta)) ** 2)

    drdt = -u0 * v0p / v0 ** 2
    drdp = 0

    return derivative_spherical_to_cartesian(r, theta, phi, drdt, drdp)


def mp_shue1998_normal(theta, phi, **kwargs):
    tth, tph = mp_shue1998_tangents(theta, phi, **kwargs)
    return find_normal_from_tangents(tth, tph)


def bs_jelinek2012_normal(theta, phi, **kwargs):
    tth, tph = bs_jelinek2012_tangents(theta, phi, **kwargs)
    return find_normal_from_tangents(tth, tph)


_models =   {"mp_shue1998": mp_shue1998,
             "mp_shue1997": mp_shue1997,
             "mp_formisano1979": mp_formisano1979,
             "mp_liu2015": mp_liu2015,
             "mp_jelinek2012" : mp_jelinek2012,
             "mp_lin2010" : mp_lin2010,
             "mp_nguyen2020" : mp_nguyen2020,
             "mp_sibeck1991" : mp_sibeck1991,
             "bs_formisano1979": bs_formisano1979,
             "bs_jerab2005": bs_Jerab2005,
             "bs_jelinek2012": bs_jelinek2012
            }

_tangents = {"mp_shue1998" : mp_shue1998_tangents,
             "mp_sibeck1991" : mp_sibeck1991_tangents,
             "bs_jelinek2012" : bs_jelinek2012_tangents,
             "mp_shue1997": None,
             "mp_formisano1979": None,
             "mp_liu2015": None,
             "mp_jelinek2012": None,
             "mp_lin2010": None,
             "mp_nguyen2020": None,
             "bs_formisano1979": None,
             "bs_jerab2005": None
            }

_normal =  {"mp_shue1998" : mp_shue1998_normal,
            "mp_sibeck1991" : mp_sibeck1991_normal,
            "bs_jelinek2012" : bs_jelinek2012_normal,
            "mp_shue1997": None,
            "mp_formisano1979": None,
            "mp_liu2015": None,
            "mp_jelinek2012": None,
            "mp_lin2010": None,
            "mp_nguyen2020": None,
            "bs_formisano1979": None,
            "bs_jerab2005": None
           }


def _interest_points(model, **kwargs):
    dup = kwargs.copy()
    dup["base"] = "cartesian"
    x = model(0, 0, **dup)[0]
    y = model(np.pi / 2, 0, **dup)[1]
    xf = x - y ** 2 / (4 * x)
    return x, y, xf


def _parabolic_approx(theta, phi, x, xf, **kwargs):
    theta, phi = _checking_angles(theta, phi)
    K = x - xf
    a = np.sin(theta) ** 2
    b = 4 * K * np.cos(theta)
    c = -4 * K * x
    r = resolve_poly2(a, b, c)[0]
    return coords.BaseChoice(kwargs.get("base", "cartesian"), r, theta, phi)



def check_parabconfoc(func):
    def wrapper(self, theta, phi, **kwargs):
        kwargs["parabolic"] = kwargs.get("parabolic", False)
        kwargs["confocal"] = kwargs.get("confocal", False)
        if kwargs["parabolic"] is False and kwargs["confocal"] is True:
            raise ValueError("cannot be confocal if not parabolic")
        return func(self, theta, phi, **kwargs)
    return wrapper


class Magnetosheath:
    def __init__(self, **kwargs):
        kwargs["magnetopause"] = kwargs.get("magnetopause", "mp_shue1998")
        kwargs["bow_shock"] = kwargs.get("bow_shock", "bs_jelinek2012")

        if not kwargs["magnetopause"].startswith("mp_")  or not kwargs["bow_shock"].startswith("bs_"):
            raise ValueError(f"Invalid model name. Valid models are: {list(_models.keys())}")

        self._magnetopause = _models[kwargs["magnetopause"]]
        self._bow_shock = _models[kwargs["bow_shock"]]
        self.model_magnetopause = kwargs["magnetopause"]
        self.model_bow_shock = kwargs["bow_shock"]
        self._tangents_magnetopause = _tangents[kwargs["magnetopause"]]
        self._normal_magnetopause = _normal[kwargs["magnetopause"]]
        self._tangents_bow_shock = _tangents[kwargs["bow_shock"]]
        self._normal_bow_shock = _normal[kwargs["bow_shock"]]
        

    @check_parabconfoc
    def magnetopause(self, theta, phi, **kwargs):
        if kwargs["parabolic"]:
            return self._parabolize(theta, phi, **kwargs)[0]
        else:
            return self._magnetopause(theta, phi, **kwargs)


    @check_parabconfoc
    def bow_shock(self, theta, phi, **kwargs):
        if kwargs["parabolic"]:
            return self._parabolize(theta, phi, **kwargs)[1]
        else:
            return self._bow_shock(theta, phi, **kwargs)

    @check_parabconfoc
    def boundaries(self, theta, phi, **kwargs):
        if kwargs["parabolic"]:
            return self._parabolize(theta, phi, **kwargs)
        else:
            return self._magnetopause(theta, phi, **kwargs),\
                   self._bow_shock(theta, phi, **kwargs)

    def tangents_magnetopause(self, theta, phi, **kwargs):
        return self._tangents_magnetopause(listify(theta), listify(phi), **kwargs)

    def normal_magnetopause(self, theta, phi, **kwargs):
        return self._normal_magnetopause(listify(theta), listify(phi), **kwargs)


    def tangents_bow_shock(self, theta, phi,**kwargs):
        return self._tangents_bow_shock(listify(theta), listify(phi), **kwargs)

    def normal_bow_shock(self, theta, phi,**kwargs):
        return self._normal_bow_shock(listify(theta), listify(phi), **kwargs)


    def _parabolize(self, theta, phi, **kwargs):
        xmp, y, xfmp = _interest_points(self._magnetopause, **kwargs)
        xbs, y, xfbs = _interest_points(self._bow_shock, **kwargs)
        if kwargs.get("confocal", False) is True:
            xfmp = xmp / 2
            xfbs = xmp / 2
        mp_coords = _parabolic_approx(theta, phi, xmp, xfmp, **kwargs)
        bs_coords = _parabolic_approx(theta, phi, xbs, xfbs, **kwargs)
        return mp_coords, bs_coords
