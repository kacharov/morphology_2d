#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:45:18 2023

@author: kacharov
"""

import numpy as np
from scipy.integrate import simps

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


def argnearest(array, value):
    """
    Finds the index of the nearest value in an array to the provided value.

    Parameters
    ----------
    array : array_like
        An array for which the index of the nearest value 
        to the provided value is to be found.

    value : float or int
        The value to which the nearest value's index is to be found.

    Returns
    -------
    idx : int
        The index of the nearest value in the array to the provided value.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([0, 1, 2, 3, 4, 5])
    >>> argnearest(arr, 3.6)
    4

    Notes
    -----
    The function converts the input array to a NumPy array for processing.

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plummer1d(r, rs=1):
    """
    Computes the 1D projected Plummer model at a specified radius.

    Parameters
    ----------
    r : float
        The radius at which the Plummer model is to be evaluated.

    rs : float, optional
        The scale radius that characterizes the size of the model, by default 1.

    Returns
    -------
    density : float
        The mass density of the Plummer model at the specified radius.

    Examples
    --------
    >>> plummer1d(0.5)
    1.0

    >>> plummer1d(2, rs=2)
    0.2

    Notes
    -----
    The larger the scale radius, the more extended the model.

    """

    return (1+(r/rs)**2)**(-2)


def plummer2d(x, y, rs=1, q=1, theta=0, x0=0, y0=0):
    """
    Computes the 2D Plummer model at a specified position.

    Parameters
    ----------
    x, y : float
        The coordinates at which the Plummer model is to be evaluated.

    rs : float, optional
        The scale radius that characterizes the size of the model, by default 1.

    q : float, optional
        The axial ratio of the model, by default 1 (spherical).

    theta : float, optional
        The position angle in radians, measured counter-clockwise from the x-axis,
        by default 0.

    x0, y0 : float, optional
        The coordinates of the center of the model, by default 0, 0.

    Returns
    -------
    density : float
        The mass density of the Plummer model at the specified position.

    Examples
    --------
    >>> plummer2d(0.5, 0.5)
    0.8

    >>> plummer2d(2, 2, rs=2, q=0.5, theta=np.pi/4)
    0.2

    Notes
    -----
    The larger the scale radius, the more extended the model.
    The coordinates are transformed and rotated as per the user's input
    before evaluating the model.

    """
    xc = x - x0
    yc = y - y0

    # rotate the cluster at the desired position angle
    xp = xc * np.cos(theta) + yc * np.sin(theta)
    yp = xc * np.sin(theta) - yc * np.cos(theta)

    # calculate an elliptical radius
    r = np.sqrt(xp**2 + (yp/q)**2)

    return (1+(r/rs)**2)**(-2)


def plummer2d_int(rs=1, q=1):
    """
    Computes the integral of an elliptical 2D Plummer model over 2*np.pi*q*r*dr.

    Parameters
    ----------
    rs : float, optional
        The scale radius that characterizes the size of the model, by default 1.

    q  : float, optional
        The axial ratio of the model, by default 1 (spherical).

    Returns
    -------
    integrated_density : float
        The integral of the spherical Plummer model over the entire space.

    Examples
    --------
    >>> plummer2d_int(1)
    3.141592653589793

    >>> plummer2d_int(2)
    12.566370614359172

    """
    return np.pi*q*rs**2


def plummer2d_cdf(r, rs=1):
    """
    Computes the cumulative distribution function (CDF) of a spherical
    2D Plummer model over r*dr.

    Parameters
    ----------
    r : float
        The radius at which the CDF of the Plummer model is computed.

    rs : float, optional
        The scale radius that characterizes the size of the model, by default 1.

    Returns
    -------
    cdf_value : float
        The CDF value of the Plummer model at the specified radius.

    Examples
    --------
    >>> plummer2d_cdf(0.5)
    0.2

    >>> plummer2d_cdf(2, rs=2)
    0.5

    Notes
    -----
    The CDF is calculated as r² / (rs² + r²), giving the cumulative distribution
    of surface density within a given radius in the Plummer model.

    """
    return r**2 / (rs**2 + r**2)


def plummer2d_icdf(a, rs=1):
    """
    Computes the inverse cumulative distribution function (iCDF) of a 
    projected 2D Plummer model over r*dr.

    Parameters
    ----------
    a : float
        The value at which the iCDF of the Plummer model is computed.
        'a' must be in the range [0,1).

    rs : float, optional
        The scale radius that characterizes the size of the model, by default 1.

    Returns
    -------
    icdf_value : float
        The iCDF value of the Plummer model at the specified value 'a'.

    Examples
    --------
    >>> plummer2d_icdf(0.5)
    0.7071067811865476

    >>> plummer2d_icdf(0.2, rs=2)
    0.8944271909999159

    Notes
    -----
    The iCDF is calculated as rs * √(a/(1-a)), inverting the cumulative distribution
    function of the Plummer model.

    """
    return rs * (a/(1-a))**0.5


def gen_mock_plummer_cluster(N, rs=1, q=1, theta=0, background=0, x0=0, y0=0,
                             fov=[-10, 10, -10, 10], plot=True):
    """
    Generates a mock Plummer cluster with a certain number of stars.

    Parameters
    ----------
    N : int
        The number of stars in the cluster.

    rs : float, optional
        The scale radius that characterizes the size of the model.

    q : float, optional
        The axial ratio of the model.

    theta : float, optional
        The position angle in radians, measured counter-clockwise from the x-axis.

    background : float, optional
        The number of uniformly distributed background stars as
        a fraction of N, by default 0.

    x0, y0 : float, optional
        The coordinates of the center of the model, by default 0, 0.

    fov : list of float, optional
        The field of view in the format [xmin, xmax, ymin, ymax].

    plot : bool, optional
        If True, the function will plot the 2D distribution and 1D density profile of the cluster.

    Returns
    -------
    x, y : array_like
        The x and y coordinates of the stars in the cluster.

    Examples
    --------
    >>> x, y = gen_mock_plummer_cluster(1000, plot=False)
    >>> len(x), len(y)
    1000, 1000

    Notes
    -----
    The function first generates the radial distribution of the cluster, and then converts
    this into cartesian coordinates. It also has an option to generate and add a uniform
    background distribution of stars to the final model. The function returns the star
    positions within the defined field of view (FoV).

    """

    # generate the radial distribution
    a = np.random.uniform(0, 1, N)
    r = plummer2d_icdf(a, rs=rs)

    # generate a uniform distribution of position angles
    phi = np.random.uniform(0, 2*np.pi, N)

    # generate cartesian coordinates
    xp = r * np.cos(phi)
    yp = q * r * np.sin(phi)

    # rotate the cluster at the desired position angle
    x = xp * np.cos(theta) - yp * np.sin(theta)
    y = xp * np.sin(theta) + yp * np.cos(theta)

    # generate uniform background sample going out to 10*rs
    r_b = np.sqrt(np.random.uniform(0, 1, np.int64(
        background * N))) * 2 * np.max(np.abs(fov))
    phi_b = np.random.uniform(0, 2*np.pi, np.int64(background * N))

    x_b = r_b * np.cos(phi_b)
    y_b = r_b * np.sin(phi_b)

    x_full = np.concatenate((x, x_b)) + x0
    y_full = np.concatenate((y, y_b)) + y0

    # cut the star to a defined FoV
    x = x_full[(x_full > fov[0]) & (x_full < fov[1]) &
               (y_full > fov[2]) & (y_full < fov[3])]
    y = y_full[(x_full > fov[0]) & (x_full < fov[1]) &
               (y_full > fov[2]) & (y_full < fov[3])]

    if plot:
        # plot the 2D distribution
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.scatter(x, y, s=2, alpha=0.2)
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        # plot the 1D density profile
        nbin = 100
        rbin = np.logspace(np.log10(0.1*rs), np.log10(10*rs), nbin)
        rho = np.zeros(nbin)
        rho[0] = len(r[r < rbin[0]]) / (np.pi*rbin[0]**2)
        for i in range(1, nbin):
            rho[i] = len(r[(r < rbin[i]) & (r >= rbin[i-1])]) / \
                (np.pi*(rbin[i]**2 - rbin[i-1]**2))

        plt.figure(figsize=(10, 10))
        plt.axvline(rs, linestyle='--', color='k')
        plt.scatter(rbin, rho, color='b')
        plt.plot(rbin, rho[argnearest(rbin, rs)]/plummer1d(rs,
                                                           rs=rs) * plummer1d(rbin, rs=rs), color='r')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('R')
        plt.ylabel('Number Density')
        plt.show()

    return x, y


def lnprob_big_fov(pos, x, y):
    """
    Computes the log probability for a given set of parameters, assuming a Plummer model.

    This function is usually used in a Bayesian data analysis context, and it computes the
    natural logarithm of the posterior probability distribution for a given set of parameters.

    This function uses analytical integrals to normalise the likelihood over the FoV,
    so it is assumed that the entire cluster is contained within the FoV. It will
    create a biased result if there are many cluster stars outside the FoV, but it is
    fast.

    Parameters
    ----------
    pos : array_like
        An array containing the values of the parameters. The order of parameters is:
        [scale radius, axis ratio, position angle, centre x coordinate, 
        centre y coordinate, background fraction].

    Returns
    -------
    log_prob : float
        The logarithm of the probability of the given model parameters.

    Notes
    -----
    If any of the parameters are outside the defined priors, the function returns negative
    infinity (-inf), as required for log-probability functions used in Bayesian analysis.
    The function first checks the validity of the parameters, then calculates the likelihood
    for the observed stars given the model parameters. It then returns the sum of the log
    probabilities.

    """

    rs = pos[0]  # scale radius
    q = pos[1]  # axis ratio
    theta = pos[2]  # position angle
    x0 = pos[3]  # centre x coordinate
    y0 = pos[4]  # centre y coordinate
    bg = pos[5]  # background fraction

    if (rs < 0.) or (rs > 50.) or \
       (q < 0.2) or (q > 1.1) or \
       (theta < 0) or (theta > np.pi/2) or \
       (bg < 0) or (bg > 1):
        return(-np.inf)

    xc = x - x0
    yc = y - y0

    # rotate the cluster at the desired position angle
    xp = xc * np.cos(theta) + yc * np.sin(theta)
    yp = xc * np.sin(theta) - yc * np.cos(theta)

    # calculate an elliptical radius
    r = np.sqrt(xp**2 + (yp/q)**2)

    # caclulate the FoV area
    fov_area = (np.max(x)-np.min(x)) * (np.max(y) - np.min(y))

    # calculate the likelihood per star
    p = (plummer1d(r, rs=rs) + bg) / (plummer2d_int(rs, q) + bg*fov_area)

    return sum(np.log(p))


def lnprob_small_fov(pos, x, y):
    """
    Computes the log probability for a given set of parameters, assuming a Plummer model.

    This function is usually used in a Bayesian data analysis context, and it computes the
    natural logarithm of the posterior probability distribution for a given set of parameters.

    This function uses numerical integrals over a defined meshgrid to normalise
    the likelihood over the FoV. It is slower, but more versitlie, as it can be
    easily extended to density distribution that cannot be integrated analytically.

    Parameters
    ----------
    pos : array_like
        An array containing the values of the parameters. The order of parameters is:
        [scale radius, axis ratio, position angle, centre x coordinate, 
        centre y coordinate, background fraction].

    Returns
    -------
    log_prob : float
        The logarithm of the probability of the given model parameters.

    Notes
    -----
    If any of the parameters are outside the defined priors, the function returns negative
    infinity (-inf), as required for log-probability functions used in Bayesian analysis.
    The function first checks the validity of the parameters, then calculates the likelihood
    for the observed stars given the model parameters. It then returns the sum of the log
    probabilities.

    """

    rs = pos[0]  # scale radius
    q = pos[1]  # axis ratio
    theta = pos[2]  # position angle
    x0 = pos[3]  # centre x coordinate
    y0 = pos[4]  # centre y coordinate
    bg = pos[5]  # background fraction

    if (rs < 0.) or (rs > 50.) or \
       (q < 0.2) or (q > 1.1) or \
       (theta < 0) or (theta > np.pi/2) or \
       (bg < 0) or (bg > 1):
        return(-np.inf)

    # get an integration grid
    nbin = 300  # increase this value for a more precise integration (slower)
    xbin, ybin = np.meshgrid(np.linspace(min(x), max(x), nbin),
                             np.linspace(min(y), max(y), nbin))
    f = plummer2d(xbin, ybin, rs=rs, q=q, theta=theta, x0=x0, y0=y0) + bg
    # evaluate the integral
    int_f = simps(simps(f, ybin[:, 0]), xbin[0, :])

    p = (plummer2d(x, y, rs=rs, q=q, theta=theta, x0=x0, y0=y0) + bg) / int_f

    return sum(np.log(p))
