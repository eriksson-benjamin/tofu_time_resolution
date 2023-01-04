# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:07:57 2022

@author: bener807
"""

'''
Plots the widths of the Gaussians fit to the TOF spectra for the different
energy cuts found in plot_coincidences.py as a function of energy. A fit is 
applied to find timing resolution as a function of energy.
'''

import sys
sys.path.insert(0, 'C:/python/useful_definitions/')
import useful_defs as udfs
sys.path.insert(0, 'C:/python/TOFu/functions/')
import tofu_functions as dfs
import numpy as np
import matplotlib.pyplot as plt
import scipy
from inspect import signature
# Suppress warnings
np.seterr(invalid='ignore')
udfs.set_nes_plot_style()


def get_outliers(input_file):
    """Return list of outliers. Used only during testing."""
    # List of hardcoded outliers
    if input_file == 'fit_parameters_10_30.pickle':
        outliers = {'S1_01': [2.3],
                    'S2_03': [2.1],
                    'S2_05': [2.3],
                    'S2_12': [2.1],
                    'S2_21': [1.9],
                    'S2_24': [2.1, 2.3],
                    'S2_28': [1.9],
                    'S2_32': [2.3]
                    }
    else:
        raise NameError(f'Unknown input file name: {input_file}')
    return outliers


def draw_samples(popt, pcov, n):
    """Draw n random samples from posterior distribution of fit params."""
    stdev = np.sqrt(np.diag(pcov))
    samples = []
    for param, std in zip(popt, stdev):
        p = np.random.normal(param, std, n)
        samples.append(p)

    return np.array(samples).T


def conf_level(n_std, popt, pcov, x_axis):
    """
    Calculate confidence bands.

    Note
    ----
    Multiple confidence bands are returned if n_std is an array of values.
    """
    # Set very small values to zero
    mask = popt < 1E-5

    # Copy parameters/covariance matrix
    popt_c = np.copy(popt)
    pcov_c = np.copy(pcov)
    if np.any(mask):
        print('Parameter value smaller than 1E-5 detected.')
        print(f'popt = {popt}')
        print(f'pcov = \n{pcov}')
        print('Setting value to zero for confidence bands.\n')
        popt_c[mask] = 0
        pcov_c[np.diagflat(mask)] = 0

    # Draw n random samples from posterior
    samples = draw_samples(popt_c, pcov_c, 2000)
    if np.any(samples < 0):
        print('Warning: negative parameter samples detected.')

    # Run through fit function
    p0 = np.expand_dims(samples[:, 0], axis=1)
    p1 = np.expand_dims(samples[:, 1], axis=1)
    p2 = np.expand_dims(samples[:, 2], axis=1)

    # Generate curves
    curves = fit_function(x_axis, p0, p1, p2)
    best_fit = fit_function(x_axis, *popt)

    n_std = np.expand_dims(n_std, axis=1)
    lower_bound = best_fit - n_std * np.std(curves, axis=0)
    upper_bound = best_fit + n_std * np.std(curves, axis=0)

    return lower_bound, upper_bound


def calculate_resolution(detector):
    """
    Calculate resolution from output files.

    Note
    ----
    The contribution from synch vs. synch is removed here.
    """
    # Import fit parameters
    path = 'output/gaussians/fit_parameters'
    p = udfs.unpickle(f'{path}/{detector}_gauss.pickle')
    s = udfs.unpickle(f'{path}/ABS_REF.pickle')

    # Merge parameters to single array
    res = np.array([res[2] for res in p['parameters']])
    u_res = np.array([u_res[2] for u_res in p['u_parameters']])

    # Divide by sqrt(2) for contribution from one synch signal
    synch = s['parameters'] / np.sqrt(2)
    u_synch = s['u_parameters'] / np.sqrt(2)

    # Remove synch contribution to resolution to get the detector resolution
    det_res = np.sqrt(res**2 - synch**2)

    # Calculate propagated uncertainty
    u_det_res = np.sqrt(1 / (res**2 - synch**2) * ((res * u_res)**2
                                                   + (synch * u_synch)**2))

    return det_res, u_det_res


def plot_resolution(res, u_res, e_bin_centers, detector, all_data=False):
    """Plot the energy sliced time resolution of the detectors with the fit."""
    fig = plt.figure(detector)
    ax = plt.gca()
    # Perform fit
    popt, pcov = fit_resolution(res, u_res, e_bin_centers, detector)

    # Only plot the fitting range?
    if not all_data:
        f_bool = select_fit_range(detector, e_bin_centers)
        res = res[f_bool]
        u_res = u_res[f_bool]
        e_bin_centers = e_bin_centers[f_bool]
        plt.ylim(0.8 * res.min(), 1.1 * res.max())

    # Plot data
    ax.plot(e_bin_centers, res, 'k.', markersize=1.5)
    ax.errorbar(e_bin_centers, res, yerr=u_res, linestyle='None', color='k')

    ax.set_xlabel('light yield (MeV$_{ee}$)')
    ax.set_ylabel(r'$\sigma_{\Delta t}$ (ns)')

    # Plot fit
    x_axis = np.linspace(0.05, 4, 1000)
    y_axis = fit_function(x_axis, *popt)
    ax.plot(x_axis, y_axis, 'k', label='best fit curve')

    # Calculate confidence levels
    lower_bounds, upper_bounds = conf_level([1, 2], popt, pcov, x_axis)

    # Plot 1/2 sigma bands
    ax.fill_between(x_axis, upper_bounds[1], lower_bounds[1], alpha=0.6,
                    color='lightskyblue', label='2$\sigma$ interval')
    ax.fill_between(x_axis, upper_bounds[0], lower_bounds[0], alpha=0.8,
                    color='C0', label='1$\sigma$ interval')

    # Change order of legend
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    ax.text(0.21, 0.85, '(b)', transform=fig.transFigure)
    ax.set_xlim(-0.3, 3)
    ax.set_ylim(0.2, 0.8)
    ax.set_title(detector, loc='left')


def get_fit_range(detector):
    """Get fit range from file."""
    ranges = np.loadtxt('output/time_resolution/fit_range.txt', dtype='str')
    r_d = {r[0]: [float(r[1]), float(r[2])] for r in ranges}
    return r_d[detector]


def select_fit_range(detector, e_bin_centers):
    """Return bins corresponding to fit range."""
    r = get_fit_range(detector)
    e_bin_centers = np.round(e_bin_centers, 8)
    fit_bool = ((e_bin_centers >= r[0]) & (e_bin_centers <= r[1]))
    return fit_bool


def fit_function(x, a, b, c):
    """Return fit function."""
    return np.sqrt(a**2 + b**2 / x + c**2 / x**2)


def fit_resolution(res, u_res, e_bin_centers, detector):
    """Fit curve to time resolution."""
    # Select fit range
    r = get_fit_range(detector)
    e_bin_centers = np.round(e_bin_centers, 8)
    fit_bool = ((e_bin_centers >= r[0]) & (e_bin_centers <= r[1]))
    y = res[fit_bool]
    u_y = u_res[fit_bool]
    x = e_bin_centers[fit_bool]

    # Set bounds
    n_args = len(signature(fit_function).parameters.keys()) - 1
    bounds = (n_args * [0], n_args * [np.inf])
    popt, pcov = scipy.optimize.curve_fit(fit_function, x, y, sigma=u_y,
                                          absolute_sigma=True, bounds=bounds)

    return popt, pcov


if __name__ == '__main__':
    # Get detector names
    detectors = dfs.get_dictionaries('merged')

    # Import the energy bins
    p_name = 'output/gaussians/fit_parameters/e_bin_centers.pickle'
    e_bin_centers = udfs.unpickle(p_name)

    # Create parameter file (move this to output/time_resolution/)
    with open('fit_parameters.txt', 'w') as handle:
        handle.write('# Det  a      b      c      u_a    u_b    u_c\n')

    # Calculate resolution, perform fit, and plot
    for detector in detectors:
        res, u_res = calculate_resolution(detector)
        plot_resolution(res, u_res, e_bin_centers, detector, all_data=False)
        popt, pcov = fit_resolution(res, u_res, e_bin_centers, detector)
        u_popt = np.sqrt(np.diag(pcov))

        # Write fit parameters to the file
        line = (f'{detector}  {popt[0]:.4f} {popt[1]:.4f} {popt[2]:.4f} '
                f'{u_popt[0]:.4f} {u_popt[1]:.4f} {u_popt[2]:.4f}\n')
        with open('fit_parameters.txt', 'a') as handle:
            handle.write(line)

        print(f'{detector} done')
