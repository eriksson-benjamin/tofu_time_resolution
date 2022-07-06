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
sys.path.insert(0, 'C:/python/definitions/')
import useful_defs as udfs
import definitions_heimdall as dfs
import numpy as np
import matplotlib.pyplot as plt
import scipy
from inspect import signature
# Suppress warnings
np.seterr(invalid='ignore')
udfs.set_nes_plot_style()


def get_outliers(input_file):
    # List of hardcoded outliers
    if input_file == 'fit_parameters_10_30.pickle':
        outliers = {'S1_01':[2.3],
                    'S2_03':[2.1],
                    'S2_05':[2.3],
                    'S2_12':[2.1],
                    'S2_21':[1.9],
                    'S2_24':[2.1, 2.3],
                    'S2_28':[1.9],
                    'S2_32':[2.3]
                    }
    else: raise NameError(f'Unknown input file name: {input_file}')
    return outliers


def conf_level(n_std, popt, pcov, x_axis):
    # Prepare confidence level curves
    popt_up = popt + n_std*np.sqrt(np.diag(pcov))
    popt_dw = popt - n_std*np.sqrt(np.diag(pcov))
    
    lower_bound = fit_function(x_axis, *popt_dw)
    upper_bound = fit_function(x_axis, *popt_up)
    
    return lower_bound, upper_bound

def calculate_resolution(detector):
    # Import fit parameters
    path = 'output/gaussians/fit_parameters'
    p = udfs.unpickle(f'{path}/{detector}_gauss.pickle')
    s = udfs.unpickle(f'{path}/ABS_REF.pickle')
    
    # Merge parameters to single array
    res = np.array([res[2] for res in p['parameters']])
    u_res = np.array([u_res[2] for u_res in p['u_parameters']])
    
    # Divide by sqrt(2) for contribution from one synch signal
    synch = s['parameters']/np.sqrt(2)
    u_synch = s['u_parameters']/np.sqrt(2)

    # Remove synch contribution to resolution to get the detector resolution
    det_res = np.sqrt(res**2 - synch**2)
    
    # Calculate propagated uncertainty
    u_det_res = np.sqrt(1/(res**2 - synch**2) * ((res*u_res)**2 + (synch*u_synch)**2))
    
    return det_res, u_det_res
    

def plot_resolution(res, u_res, e_bin_centers, detector, all_data=False):
    '''
    Plots the energy sliced time resolution of the detectors with the fit.
    '''
    fig = plt.figure(detector)
    # plt.title(detector.replace('_', '-'), loc='left')
    
    # Perform fit    
    popt, pcov = fit_resolution(res, u_res, e_bin_centers, detector)
    
    # Only plot the fitting range?
    if not all_data:
        f_bool = select_fit_range(detector, e_bin_centers)
        res = res[f_bool]
        u_res = u_res[f_bool]
        e_bin_centers = e_bin_centers[f_bool]
        plt.ylim(0.8*res.min(), 1.1*res.max())
        
    # Plot data
    plt.plot(e_bin_centers, res, 'k.')
    plt.errorbar(e_bin_centers, res, yerr=u_res, linestyle='None', color='k')
    
    plt.xlabel('energy (MeV$_{ee}$)')
    plt.ylabel(r'$\sigma$ (ns)')
 
    # Plot fit
    x_axis = np.linspace(0.05, 4, 1000)
    y_axis = fit_function(x_axis, *popt)
    plt.figure(detector)
    plt.plot(x_axis, y_axis, 'k', label='best fit curve')
    
    # Calculate confidence levels
    l1, u1 = conf_level(1, popt, pcov, x_axis)
    l2, u2 = conf_level(2, popt, pcov, x_axis)
    
    plt.fill_between(x_axis, u2, l2, alpha=0.6, color='lightskyblue', 
                     label='2$\sigma$ interval')
    plt.fill_between(x_axis, u1, l1, alpha=0.8, color='C0', 
                     label='1$\sigma$ interval')
    
    # Change order of legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])    
    plt.text(0.21, 0.85, '(b)', transform=fig.transFigure);

def get_fit_range(detector):
    ranges = np.loadtxt('output/time_resolution/fit_range.txt', dtype='str')
    r_d = {r[0]:[float(r[1]), float(r[2])] for r in ranges}
    return r_d[detector]


def select_fit_range(detector, e_bin_centers):
    r = get_fit_range(detector)
    e_bin_centers = np.round(e_bin_centers, 8)
    fit_bool = ((e_bin_centers >= r[0]) & (e_bin_centers <= r[1]))
    return fit_bool


def fit_function(x, a, b, c):
    return np.sqrt(a**2 + b**2/x + c**2/x**2)


def fit_resolution(res, u_res, e_bin_centers, detector):
    # Select fit range
    r = get_fit_range(detector)
    e_bin_centers = np.round(e_bin_centers, 8)
    fit_bool = ((e_bin_centers >= r[0]) & (e_bin_centers <= r[1]))
    y = res[fit_bool]
    u_y = u_res[fit_bool]
    x = e_bin_centers[fit_bool]
    
    # Set bounds
    n_args = len(signature(fit_function).parameters.keys()) - 1
    bounds = (n_args*[0], n_args*[np.inf])
    popt, pcov = scipy.optimize.curve_fit(fit_function, x, y, sigma=u_y, 
                                          absolute_sigma=True, bounds=bounds)
    
    return popt, pcov


if __name__ == '__main__':    
    # Get detector names
    detectors = dfs.get_dictionaries('merged')
    
    # Import the energy bins
    e_bin_centers = udfs.unpickle('output/gaussians/fit_parameters/e_bin_centers.pickle')
    
    # Create parameter file
    with open('fit_parameters.txt', 'w') as handle:
        handle.write('# Det  a      b      c      u_a    u_b    u_c\n')
        
    # Calculate resolution, perform fit, and plot
    for detector in detectors:
        res, u_res = calculate_resolution(detector)
        plot_resolution(res, u_res, e_bin_centers, detector, all_data=False)
        popt, pcov = fit_resolution(res, u_res, e_bin_centers, detector)
        u_popt = np.sqrt(np.diag(pcov))

        # Write fit parameters to file
        line = (f'{detector}  {popt[0]:.4f} {popt[1]:.4f} {popt[2]:.4f} '
                f'{u_popt[0]:.4f} {u_popt[1]:.4f} {u_popt[2]:.4f}\n')
        with open('fit_parameters.txt', 'a') as handle:
            handle.write(line)
    
        print(f'{detector} done')

