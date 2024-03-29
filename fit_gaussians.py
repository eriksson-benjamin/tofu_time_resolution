# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:21:14 2021

@author: bener807
"""

'''
Plots 2D TOF-energy histograms using the output from coincidences.py. Projects
the 2D histogram onto the TOF axis for a set of energy cuts. Fits a Gaussian
to the TOF spectra for each energy cut and saves the fit parameters to a
pickle file which is used in plot_sigma.py.
'''

import sys
import scipy.optimize as optimize
sys.path.insert(0, 'C:/python/useful_definitions/')
import useful_defs as udfs
sys.path.insert(0, 'C:/python/TOFu/functions/')
import tofu_functions as dfs
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
udfs.set_nes_plot_style()


def sort_data(sx, energy_bins, directory):
    """
    Return the sorted conicidences/energies for detector sx using energy bins.

    Note
    ----
    The binning is such that energy_bins[0] < c_sort[1] < energy_bins[1].
    """
    # Import coincidences
    p = udfs.unpickle(f'data/coincidences/{directory}/{sx}.pickle')
    all_coincidences = p['coincidences']
    all_energies = p['energy']

    # Remove pulses with negative energy and times outside +/- 10 ns
    boolean = ((all_energies > 0) &
               (all_coincidences < 10) &
               (all_coincidences > -10))
    coincidences = all_coincidences[boolean]
    energies = all_energies[boolean]

    # Normalize to zero
    coincidences -= np.mean(coincidences)

    # Sort into bins
    indices = np.digitize(energies, energy_bins)

    '''
    The first list (c_sort[0], e_sort[0]) contains any events below the lowest
    bin in energy_bins. The final list (c_sort[-1], e_sort[-1]) contains any
    events above the highest bin in energy_bins.
    '''
    n = len(energy_bins) + 1
    c_sort = [[coincidences[indices == ind]] for ind in range(n)]
    e_sort = [[energies[indices == ind]] for ind in range(n)]

    # Skip the first and last list (i.e. anything outside our energy binning)
    return c_sort[1:-1], e_sort[1:-1]


def bin_coincidences(coincidences, c_bins):
    """Bin the coincidence data for each energy cut."""
    c_binned = []
    for c_list in coincidences:
        binned, _ = np.histogram(c_list[0], c_bins)
        c_binned.append(binned)
    return c_binned


def plot_coincidences(c_binned, c_bin_centers, e_bin_centers, detector=''):
    """Plot histogrammed coincidences for each energy cut."""
    for counts, e_bin in zip(c_binned, e_bin_centers):
        plt.figure(f'{e_bin:.3f}')
        plt.plot(c_bin_centers, counts, 'k.')
        plt.errorbar(c_bin_centers, counts, yerr=np.sqrt(counts),
                     linestyle='None', color='k')
        plt.xlabel('time (ns)')
        plt.ylabel('counts')
        plt.title(f'{detector} {e_bin:.3f} MeVee', loc='right')


def plot_2d(coincidences, energies, c_bins, e_bins, title=''):
    """Plot 2D histogram spectrum."""
    # Flatten data
    c_all = np.array([])
    e_all = np.array([])
    for c_list, e_list in zip(coincidences, energies):
        c_all = np.append(c_all, c_list)
        e_all = np.append(e_all, e_list)

    # Plot 2D spectrum
    # ----------------
    plt.figure(title)

    # Set white background
    my_cmap = copy.copy(plt.cm.get_cmap('jet'))
    my_cmap.set_under('w', 1)

    # Plot 2D histogram
    plt.hist2d(c_all, e_all, bins=(c_bins, e_bins), cmap=my_cmap, vmin=1)

    # Set limits/labels/legends
    plt.xlabel('time (ns)')
    plt.ylabel('energy (MeV$_{ee}$)')
    plt.title(title, loc='left')
    plt.colorbar()


def gauss(x, a, b, c):
    """Return Gaussian."""
    return a * np.exp(-0.5 * ((x - b) / c)**2)


def fit_gaussians(coincidences, c_bin_centers):
    """Fit Gaussian to coincidences, return fit parameters with uncertainty."""
    parameters = []
    u_parameters = []
    for i, c_list in enumerate(coincidences):
        # If no coincidences
        if np.all(c_list == 0):
            pars = np.array([-1, -1, -1])
            u_pars = np.array([-1, -1, -1])

        # Otherwise fit Gaussian
        else:
            try:
                uncrt = np.sqrt(c_list)
                uncrt[uncrt == 0] = 1E8
                popt, pcov = optimize.curve_fit(gauss, c_bin_centers, c_list,
                                                sigma=uncrt,
                                                absolute_sigma=True)
                popt[2] = np.abs(popt[2])
                pars = popt
                u_pars = np.sqrt(np.diag(pcov))

            except Exception:
                pars = np.array([-1, -1, -1])
                u_pars = np.array([-1, -1, -1])
        parameters.append(pars)
        u_parameters.append(u_pars)
    return parameters, u_parameters


def plot_gaussians(parameters, x_axis, e_bin_centers, detector=''):
    """Plot Gaussian fit on top of coincidence plot (if available)."""
    for p, e_bin in zip(parameters, e_bin_centers):
        plt.figure(f'{e_bin:.3f}')
        gaussian = gauss(x_axis, *p)
        plt.plot(x_axis, gaussian, 'k')


def analyse_synch(c_bins, c_bin_centers, directory, plot=False):
    """
    Estimate the width of the synch vs. synch distribution.

    Note
    ----
    Energy slices are not available here since the synch signal has a square
    waveform with a constant amplitude.
    """
    path = f'data/coincidences/{directory}/abs_ref'
    files = os.listdir(path)

    stdev = np.array([])
    for file in files:
        # Import coincidences
        all_coincidences = udfs.unpickle(f'{path}/{file}')

        # Histogram data
        coincidences, _ = np.histogram(all_coincidences, c_bins)

        # Remove outliers and calculate standard deviation
        outlier_bool = ((all_coincidences > -0.2) & (all_coincidences < 0.2))
        stdev = np.append(stdev, np.std(all_coincidences[outlier_bool]))

        # Plot
        if plot:
            title = file.replace('.pickle', '').replace('_', '-')
            plt.figure(title)
            plt.plot(c_bin_centers, coincidences, 'k.')
            plt.errorbar(c_bin_centers, coincidences, linestyle='None',
                         color='k', yerr=np.sqrt(coincidences))
            plt.title(title, loc='right')
            plt.xlabel('time (ns)')
            plt.ylabel('counts')

    return stdev.mean(), np.std(stdev).mean()


def plot_for_paper(detector, directory):
    """Create plot for TOFu technical paper."""
    # Energy/time bins
    e_bins = np.arange(0, 4, 0.05)
    e_bin_centers = e_bins[1:] - np.diff(e_bins) / 2
    c_bins = np.arange(-2, 2, 0.05)
    c_bin_centers = c_bins[1:] - np.diff(c_bins) / 2

    # Sort data into energy bins
    coincidences, energies = sort_data(detector, e_bins, directory)

    # Bin coincidences
    c_binned = np.array(bin_coincidences(coincidences, c_bins))

    # Select bins to plot
    selected_bins = [0.275, 0.475, 0.675]

    linestyles = ['-', '--', '-.', ':']
    arg = [np.argwhere(sb == e_bin_centers.round(8))[0][0]
           for sb in selected_bins]

    # Plot a few energy slices
    fig_name = f'{detector} gaussians'
    fig = plt.figure(fig_name)
    colors = udfs.get_colors(len(selected_bins))
    to_zip = (c_binned[arg], e_bin_centers[arg], colors, linestyles)
    for counts, e_bin, color, ls in zip(*to_zip):
        # Normalize
        u_counts = np.sqrt(counts)
        u_counts = u_counts / counts.max()
        counts = counts / counts.max()

        plt.plot(c_bin_centers, counts, color=color, marker='.',
                 linestyle='None')
        plt.errorbar(c_bin_centers, counts, yerr=u_counts,
                     linestyle='None', color=color)
        plt.xlabel('$\Delta t$ (ns)')
        plt.ylabel('counts (a.u.)')

        # Fit gaussian
        popt, pcov = fit_gaussians([counts], c_bin_centers)
        x_axis = np.linspace(-1, 1, 1000)
        gaussian = gauss(x_axis, *popt[0])
        plt.plot(x_axis, gaussian, color=color, linestyle=ls,
                 label=f'{e_bin:.3f} MeV$_{{ee}}$')

    # Plot synch signal distribution
    s_name = f'data/coincidences/{directory}/abs_ref/001_abs_ref.pickle'
    synch = udfs.unpickle(s_name)

    # Clear outliers
    mask = ((synch > -0.2) & (synch < 0.2))
    synch = synch[mask]

    # Divide by sqrt(2) to find sigma for single synch channel
    synch_sigma = np.std(synch) / np.sqrt(2)

    plt.plot([-synch_sigma, synch_sigma], [0.3, 0.3], 'k', marker='|',
             linestyle='--')
    plt.text(-0.08, 0.34, f'$\pm${1000 * synch_sigma:.0f} ps')
    plt.legend()
    plt.xlim(-0.75, 0.75)
    plt.text(0.21, 0.85, '(a)', transform=fig.transFigure)


def main(detector, directory):
    """Fit Gaussians to coincidences and plot."""
    # Energy/time bins
    e_bins = np.arange(0, 4, 0.05)
    e_bin_centers = e_bins[1:] - np.diff(e_bins) / 2
    c_bins = np.arange(-2, 2, 0.05)
    c_bin_centers = c_bins[1:] - np.diff(c_bins) / 2

    # Sort data into energy bins
    coincidences, energies = sort_data(detector, e_bins, directory)

    # Plot 2D histogram
    plot_2d(coincidences, energies, c_bins, e_bins, detector.replace('_', '-'))

    # Bin coincidences
    c_binned = bin_coincidences(coincidences, c_bins)

    # Plot coincidences
    plot_coincidences(c_binned, c_bin_centers, e_bin_centers,
                      detector=detector.replace('_', '-'))

    # Fit Gaussians to energy cut coincidences
    parameters, u_parameters = fit_gaussians(c_binned, c_bin_centers)

    # Plot Gaussians
    x_axis = np.linspace(-2, 2, 1000)
    plot_gaussians(parameters, x_axis, e_bin_centers)

    return parameters, u_parameters, e_bin_centers


if __name__ == '__main__':
    # Settings
    save_plots = False
    save_param = True

    # Directory to read coincidences from
    directory = '26-11-2022-energy-calibration'

    plot_for_paper('S2_01', directory)

    # List of detector names
    detectors = list(dfs.get_dictionaries('merged').keys())
    for detector in detectors:
        parameters, u_parameters, e_bin_centers = main(detector, directory)

        # Save plots (move these to output/gaussians/figures/)
        if save_plots:
            udfs.multipage(f'{detector}_gauss.pdf', tight_layout=True)

        # Save parameters (move these to output/gaussians/fit_parameters/)
        if save_param:
            to_pickle = {'parameters': parameters,
                         'u_parameters': u_parameters}
            udfs.pickler(f'{detector}_gauss.pickle', to_pickle)

        # Close figures
        plt.close('all')
        print(f'{detector} done.')

    # Analyse synch vs. synch coincidences
    c_bins = np.arange(-0.2, 0.2, 0.005)
    c_bin_centers = c_bins[1:] - np.diff(c_bins) / 2
    stdev, u_stdev = analyse_synch(c_bins, c_bin_centers,
                                   directory, plot=False)

    # Save parameters and bins (move these to output/gaussians/fit_parameters/)
    if save_param:
        to_pickle = {'parameters': stdev, 'u_parameters': u_stdev}
        udfs.pickler('ABS_REF.pickle', to_pickle)
        udfs.pickler('e_bin_centers.pickle', e_bin_centers)
