Four scripts used to perform the analysis to determine the time resolution of the TOFu system using laser data acquired at JET.

Scripts are run in the following order to produce the time resolution fit:
1. generate_data.py 
2. generate_coincidences.py 
3. fit_gaussians.py 
4. fit_time_resolution.py

1. generate_data.py
-------------------
Analysis of the laser (and synch) raw pulse waveforms located in data/raw_data/ to produce time stamps and determine the corresponding energies of each pulse. Input data is saved locally on PC as well as on hard-drive backup. Output is saved to data/decimated/10_30_ns/.

2. generate_coincidences.py
---------------------------
Finds coincidences between the synch pulses and all the S1/S2 detectors using the output from generate_data.py. The coincidences are stored in data/coincidences/. One file for each S1/S2 detector is stored. The synch vs. synch coincidences are stored one level down in abs_ref/. 

3. fit_gaussians.py
-------------------
Fits Gaussians to the coincidence spectra found in generate_coincidences.py. Coincidence spectra for different energy slices are plotted. The width of the Gaussian decreases as the energy (integrated charge) of the pulses increases. We thus see that the time resolution of our detectors improves with higher energies. Gaussian fit parameters are saved in output/gaussians/fit_parameters/ for each S1/S2 vs. synch combination. The synch vs. synch combination is aved to ABS_REF.pickle.

4. fit_time_resolution.py
-------------------------
Takes the output from fit_gaussians.py and fits a function on the form sqrt(a^2 + b^2/x + c^2/x^2) to the widths of the Gaussians as a function of energy. Calculates the 1- and 2-sigma confidence levels of the fit. Fit parameters (a^2, b^2, and c^2) are saved to output/time_resolution/fit_parameters.txt. Note that a^2, b^2, and c^2 are saved, not a, b, and c.


Notes
-----
The input data (~15 GB) is not uploaded to Github. Data is available upon request to benjamin.eriksson@physics.uu.se.
