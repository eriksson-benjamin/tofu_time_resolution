# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:09:00 2020

@author: bener807
"""

'''
Performs coincidence analysis for each laser intensity. Coincidences and
corresponding energies are saved to pickle file used in plot_coincidences.py
'''

import sys
sys.path.insert(0, 'C:/python/useful_definitions/')
import useful_defs as udfs
sys.path.insert(0, 'C:/python/TOFu/functions/')
import tofu_functions as dfs
import numpy as np

directories = np.array(['001', '002', '003', '004', '005',
                        '006', '007', '008', '009', '010',
                        '012', '014', '016', '018', '020',
                        '025', '030', '035', '040', '045',
                        '050', '055', '060', '065', '070',
                        '075', '080', '085', '090', '095',
                        '100'])

# Dictionaries
coincidence_dict = dfs.get_dictionaries('merged')
erg_sx_dict = dfs.get_dictionaries('merged')
sx_dict = dfs.get_dictionaries('merged')

for directory in directories:
    print(f'---- {directory} ----')

    # Get times/energies
    P = udfs.unpickle(f'data/decimated/10_30_ns/{directory}.pickle')
    erg_dict = P['energy']
    time_dict = P['time']

    # Synch times
    time_synch = time_dict['1kHz_CLK']

    for sx in sx_dict.keys():
        time_sx = time_dict[sx]
        energy_sx = erg_dict[sx]

        print(f'{sx}')

        # Find coincidences
        coincidences, [sx_args, synch_args] = dfs.sTOF4(time_sx, time_synch,
                                                        t_back=200,
                                                        t_forward=200,
                                                        return_indices=True)
        # Move coincidences to around zero
        coincidences -= np.median(coincidences)

        # Store in dictionaries
        coincidence_dict[sx] = np.append(coincidence_dict[sx], coincidences)
        erg_sx_dict[sx] = np.append(erg_sx_dict[sx], energy_sx[sx_args])

    # Synch vs. synch coincidences are saved for each laser intensity
    print('ABS_REF')
    coincidences, [sx_args, synch_args] = dfs.sTOF4(time_dict['ABS_REF'],
                                                    time_synch,
                                                    t_back=200,
                                                    t_forward=200,
                                                    return_indices=True)
    coincidences -= np.median(coincidences)

    # Save to file (move these to data/coincidences/abs_ref/)
    udfs.pickler(f'{directory}_abs_ref.pickle', coincidences)
    print('\n')

# Save to file (move these to data/coincidences/)
for sx in sx_dict.keys():
    to_pickle = {'coincidences': coincidence_dict[sx],
                 'energy': erg_sx_dict[sx]}
    udfs.pickler(f'{sx}.pickle', to_pickle)
