# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:57:44 2020

@author: bener807
"""

import sys
sys.path.insert(0, 'C:/python/definitions/')
import definitions_heimdall as dfs
import useful_defs as udfs
import numpy as np
import matplotlib.pyplot as plt

'''
Performs time pickoff analysis and gives energies for the given pulses. Saves
times and energies to file to be used in coincidences.py.
'''

directories = np.array(['001', '002', '003', '004', '005', 
                        '006', '007', '008', '009', '010',
                        '012', '014', '016', '018', '020',
                        '025', '030', '035', '040', '045',
                        '050', '055', '060', '065', '070',
                        '075', '080', '085', '090', '095',
                        '100'])

# Create dictionaries/get board and channel names
time_dict = dfs.get_dictionaries('full')
erg_dict  = dfs.get_dictionaries('full')
boards = dfs.get_boards()
channels = dfs.get_channels()
timer_level = 0

for directory in directories:
    path = f'data/raw_data/{directory}/'
    print(f'----{directory}----')
    for boa in boards:
        for cha in channels:
            
            sx = dfs.get_detector_name(board=boa,  channel=cha)
            if sx not in ['ABS_REF', '1kHz_CLK']: continue
            print(f'{boa}{cha} - {sx}')
            
            # Set record length to 56/64 depending on ADQ412/ADQ14
            if int(boa) > 5: 
                rec_len  = 56
                bias     = 1600 
                trig_lvl = 1500
            else: 
                rec_len  = 64
                bias     = 27000
                trig_lvl = 26000
            
            # Import pulses/times
            pulse_data   = udfs.import_pulses(board=boa, channel=cha, 
                                              path=path, record_length=rec_len)
            time_stamps  = udfs.import_times(board=boa, channel=cha, path=path)
             
            # Remove all clipped pulses
            if sx not in ['ABS_REF', '1kHz_CLK']:
                clipped_args = np.where(np.min(pulse_data, axis=1)>-0.95*bias)[0]
                pulse_data = pulse_data[clipped_args]
                time_stamps = time_stamps[clipped_args]
            
            # Find adjustment to number of pre trigger samples for pulse data
            '''
            We ask for 16 pre trigger samples but receive anything between 
            16-19  (feature of the fwpd_disk_stream software), but since we 
            know the trigger level used we can find the number of pre trigger 
            samples.
            '''
            
            # Get pre trigger adjustment
            pre_trig_adjustment = dfs.find_threshold(pulse_data, 
                                                     trig_level=trig_lvl, 
                                                     timer=timer_level)            
            
            # Find where pulses trigger as normal
            normal_trig = np.where(pre_trig_adjustment > 16-1)[0] 
            
            # Remove oddly triggering pulses
            time_stamps = time_stamps[normal_trig]
            pulse_data = pulse_data[normal_trig]
            pre_trig_adjustment = pre_trig_adjustment[normal_trig]
            
            # Baseline reduction
            pulse_data_bl = dfs.baseline_reduction(pulse_data, timer=timer_level)
            
            # Remove junk pulses and corresponding times
            if sx not in ['ABS_REF', '1kHz_CLK']:
                bl_cut = np.array([[-150, 150], [-np.inf, 100]])
                pulse_data_bl, bad_indices = dfs.cleanup(pulses=pulse_data_bl, 
                                                         dx=1, bias_level=bias,
                                                         detector_name=sx, 
                                                         board=boa, channel=cha,
                                                         baseline_cut=bl_cut)
                pulse_data = np.delete(pulse_data, bad_indices, axis=0)
                time_stamps = np.delete(time_stamps, bad_indices, axis=0)
                pre_trig_adjustment = np.delete(pre_trig_adjustment, 
                                                bad_indices, axis=0)

            # Remove synch pulses with bad pulse heights
            if sx == 'ABS_REF':
                minima = np.min(pulse_data, axis=1)
                pulse_data =  pulse_data[np.where(minima < -1500)[0]]
            if sx == '1kHz_CLK':
                minima = np.min(pulse_data, axis=1)
                pulse_data = pulse_data[np.where(minima < 0)[0]]
            
            # Set up x-axes for sinc interpolation
            u_factor = 10
            record_length = np.shape(pulse_data)[1]
            x_axis = np.arange(0, record_length)
            ux_axis = np.arange(0, record_length, 1./u_factor)
            
            # Perform sinc-interpolation
            pulse_data_sinc = dfs.sinc_interpolation(pulse_data_bl, x_axis, 
                                                     ux_axis, timer=timer_level)
        
            # Get area under pulse between 10-30 ns
            pulse_area = dfs.get_pulse_area(pulse_data_sinc[:, 100:300], 
                                            u_factor, timer=timer_level)
            
            # Convert to energy
            if sx in ['DEAD', '1kHz_CLK', 'ABS_REF']:
                pulse_energy = 0
            else: 
                erg_s1 = 'C:/python/definitions/energy_calibration_10_30_S1.txt'
                erg_s2 = 'C:/python/definitions/energy_calibration_10_30_S2.txt'
                pulse_energy = dfs.get_energy_calibration(-pulse_area, 
                                                          detector_name=sx, 
                                                          file_path_S1=erg_s1,
                                                          file_path_S2=erg_s2,
                                                          timer = timer_level)
            
            '''
            Remove end of synch pulses to avoid getting a lower minimum than
            is true. We keep 0-22 ns.
            '''
            if sx in ['ABS_REF', '1kHz_CLK']:
                pulse_data_sinc = pulse_data_sinc[:, 0:220]
                       
            # Perform time pickoff method
            time_pickoff = dfs.time_pickoff_CFD(pulse_data_sinc, fraction=0.05,
                                                timer=timer_level) / u_factor
            
            # Adjust time stamps
            if boa in ['01', '02', '03', '04', '05']: 
                time_adjustment = time_pickoff - pre_trig_adjustment
            else: 
                time_offset = udfs.import_offset(board=boa, path=path)
                time_adjustment = time_pickoff-time_offset
            
            new_times = time_stamps+time_adjustment
                            
            # Store results and save
            time_dict[sx] = new_times
            erg_dict[sx] = pulse_energy
            
    info = (f'Contains times and energies for laser intensity {directory}' 
            'using standard time pickoff method and energy calibration.')
    
    to_pickle = {'time': time_dict,
                  'energy':erg_dict,
                  'info':info}
    udfs.pickler(f'data/decimated/10_30_ns/{directory}.pickle', to_pickle)










