"""
Script to plot results from charge state scan in "scan_charge_states_for_lhc_intensities"
"""
import matplotlib.pyplot as plt
import pandas as pd 
import json
import numpy as np
import injector_model
from pathlib import Path

# Which ion scenarios we consider - also whether to include electron cooling or not
ions_not_stripped = ['He', 'O', 'Mg', 'Ar', 'Kr']
account_for_LEIR_ecooling = True
ecool_str = 'with_ecooling_limits' if account_for_LEIR_ecooling else ''

# Charge states with 3d10 shell structure - to be avoided for recombination
recombination_dict = {'He': [], 'O': [], 'Mg': [], 'Ar':[], 'Ca': [], 'Kr': [7], 
                      'In': [20], 'Xe' : [25], 'Pb': [53]}

# Load ion data and initialize for test for bunch intensities 
data_folder = Path(__file__).resolve().parent.joinpath('../../../data').absolute()
full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), header=0, index_col=0).T


def read_charge_scan_results(ion_type, output_extra_str):
    
    # Define output strings to read correct file
    output_1 = '1_baseline_{}'.format(output_extra_str)
    output_2 ='2_no_PS_splitting_{}'.format(output_extra_str)
    output_3 = '3_LEIR_PS_stripping_{}'.format(output_extra_str)
    output_4 = '4_no_PS_splitting_and_LEIR_PS_stripping_{}'.format(output_extra_str)
    
    df1 = pd.read_csv("output/charge_scan_results/{}_{}_{}.csv".format(ion_type, output_1, ecool_str), index_col=0)
    df2 = pd.read_csv("output/charge_scan_results/{}_{}_{}.csv".format(ion_type, output_2, ecool_str), header=None, index_col=0)
    df3 = pd.read_csv("output/charge_scan_results/{}_{}_{}.csv".format(ion_type, output_3, ecool_str), header=None, index_col=0)
    df4 = pd.read_csv("output/charge_scan_results/{}_{}_{}.csv".format(ion_type, output_4, ecool_str), header=None, index_col=0)
    
    # Convert strings of charge states to numpy array
    Q_states = np.array(list(map(int, df1.columns.values)))
    
    # Find default charge state and its performance in baseline scenario
    if ion_type == 'Kr' and output_extra_str == 'STRIPPED_no_PS_SC_limit':
        Q_default = 29
    else:
        Q_default = full_ion_data[ion_type]['Q before stripping']
    ind_Q_default = np.where(Q_states == Q_default)[0][0]
    Nb0 = float(df1.loc['LHC_ionsPerBunch'].iloc[ind_Q_default])

    #### PLOTTING - performance of all charge states ####
    fig, ax = plt.subplots(1, 1, figsize = (6,5))
    #fig.suptitle(ion, fontsize=20)
    #if row['Z'] > 2.0:
    #    ax.axvspan(0.0, np.max(Q_states[Q_states / row['A'] < LinacLEIRlim]), alpha=0.25, color='coral', label='Not accessible LINAC3-LEIR')
    ax.plot(Q_states, np.array(list(map(float, df1.loc['LHC_ionsPerBunch'].values))), color='blue', linewidth=4.2, linestyle='-', label='1: Baseline')
    ax.plot(Q_states, np.array(list(map(float, df2.loc['LHC_ionsPerBunch'].values))), linestyle='-.', color='gold', linewidth=3.8, label='2: No PS splitting') #
    ax.plot(Q_states, np.array(list(map(float, df3.loc['LHC_ionsPerBunch'].values))), linestyle='-.', color='limegreen', linewidth=3.5, label='3: LEIR-PS stripping') #
    ax.plot(Q_states, np.array(list(map(float, df4.loc['LHC_ionsPerBunch'].values))), linestyle='--', color='gray', linewidth=3, label='4: LEIR-PS stripping, \nno PS splitting') #
    ax.plot(Q_default, Nb0, 'ro', markersize=13, alpha=0.8, label='1: Baseline with\ndefault charge state')
    ax.set_ylabel('LHC bunch intensity')
    ax.set_xlabel('LEIR charge state')
    ax.legend(fontsize=12)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig('output/figures/{}_{}_LEIR_charge_state_scan_{}.png'.format(ion_type, output_extra_str, ecool_str), dpi=250)
    plt.close()
    
    
    # Find best charge state, compare to baseline case
    print('\nIon type: {}'.format(ion_type))
    print('Default Q: {}'.format(Q_default))
    print('Scenario 2: no bunch splitting')
    Nb0_case2 = float(df2.loc['LHC_ionsPerBunch'].iloc[ind_Q_default])
    Nb_case2_relative = np.array(list(map(float, df2.loc['LHC_ionsPerBunch'].values))) / Nb0_case2
    ind_Q_max = np.argmax(Nb_case2_relative)
    print('Best Q = {}, with {:.2f} percent of outgoing Nb\n'.format(Q_states[ind_Q_max], 100 * Nb_case2_relative[ind_Q_max]))
    
    return Q_states[ind_Q_max], float(Nb_case2_relative[ind_Q_max])
    

# Make empty dictionary 
isotope_dict = {'Ion' : [],
                'Best Q': [],
                'Scenario2_Nb0_improvement_factor': []
                }

# Scan over ions and load results
for ion_type in full_ion_data.columns:

    if ion_type in ions_not_stripped:

        # Untripped ions after LINAC3 - define path and load LINAC3 current data
        print('\nIon type: {}, UNSTRIPPED'.format(ion_type))
        best_Q, improvement_Nb_case2_rel = read_charge_scan_results(ion_type, 'UNSTRIPPED_no_PS_SC_limit')
        
        isotope_dict['Ion'].append(ion_type)
        isotope_dict['Best Q'].append(best_Q)
        isotope_dict['Scenario2_Nb0_improvement_factor'].append(improvement_Nb_case2_rel)
        
    if ion_type not in ions_not_stripped or ion_type == 'Kr':
    
        # Same for unstripped ions after LINAC3
        print('\nIon type: {}, STRIPPED'.format(ion_type))
        best_Q, improvement_Nb_case2_rel = read_charge_scan_results(ion_type, 'STRIPPED_no_PS_SC_limit')

        # Append only krypton for its best case, not here when stripped        
        if ion_type not in ions_not_stripped:
            isotope_dict['Ion'].append(ion_type)
            isotope_dict['Best Q'].append(best_Q)
            isotope_dict['Scenario2_Nb0_improvement_factor'].append(improvement_Nb_case2_rel)
        

# Flatten array
isotope_dict['Scenario2_Nb0_improvement_factor'] = np.array(isotope_dict['Scenario2_Nb0_improvement_factor']).tolist()

# save dictionary
with open("output/charge_state_scan.json", "w") as fp:
    json.dump(isotope_dict , fp, default=str)     
print('Created dictionary:\n{}'.format(isotope_dict))
        
        
        