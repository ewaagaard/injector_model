"""
Script to plot results from charge state scan in "scan_charge_states_for_lhc_intensities"
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import pandas as pd 
import json
import numpy as np
import injector_model
from pathlib import Path

# Which ion scenarios we consider - also whether to include electron cooling or not
ions_not_stripped = ['He', 'O', 'Mg', 'Ar', 'Kr']
account_for_LEIR_ecooling = True
ecool_str = '' if account_for_LEIR_ecooling else '_without_ecooling_limits'

# Charge states with 3d10 shell structure - to be avoided for recombination
recombination_dict = {'He': [], 'O': [], 'Mg': [], 'Ar':[], 'Ca': [], 'Kr': [7], 
                      'In': [20], 'Xe' : [25], 'Pb': [53]}

# Load ion data and initialize for test for bunch intensities 
data_folder = Path(__file__).resolve().parent.joinpath('../../../data').absolute()
full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), header=0, index_col=0).T

# Create the combined figure with subplots
num_cols = 2  # Two columns
num_rows = (len(full_ion_data.T.index) + 1) // num_cols  # Integer division to determine the number of rows, Kr appears twice
fig0, axs = plt.subplots(num_rows, num_cols, figsize=(8.27, 10.2), constrained_layout=True)

def read_charge_scan_results(ion_type, output_extra_str, count, stripped):
    
    # Define output strings to read correct file
    output_1 = '1_baseline_{}'.format(output_extra_str)
    output_2 ='2_no_PS_splitting_{}'.format(output_extra_str)
    output_3 = '3_LEIR_PS_stripping_{}'.format(output_extra_str)
    output_4 = '4_no_PS_splitting_and_LEIR_PS_stripping_{}'.format(output_extra_str)
    
    df1 = pd.read_csv("output/charge_scan_results/{}_{}{}_ps_rest_gas.csv".format(ion_type, output_1, ecool_str), index_col=0)
    df2 = pd.read_csv("output/charge_scan_results/{}_{}{}_ps_rest_gas.csv".format(ion_type, output_2, ecool_str), header=None, index_col=0)
    df3 = pd.read_csv("output/charge_scan_results/{}_{}{}_ps_rest_gas.csv".format(ion_type, output_3, ecool_str), header=None, index_col=0)
    df4 = pd.read_csv("output/charge_scan_results/{}_{}{}_ps_rest_gas.csv".format(ion_type, output_4, ecool_str), header=None, index_col=0)
    
    # Convert strings of charge states to numpy array
    Q_states = np.array(list(map(int, df1.columns.values)))
    
    # Find default charge state and its performance in baseline scenario
    if ion_type == 'Kr' and output_extra_str == 'STRIPPED':
        Q_default = 29
    else:
        Q_default = full_ion_data[ion_type]['Q before stripping']
    ind_Q_default = np.where(Q_states == Q_default)[0][0]
    Nb0 = float(df1.loc['LHC_ionsPerBunch'].iloc[ind_Q_default])

    #### PLOTTING - performance of all charge states ####
    fig, ax = plt.subplots(1, 1, figsize = (6,5), constrained_layout=True)
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
    #fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig('output/figures/{}_{}_LEIR_charge_state_scan_ps_rest_gas_{}.png'.format(ion_type, output_extra_str, ecool_str), dpi=250)
    plt.close()
    
    # Also fill in the combined superplot
    row3 = (count-1) // num_cols  # Row index
    col3 = (count-1) % num_cols  # Column index
    ax3 = axs[row3, col3]  # Select the current subplot
    ax3.plot(Q_states, np.array(list(map(float, df1.loc['LHC_ionsPerBunch'].values))), color='blue', linewidth=4.2, linestyle='-', label='1: Baseline')
    ax3.plot(Q_states, np.array(list(map(float, df2.loc['LHC_ionsPerBunch'].values))), linestyle='-.', color='gold', linewidth=3.8, label='2: No PS splitting') #
    ax3.plot(Q_states, np.array(list(map(float, df3.loc['LHC_ionsPerBunch'].values))), linestyle='-.', color='limegreen', linewidth=3.5, label='3: LEIR-PS stripping') #
    ax3.plot(Q_states, np.array(list(map(float, df4.loc['LHC_ionsPerBunch'].values))), linestyle='--', color='gray', linewidth=3, label='4: LEIR-PS stripping, \nno PS splitting') #
    ax3.plot(Q_default, Nb0, 'ro', markersize=13, alpha=0.8, label='1: Baseline with\ndefault charge state')
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.text(0.023, 0.71, '{}'.format(ion_type), fontsize=18, weight='bold', transform=ax3.transAxes, color='teal' if stripped else 'purple')
    ax3.grid(alpha=0.45)
    #ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    
    # Add legend in oxygen plot
    if count == 3:
        ax3.legend(fontsize=8, loc='upper right')
    
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
charge_dict = {'Ion' : [],
                'Best Q': [],
                'Scenario2_Nb0_improvement_factor': []
                }

# Keep track of subplot position
count = 1  

# Scan over ions and load results
for ion_type in full_ion_data.columns:

  
    if ion_type in ions_not_stripped:

        # Untripped ions after LINAC3 - define path and load LINAC3 current data
        print('\nIon type: {}, UNSTRIPPED'.format(ion_type))
        best_Q, improvement_Nb_case2_rel = read_charge_scan_results(ion_type, 'UNSTRIPPED', count, stripped=False)

        # Append only Kr when stripped
        if ion_type != 'Kr':        
            charge_dict['Ion'].append(ion_type)
            charge_dict['Best Q'].append(best_Q)
            charge_dict['Scenario2_Nb0_improvement_factor'].append(improvement_Nb_case2_rel)
        count += 1
        
    if ion_type not in ions_not_stripped or ion_type == 'Kr':
    
        # Same for unstripped ions after LINAC3
        print('\nIon type: {}, STRIPPED'.format(ion_type))
        best_Q, improvement_Nb_case2_rel = read_charge_scan_results(ion_type, 'STRIPPED', count, stripped=True)

        charge_dict['Ion'].append(ion_type)
        charge_dict['Best Q'].append(best_Q)
        charge_dict['Scenario2_Nb0_improvement_factor'].append(improvement_Nb_case2_rel)
    
        count += 1
        
# Fix superplot
# Share y-axes for the same row and share x-label for the same column
for col in range(num_cols):
    axs[-1, col].set_xlabel('LEIR charge state', fontsize=13)

# Share y-label for the same row
for row in axs:
    row[0].set_ylabel('$N_{b}$ into LHC', fontsize=13)

#### PLOT SETTINGS #######
fig0.savefig('output/figures/Combined_leir_charge_state_scan_ps_rest_gas.png', dpi=250)
plt.show()


# Flatten array
charge_dict['Scenario2_Nb0_improvement_factor'] = np.array(charge_dict['Scenario2_Nb0_improvement_factor']).tolist()

# save dictionary
with open("output/charge_state_scan.json", "w") as fp:
    json.dump(charge_dict , fp, default=str)     
print('Created dictionary:\n{}'.format(charge_dict))
