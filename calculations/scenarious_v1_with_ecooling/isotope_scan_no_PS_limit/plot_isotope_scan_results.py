"""
Script to plot results from isotope scan in "scan_isotopes_for_lhc_intensities"
"""
import matplotlib.pyplot as plt
import pandas as pd 
import json
import numpy as np
from pathlib import Path
import matplotlib.ticker as mticker

# Which ion scenarios we consider - also whether to include electron cooling or not
account_for_LEIR_ecooling = True
ecool_str = 'with_ecooling_limits' if account_for_LEIR_ecooling else ''

# Load ion data and initialize for test for bunch intensities 
data_folder = Path(__file__).resolve().parent.joinpath('../../../data').absolute()
full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), header=0, index_col=0).T

# Stable isotope data
He_isotopes = np.array([3., 4.])
O_isotopes = np.array([16., 17., 18.])
Mg_isotopes = np.array([24., 25., 26.])
Ar_isotopes = np.array([36., 38., 40.])
Ca_isotopes = np.array([40., 42., 43., 44., 46., 48.])
Kr_isotopes = np.array([78., 80., 82., 83., 84., 86.])
In_isotopes = np.array([113., 115.])
Xe_isotopes = np.array([124., 126., 128., 129., 130., 131., 132, 134., 136])
Pb_isotopes = np.array([204., 206., 207., 208.])

all_isotopes = [He_isotopes, O_isotopes, Mg_isotopes, Ar_isotopes, Ca_isotopes, Kr_isotopes, In_isotopes, Xe_isotopes, Pb_isotopes]

def read_isotope_scan_results(ion_type, output_extra_str):
    
    # Define output strings to read correct file
    output_1 = '1_baseline_{}'.format(output_extra_str)
    output_2 ='2_no_PS_splitting_{}'.format(output_extra_str)
    output_3 = '3_LEIR_PS_stripping_{}'.format(output_extra_str)
    output_4 = '4_no_PS_splitting_and_LEIR_PS_stripping_{}'.format(output_extra_str)
    
    df1 = pd.read_csv("output/isotope_scan_results/{}_{}_{}.csv".format(ion_type, output_1, ecool_str), header=3, index_col=0)
    df2 = pd.read_csv("output/isotope_scan_results/{}_{}_{}.csv".format(ion_type, output_2, ecool_str), header=3, index_col=0)
    df3 = pd.read_csv("output/isotope_scan_results/{}_{}_{}.csv".format(ion_type, output_3, ecool_str), header=3, index_col=0)
    df4 = pd.read_csv("output/isotope_scan_results/{}_{}_{}.csv".format(ion_type, output_4, ecool_str), header=3, index_col=0)
    
    # Convert strings of charge states to numpy array
    A_states = np.array(list(map(int, df1.columns.values)))
    
    # Find default isotope and its performance in baseline scenario
    A_default = full_ion_data[ion_type]['A']
    ind_A_default = np.where(A_states == A_default)[0][0]
    Nb0 = float(df1.loc['LHC_ionsPerBunch'].iloc[ind_A_default])

    #### PLOTTING - performance of all charge states ####
    fig, ax = plt.subplots(1, 1, figsize = (6,5))
    ax.plot(A_default, Nb0, 'ro', markersize=13, alpha=0.8, label='1: Baseline with\ndefault charge state')
    ax.plot(A_states, np.array(list(map(float, df1.loc['LHC_ionsPerBunch'].values))), marker='o', color='blue', linewidth=4, linestyle='-', label='1: Baseline')
    ax.plot(A_states, np.array(list(map(float, df2.loc['LHC_ionsPerBunch'].values))), marker='o', linestyle='--', color='gold', linewidth=3, label='2: No PS splitting') #
    ax.plot(A_states, np.array(list(map(float, df3.loc['LHC_ionsPerBunch'].values))), marker='o', linestyle='-.', color='limegreen', linewidth=3.5, label='3: LEIR-PS stripping') #
    ax.plot(A_states, np.array(list(map(float, df4.loc['LHC_ionsPerBunch'].values))), marker='o', linestyle='--', color='gray', linewidth=3, label='4: LEIR-PS stripping, \nno PS splitting') #
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.set_ylabel('LHC bunch intensity')
    if ion_type == 'Xe':
        ax.tick_params(axis='x', which='major', labelsize=12)
    ax.set_xlabel('Mass number A')
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig('output/figures/{}{}_isotope_state_scan_{}.png'.format(ion_type, output_extra_str, ecool_str), dpi=250)
    plt.close()
    
    # Find best isotope, compare to baseline case
    print('\nIon type: {}'.format(ion_type))
    print('Default A: {}'.format(A_default))
    print('Scenario 2: no bunch splitting')
    Nb0_case2 = float(df2.loc['LHC_ionsPerBunch'].iloc[ind_A_default])
    Nb_case2_relative = np.array(list(map(float, df2.loc['LHC_ionsPerBunch'].values))) / Nb0_case2
    ind_A_max = np.argmax(Nb_case2_relative)
    print('Best A = {}, with {:.2f} percent of outgoing Nb\n'.format(A_states[ind_A_max], 100 * Nb_case2_relative[ind_A_max]))
    
    return A_states[ind_A_max], float(Nb_case2_relative[ind_A_max])
    

# Make empty dictionary 
isotope_dict = {'Ion' : [],
                'Best A': [],
                'Scenario2_Nb0_improvement_factor': []
                }

# Scan over ions and load results
for ion_type in full_ion_data.columns:

    # Plot results
    best_A, improvement_Nb_case2_rel = read_isotope_scan_results(ion_type, '_no_PS_SC_limit')
    
    isotope_dict['Ion'].append(ion_type)
    isotope_dict['Best A'].append(best_A)
    isotope_dict['Scenario2_Nb0_improvement_factor'].append(improvement_Nb_case2_rel)
        

# Flatten array
isotope_dict['Scenario2_Nb0_improvement_factor'] = np.array(isotope_dict['Scenario2_Nb0_improvement_factor']).tolist()

# save dictionary
with open("output/isotope_scan.json", "w") as fp:
    json.dump(isotope_dict , fp, default=str)     
print('Created dictionary:\n{}'.format(isotope_dict))



