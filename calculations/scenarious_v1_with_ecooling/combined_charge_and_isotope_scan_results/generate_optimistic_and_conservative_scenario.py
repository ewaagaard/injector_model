"""
Load charge state and isotope scan data - generate conservative and optimistic cases

Account for LEIR e-cooling
Check PS space charge tune shift, assume that it will not be a limit for now
"""
import json
import numpy as np
from injector_model import InjectorChain
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os

os.makedirs('output', exist_ok=True)

# Consider PS space charge limit to be at present Pb tune shifts, or not
consider_PS_SC_limit = False

if consider_PS_SC_limit:
    load_string = 'scenarios_with_ecooling'
    PS_string = 'with_PS_limit'
else:
    load_string = 'test_no_PS_space_charge_limit'
    PS_string = 'no_PS_limit'

# Define output string for optimistic value
ions_not_stripped = ['He', 'O', 'Mg', 'Ar', 'Kr']

# Load reference ion data
ion_data = pd.read_csv("../../../data/Ion_species.csv", header=0, index_col=0).T
mass_number = ion_data.loc['A']
ref_Table_SPS = pd.read_csv('../../../data/test_and_benchmark_data/SPS_final_intensities_WG5_and_Hannes.csv', index_col=0)
Roderik_values = pd.read_csv('../../../data/test_and_benchmark_data/Roderik_2021_LHC_charges_per_bunch_output.csv', index_col=0)

# Load charge state scan - no PS space charge limit
with open("../charge_state_scan_{}/output/charge_state_scan.json".format(PS_string), "r") as fp:
   charge_dict = json.load(fp)

# Load isotope scan
with open("../isotope_scan_{}/output/isotope_scan.json".format(PS_string), "r") as fp:
   isotope_dict = json.load(fp)

# Convert strings to numpy arrays
charge_dict['Scenario2_Nb0_improvement_factor'] = np.array(list(map(float, charge_dict['Scenario2_Nb0_improvement_factor'])))
charge_dict['Best Q'] = np.array(list(map(float, charge_dict['Best Q'])))
isotope_dict['Scenario2_Nb0_improvement_factor'] = np.array(list(map(float, isotope_dict['Scenario2_Nb0_improvement_factor'])))
isotope_dict['Best A'] = np.array(list(map(float, isotope_dict['Best A'])))

#### CONSERVATIVE ####
# Load conservative result - from baseline scenario
full_result_conservative = pd.read_csv('../scenarios_with_ecooling/output_csv/1_baseline.csv', index_col=0)

#### OPTIMISTIC ####
# --> find currents, charge states (isotopes not for now)
# Use best charge state, its current and generously round up to nearest number of LEIR injections

# Initialize full dicionary
full_result_optimistic = defaultdict(list)

# Run cases with best Q, best A and updated LINAC current
for i, ion_type in enumerate(ion_data.columns):

        # Load current from charge state scan
        PS_case  = 'no_PS_SC_limit' if not consider_PS_SC_limit else 'with_PS_SC_limit'
        output_extra_str='UNSTRIPPED_{}'.format(PS_case) if ion_type in ions_not_stripped else 'STRIPPED_{}'.format(PS_case)
        read_str = '../charge_state_scan_{}/output/charge_scan_results/{}_2_no_PS_splitting_{}_with_ecooling_limits.csv'.format(PS_string, ion_type, output_extra_str)       
        df_charge_state_scan = pd.read_csv(read_str, index_col=0)

        # Build custom input data
        custom_ion_data = ion_data[ion_type].copy() # copy entry
        Q = charge_dict['Best Q'][i]
        current = df_charge_state_scan['{}'.format(int(Q))].loc['Linac3_current [A]']
        custom_ion_data['Q before stripping'] = Q   
        custom_ion_data['Linac3 current [uA]'] = float(current) * 1e6 # convert from A to uA
        print('Q strip = {} has Linac3 current: {} uA'.format(Q, current))

        # Optimistic case - recalculate
        inj_optimistic = InjectorChain(PS_splitting = 1,
                                       account_for_LEIR_ecooling=True,
                                       PS_factor_SC = np.inf,
                                       round_number_of_LEIR_inj_up=True
                                       )
        inj_optimistic.init_ion(ion_type=ion_type, ion_data_custom=custom_ion_data)        
        result_optimistic = inj_optimistic.calculate_LHC_bunch_intensity()

        # Append the values to the corresponding key 
        for key, value in result_optimistic.items():
            full_result_optimistic[key].append(value)

# Make dataframe of optimistic scenario - save as csv and excel
df_optimistic = pd.DataFrame(full_result_optimistic)
df_optimistic = df_optimistic.set_index('Ion')
df_optimistic.T.to_csv('output/2_no_PS_bunch_splitting_optimistic.csv')
full_result_conservative.to_csv('output/1_baseline_conservative.csv')

ratio = df_optimistic['LHC_ionsPerBunch'] / full_result_conservative.T['LHC_ionsPerBunch'].astype(float)
print('Final ratio optimistic vs conservative:\n{}'.format(ratio))

# Make new dataframe with optimistic and conservative scenario
df_both = pd.DataFrame()
df_both['Conservative LHC_ionsPerBunch'] = full_result_conservative.T['LHC_ionsPerBunch'].astype(float)
df_both['Optimistic LHC_ionsPerBunch'] = df_optimistic['LHC_ionsPerBunch']
df_both['Ratio opt over conserv'] = ratio
df_both.to_csv('output/Conservative_vs_optimistic_scenario.csv')

# Print baseline ratio vs Roderik's baseline values from 2021 - in charges per bunch
print('Ratio number of charges into LHC: new baseline with ecooling / Roderik 2021 values:')
print(df_both['Conservative LHC_ionsPerBunch'] * ion_data.T['Z'] / Roderik_values['Baseline'])

# Define bar width for bar plot
bar_width2 = 0.25
x = np.arange(len(df_both.index))

fig3, ax3 = plt.subplots(1, 1, figsize = (6,5))
bar31 = ax3.bar(x - bar_width2, ref_Table_SPS['WG5 Intensity'] * ion_data.T['A'], bar_width2, color='red', label='WG5') #
bar32 = ax3.bar(x, df_both['Conservative LHC_ionsPerBunch'] * ion_data.T['A'], bar_width2, color='blue', label='Conservative scenario') #
bar33 = ax3.bar(x + bar_width2, df_both['Optimistic LHC_ionsPerBunch'] * ion_data.T['A'], bar_width2, color='lime', label='Optimistic scenario') #
ax3.set_xticks(x)
ax3.set_xticklabels(df_both.index)
ax3.set_ylabel("Nucleons per bunch")
ax3.legend()
fig3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig3.savefig('output/Conservative_Optimistic_vs_WG5_scecnario.png', dpi=250)
plt.close()