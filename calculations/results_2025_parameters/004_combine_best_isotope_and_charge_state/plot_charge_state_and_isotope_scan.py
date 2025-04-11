"""
Load charge state and isotope scan data, and plot
"""
import json
import numpy as np
import injector_model
import matplotlib.pyplot as plt
import pandas as pd

# Load reference ion data
ion_data = pd.read_csv("../../../data/Ion_species.csv", header=0, index_col=0).T
mass_number = ion_data.loc['A']
ref_Table_SPS = pd.read_csv('../../../data/test_and_benchmark_data/SPS_final_intensities_WG5_and_Hannes.csv', index_col=0)

# Load charge state scan - no PS space charge limit
with open("../002_charge_state_scan/output/charge_state_scan.json", "r") as fp:
   charge_dict = json.load(fp)

# Load isotope scan
with open("../003_isotope_scan/output/isotope_scan.json", "r") as fp:
   isotope_dict = json.load(fp)

# Convert strings to numpy arrays
charge_dict['Scenario2_Nb0_improvement_factor'] = np.array(list(map(float, charge_dict['Scenario2_Nb0_improvement_factor'])))
charge_dict['Best Q'] = np.array(list(map(float, charge_dict['Best Q'])))
isotope_dict['Scenario2_Nb0_improvement_factor'] = np.array(list(map(float, isotope_dict['Scenario2_Nb0_improvement_factor'])))
isotope_dict['Best A'] = np.array(list(map(float, isotope_dict['Best A'])))


# Load data from scenarios
df1 = pd.read_csv('../001_baseline_scenarios/output_csv/1_baseline.csv', index_col=0).T
df2 = pd.read_csv('../001_baseline_scenarios/output_csv/2_no_PS_splitting.csv', index_col=0).T
df3 = pd.read_csv('../001_baseline_scenarios/output_csv/3_LEIR_PS_stripping.csv', index_col=0).T
df4 = pd.read_csv('../001_baseline_scenarios/output_csv/4_no_PS_splitting_and_LEIR_PS_stripping.csv', index_col=0).T

# Define bar width for bar plot
bar_width5 = 0.09
x = np.arange(len(df1.index))

# Define B-field mask for scenario 3 and 4, where some PS magnetic fields are too low
mask = df3['PS_B_field_is_too_low'].map({'True': False, 'False': True})

# Isotope and charge state check check - NO PS splitting - also include charge state and isotope scan 
fig7, ax7 = plt.subplots(1, 1, figsize = (6,5))
ax7.bar(x - bar_width5, ref_Table_SPS['WG5 Intensity'].astype(float)*mass_number, bar_width5, color='red', label='WG5') #
ax7.bar(x , df1['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='blue', label='1: Baseline scenario') #
ax7.bar(x + bar_width5, df2['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='gold', label='2: No PS splitting') #
ax7.bar( (x + 2*bar_width5) * mask, df3['LHC_ionsPerBunch'].astype(float)*mass_number * mask, bar_width5, color='limegreen', label='3: LEIR-PS stripping') #
ax7.bar( (x + 3*bar_width5) * mask, df4['LHC_ionsPerBunch'].astype(float)*mass_number * mask, bar_width5, color='gray', label='4: LEIR-PS stripping, \nno PS splitting') #

# Also plot improvement factors from charge state and isotope scan
ax7.bar(x + 4*bar_width5, df2['LHC_ionsPerBunch'].astype(float)*mass_number * charge_dict['Scenario2_Nb0_improvement_factor'], 
        bar_width5, color='green', label='Best LEIR charge state\nwith 2: no PS splitting') #
ax7.bar(x + 5*bar_width5, df2['LHC_ionsPerBunch'].astype(float)*mass_number * isotope_dict['Scenario2_Nb0_improvement_factor'], 
        bar_width5, color='cyan', label='Best isotope with \n2: PS splitting') 

ax7.set_xticks(x + 2*bar_width5)
ax7.set_xticklabels(df1.index)
ax7.set_ylabel("Nucleons per bunch")
ax7.legend(fontsize=10, loc='upper right')
fig7.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig7.savefig('5_scenarios_with_best_charge_states_and_isotopes.png', dpi=250)
plt.close()