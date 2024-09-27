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

# Consider PS space charge limit to be at present Pb tune shifts, or not
consider_PS_SC_limit = False

if consider_PS_SC_limit:
    load_string = 'scenarios_with_ecooling'
    output_extra_str = '_with_PS_SC_limit'
else:
    load_string = 'test_no_PS_space_charge_limit'
    output_extra_str = '_no_PS_SC_limit'
    

# Load reference ion data
ion_data = pd.read_csv("../../../data/Ion_species.csv", header=0, index_col=0).T
mass_number = ion_data.loc['A']
ref_Table_SPS = pd.read_csv('../../../data/test_and_benchmark_data/SPS_final_intensities_WG5_and_Hannes.csv', index_col=0)

# Load charge state scan - no PS space charge limit
with open("../charge_state_scan_no_PS_limit/output/charge_state_scan.json", "r") as fp:
   charge_dict = json.load(fp)

# Load isotope scan
with open("../isotope_scan_no_PS_limit/output/isotope_scan.json", "r") as fp:
   isotope_dict = json.load(fp)

# Convert strings to numpy arrays
charge_dict['Scenario2_Nb0_improvement_factor'] = np.array(list(map(float, charge_dict['Scenario2_Nb0_improvement_factor'])))
charge_dict['Best Q'] = np.array(list(map(float, charge_dict['Best Q'])))
isotope_dict['Scenario2_Nb0_improvement_factor'] = np.array(list(map(float, isotope_dict['Scenario2_Nb0_improvement_factor'])))
isotope_dict['Best A'] = np.array(list(map(float, isotope_dict['Best A'])))

#### CONSERVATIVE ####
# Load conservative result - from baseline scenario
full_result_conservative = pd.read_csv(...)

#### OPTIMISTIC ####
# --> find currents, charge states (isotopes not for )

# Get current for best Q and A from Json
# Update number of allowed LEIR injections?


# Initialize full dicionary
full_result_optimistic = defaultdict(list)

# Run cases with best Q, best A and updated LINAC current
for ion_type in ion_data.columns:

        custom_ion_data = injector_chain2.full_ion_data[ion_type].copy() # copy entry
        custom_ion_data['Q before stripping'] = Q
        custom_ion_data['Linac3 current [uA]'] = current
        print('Q strip = {} has Linac3 current: {} uA'.format(Q, current))

        # Optimistic case - recalculate
        injector_chain2 = InjectorChain(PS_splitting = 1,
                                        account_for_LEIR_ecooling=True,
                                        PS_factor_SC = np.inf
                                    )
        
        result_optimistic = injector_chain2.calculate_LHC_bunch_intensity()

        # Append the values to the corresponding key 
        for key, value in result_optimistic.items():
            full_result_optimistic[key].append(value)

        injector_chain2.init_ion(ion_type=ion_type, ion_data_custom=custom_ion_data)
        df2 = injector_chain2.calculate_LHC_bunch_intensity()

# Make dataframe of optimistic scenario
df_optimistic = pd.DataFrame(full_result_optimistic)