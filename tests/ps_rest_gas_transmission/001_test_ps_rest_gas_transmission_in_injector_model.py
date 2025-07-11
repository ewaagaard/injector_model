"""
Test script to check impact on PS transmission if accounting for beam-gas lifetimes
"""
import injector_model

#### Baseline scenario ####

# V1: 2025 parameters, assume flat PS transmission of 92%
inj = injector_model.InjectorChain(account_for_PS_rest_gas=False) 
df = inj.calculate_LHC_bunch_intensity_all_ion_species()

# V2: 2025 parameters, assume rest-gas dependent PS transmission
inj2 = injector_model.InjectorChain(account_for_PS_rest_gas=True) 
df2 = inj2.calculate_LHC_bunch_intensity_all_ion_species()

print('\n1: Baseline\n')
print('Ratio injected LHC charges: rest-gas over flat transmission:')
print(df2['LHC_chargesPerBunch'] / df['LHC_chargesPerBunch'])

df2.T.to_csv("output_Scenario_1_V2_2025_PS_rest_gas_transmission.csv")
df.T.to_csv("output_Scenario_1_V1_2025_flat_PS_transmission.csv")

#### No splitting scenario ####

# V1: 2025 parameters, assume flat PS transmission of 92%
inj3 = injector_model.InjectorChain(account_for_PS_rest_gas=False, PS_splitting=1) 
df3 = inj3.calculate_LHC_bunch_intensity_all_ion_species()

# V2: 2025 parameters, assume rest-gas dependent PS transmission
inj4 = injector_model.InjectorChain(account_for_PS_rest_gas=True, PS_splitting=1) 
df4 = inj4.calculate_LHC_bunch_intensity_all_ion_species()

print('\n2: No PS bunch splitting\n')
print('Ratio injected LHC charges: rest-gas over flat transmission:')
print(df4['LHC_chargesPerBunch'] / df3['LHC_chargesPerBunch'])

df4.T.to_csv("output_Scenario_2_V2_2025_PS_rest_gas_transmission.csv")
df3.T.to_csv("output_Scenario_2_V1_2025_flat_PS_transmission.csv")