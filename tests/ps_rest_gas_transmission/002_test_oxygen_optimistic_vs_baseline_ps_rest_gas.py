"""
Test script to check impact on PS transmission if accounting for beam-gas lifetimes
in particular in Scenario 2 (no PS bunch splitting) for O4+ vs O5+ in the PS
Assume flat LINAC3 current of 90 uA
"""
import injector_model


#### No splitting scenario, with V2: 2025 parameters, assume rest-gas dependent PS transmission ####

# Default charge state: O4+
inj0 = injector_model.InjectorChain(ion_type='O', account_for_PS_rest_gas=True, PS_splitting=1) 
df0 = inj0.calculate_LHC_bunch_intensity()

# Previously optimized charge state: O5+
O5_data = inj0.full_ion_data['O'].copy()
O5_data['Q before stripping'] = 5

inj1 = injector_model.InjectorChain(ion_type='O', account_for_PS_rest_gas=True, PS_splitting=1) 
inj1.init_ion(ion_type='O', ion_data_custom=O5_data)
df1 = inj1.calculate_LHC_bunch_intensity()

#### No splitting scenario, with V2: 2025 parameters, assume flat PS transmission ####

inj2 = injector_model.InjectorChain(ion_type='O', account_for_PS_rest_gas=False, PS_splitting=1) 
df2 = inj2.calculate_LHC_bunch_intensity()
inj3 = injector_model.InjectorChain(ion_type='O', account_for_PS_rest_gas=False, PS_splitting=1) 
inj3.init_ion(ion_type='O', ion_data_custom=O5_data)
df3 = inj3.calculate_LHC_bunch_intensity()

print('\n2a: No PS bunch splitting: with rest gas transmission\n')
print('Ratio injected LHC charges: O5+ over O4+')
print(df1['LHC_chargesPerBunch'] / df0['LHC_chargesPerBunch'])

print('\n2b: No PS bunch splitting: with flat transmission\n')
print('Ratio injected LHC charges: O5+ over O4+')
print(df3['LHC_chargesPerBunch'] / df2['LHC_chargesPerBunch'])