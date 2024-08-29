"""
Test calculating propagated beam intensity with the full space charge integral
"""
from injector_model import InjectorChain
from injector_model import InjectorChain_v0

# Instantiate injector chain version 2 and calculate LHC bunch intensity for all ions
injector_chain = InjectorChain(nPulsesLEIR=7)  # 7 OR DEFAULT NUMBER?
df = injector_chain.calculate_LHC_bunch_intensity_all_ion_species()

# Run same calculation for old injector chain, and observe difference
injector_chain_old = InjectorChain_v0(nPulsesLEIR=7, use_gammas_ref=True) 
df2 = injector_chain_old.calculate_LHC_bunch_intensity_all_ion_species()
df2['LEIR SC limit ratio new vs old'] = df['LEIR_space_charge_limit'] / df2['LEIR_space_charge_limit']
df2['PS SC limit ratio new vs old'] = df['PS_space_charge_limit'] / df2['PS_space_charge_limit']
df2['SPS SC limit ratio new vs old'] = df['SPS_space_charge_limit'] / df2['SPS_space_charge_limit']
df2['Ratio LHC_chargesPerBunch new vs old'] = df['LHC_chargesPerBunch'] / df2['LHC_chargesPerBunch']
df2 = df2.T
df2.to_csv("output_csv/output_old_model.csv")


# Print difference in space charge limits
print('\nLEIR SC limit: old vs new') 
print(df2.loc['LEIR SC limit ratio new vs old'])

print('\nPS SC limit: old vs new') 
print(df2.loc['PS SC limit ratio new vs old'])

print('\nSPS SC limit: old vs new') 
print(df2.loc['SPS SC limit ratio new vs old'])

print('\nCharges into LHC ratio: old vs new')
print(df2.loc['Ratio LHC_chargesPerBunch new vs old'])