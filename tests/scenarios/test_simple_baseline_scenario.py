"""
Test baseline scenario of new model with full SC lattice integral 
"""
from injector_model import InjectorChain

# Instantiate injector chain version 2 and calculate LHC bunch intensity for all ions
injector_chain = InjectorChain() 
df = injector_chain.calculate_LHC_bunch_intensity_all_ion_species()

# Test electron cooling rates
injector_chain.account_for_LEIR_ecooling = True 
df2 = injector_chain.calculate_LHC_bunch_intensity_all_ion_species(output_name='output_with_LEIR_ecooling')

print('Relative difference #injected charges in LHC due to e-cooling:')
print(df2['LHC_chargesPerBunch'] / df['LHC_chargesPerBunch'])
