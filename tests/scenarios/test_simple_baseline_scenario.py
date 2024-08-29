"""
Test baseline scenario of new model with full SC lattice integral 
"""
from injector_model import InjectorChain

# Instantiate injector chain version 2 and calculate LHC bunch intensity for all ions
injector_chain = InjectorChain() 
df = injector_chain.calculate_LHC_bunch_intensity_all_ion_species()

# Then try assuming 7 injections in LEIR
injector_chain.nPulsesLEIR = 7 
df2 = injector_chain.calculate_LHC_bunch_intensity_all_ion_species(output_name='output_seven_injections')

# Test electron cooling rates
injector_chain.account_for_LEIR_ecooling = True 
df3 = injector_chain.calculate_LHC_bunch_intensity_all_ion_species(output_name='output_with_LEIR_ecooling')

print('Relative difference #injected charges in LHC due to e-cooling:')
print(df3['LHC_chargesPerBunch'] / df['LHC_chargesPerBunch'])
