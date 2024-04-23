"""
Test calculating propagated beam intensity with the full space charge integral
"""
from injector_model import InjectorChain, InjectorChain_v2

# Instantiate injector chain version 2 and calculate LHC bunch intensity for all ions
injector_chain = InjectorChain_v2(nPulsesLEIR=7)  # 7 OR DEFAULT NUMBER?
df = injector_chain.calculate_LHC_bunch_intensity_all_ion_species()

# Run same calculation for old injector chain, and observe difference
injector_chain_old = InjectorChain(nPulsesLEIR=7, use_gammas_ref=True) 
df2 = injector_chain_old.calculate_LHC_bunch_intensity_all_ion_species()
df2 = df2.T
df2.to_csv("output_csv/output_old_model.csv")