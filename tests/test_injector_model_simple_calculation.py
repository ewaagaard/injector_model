"""
Test calculating propagated beam intensity with the full space charge integral
"""
from injector_model import InjectorChain_v2

# Instantiate injector chain version 2 and calculate LHC bunch intensity for all ions
injector_chain = InjectorChain_v2(nPulsesLEIR=7)  # 7 OR DEFAULT NUMBER?
df = injector_chain.calculate_LHC_bunch_intensity_all_ion_species()