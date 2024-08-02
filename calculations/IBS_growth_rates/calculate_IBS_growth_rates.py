"""
Calculating growth rates with Nagaitsev formalism
"""
import injector_model

inj = injector_model.InjectorChain_v2(nPulsesLEIR=7)
inj.calculate_IBS_growth_rates_all_ion_species()