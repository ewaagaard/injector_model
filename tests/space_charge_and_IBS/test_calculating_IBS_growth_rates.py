"""
Test calculating growth rates from Injector Model
"""
import injector_model

inj = injector_model.InjectorChain_v2(nPulsesLEIR=7)
growth_rates = inj.calculate_IBS_growth_rates()
