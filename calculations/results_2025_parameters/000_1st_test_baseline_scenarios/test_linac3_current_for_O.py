"""
Small test script for varying LINAC3 current for oxygen
"""
import matplotlib.pyplot as plt
import pandas as pd 
from injector_model import InjectorChain

inj1 = InjectorChain(ion_type='O')
result_70A = inj1.calculate_LHC_bunch_intensity()

inj2 = InjectorChain(ion_type='O')
inj2.linac3_current = 90e-6
result_90A = inj2.calculate_LHC_bunch_intensity()

print('70 A:')
result_70A

print('90 A:')
result_90A