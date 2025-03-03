"""
Script to calculate kinetic energy per nucleon at injection
"""
from injector_model import InjectionEnergies

# Data for Mg7 and O4 - ['A', 'Q_low', 'm_ion_in_u', 'Z']
Mg7_data = [24,7,23.985041697,12]
O4_data = [16,4,15.99491461957,8]

inj_Ek_Mg7 = InjectionEnergies(*Mg7_data)
inj_Ek_O4 = InjectionEnergies(*O4_data)

### Test with normal PS-SPS stripping
print('---- Mg7+ ----\n')
inj_Ek_Mg7.calculate_all_gammas()
inj_Ek_Mg7.print_all_gammas()

print('\n---- O4+ ----\n')
inj_Ek_O4.calculate_all_gammas()
inj_Ek_O4.print_all_gammas()