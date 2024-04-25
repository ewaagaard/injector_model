"""
Test script to calculate typical momentum spread values (sigma delta) in each machine
"""
from injector_model import Momentum_Spread

# Instantiate class and run calculations
delta = Momentum_Spread()
delta.calculate_sigma_delta_LEIR_for_all_ions()
delta.calculate_sigma_delta_PS_for_all_ions()
delta.calculate_sigma_delta_SPS_for_all_ions()