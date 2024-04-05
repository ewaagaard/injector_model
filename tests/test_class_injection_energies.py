"""
General test class to check injection energies across the ion injector chain
"""
from injector_model import InjectionEnergies
import pandas as pd
import numpy as np

# Test values for Pb
A, Q_low, m_ion_in_u, Z = 208, 54, 207.9766525, 82

class TestInjectionEnergies:

    def test_Pb_injection_energies():
        """
        Test Pb54+, with and without moving stripping to LEIR-PS 
        """
        inj_energies = InjectionEnergies(A, Q_low, m_ion_in_u, Z)
    
        # Test with normal PS-SPS stripping
        inj_energies.calculate_all_gammas()
        inj_energies.print_all_gammas()
        
    
# Execute some test statements 
if __name__ == "__main__":
    test_E = InjectionEnergies(A, Q_low, m_ion_in_u, Z)
    TestInjectionEnergies.test_Pb_injection_energies()
