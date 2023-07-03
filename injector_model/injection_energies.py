"""
Class to estimate injection energies of any ion species in LEIR, PS and SPS from
Brho data and ion mass data 
- 
- considers if we strip between PS-SPS (as of today) or between LEIR-PS 
"""

import numpy as np
from scipy import constants

class InjectionEnergies:
    """
    Estimate injection energies for every ion and charge state, with ion input data 'elements'
    """
    def __init__(self,
                 A,
                 Q_low, # charge state after linac3
                 m_ion_in_u, # mass in atomic units (Dalton)
                 Z,
                 LEIR_PS_strip=False
                 ):
        self.A = A
        self.Z = Z
        self.Q_low = Q_low
        self.Q_high = Z # charge state after next stripping - this is full stripping, meaning Z
        self.m_ion_in_u = m_ion_in_u
        self.m_ion = m_ion_in_u * constants.physical_constants['atomic mass unit-electron volt relationship'][0] 
        self.LEIR_PS_strip = LEIR_PS_strip  # if stripping occurs between LEIR-PS or PS-SPS (as today)

        # Magnetic rigidity of different accelerators 
        self.Linac3_E_per_u = 4200000.0 # [eV/nucleon] at extraction
        self.LEIR_Brho  = 4.8 # [Tm] @extraction
        self.PS_B       = 1.2368 # [T] - magnetic field in PS, from Heiko Damerau
        self.PS_MinB    = 383 * 1e-4 # [T] - minimum magnetic field in PS, (Gauss to Tesla) from Heiko Damerau
        self.PS_MaxB    = 1.26 # [T] - minimum magnetic field in PS, from reyes Alemany Fernandez
        self.PS_rho     = 70.1206 # [m] - PS bending radius 
        self.SPS_pLHC   = 450 * 1e9 # [eV/c proton-equivalent]
        self.SPS_Brho   = self.SPS_pLHC / constants.c # [Tm] - assuming same Brho can be used for ions

        self.gammas_are_calculated = False 


    def calculate_gamma_and_beta(self, q, e_kin_per_nucleon):
        """
        Calculate relativistic beta factor from charge state q and kinetic energy per nucleon
        q - number of elementary charges
        e_kin_per_nucleon - kinetic energy per nucleon in eV
        """         
        # Calculate mass of ion in electron volt 
        mass_in_u_stripped = self.m_ion_in_u - (self.Z - q) * constants.physical_constants['electron mass in u'][0] # subtract electrons
        mass_in_eV_stripped =  mass_in_u_stripped * constants.physical_constants['atomic mass unit-electron volt relationship'][0]
        E_tot = mass_in_eV_stripped + e_kin_per_nucleon * self.A # total kinetic energy in eV per particle at injection
        gamma = E_tot/mass_in_eV_stripped
        beta = self.beta_from_gamma(gamma)
        return gamma, beta


    def beta_from_gamma(self, gamma):
        """
        Relativistic beta factor from gamma factor 
        """
        return np.sqrt(1 - 1/gamma**2)


    def calcMomentum(self, Brho, q):
        """
        Calculates mometum from Brho [Tm] and charge (number of elementary charges) 
        """
        p = Brho * q * constants.c  # in eV/c
        return p
    

    def calcBrho(self, p, q):
        """
        Calculates Brho from momentum [eV/c] and charge (number of elementary charges) 
        """
        Brho = p / (q * constants.c) # in Tm
        return Brho


    def calcKineticEnergyPerNucleon(self, p, q):
        """
        Calculates kinetic energy in eV per atomic unit u
        """
        mass_in_u_stripped = self.m_ion_in_u - (self.Z - q) * constants.physical_constants['electron mass in u'][0] # subtract electrons
        mass_in_eV_stripped =  mass_in_u_stripped * constants.physical_constants['atomic mass unit-electron volt relationship'][0]
        E_0 = mass_in_eV_stripped # in [eV]
        E_tot = np.sqrt(p**2 + E_0**2)  # in [eV]
        return 1. / self.A * (E_tot - E_0) # in [eV/nucleon]
    

    def calculate_all_gammas(self):
        """
        Find all injection energies and gammas along the ion injector chain
        """
        ############ LEIR - still not stripped ############
        # LEIR injection
        self.E_kin_per_u_LEIR_inj = self.Linac3_E_per_u
        self.gamma_LEIR_inj, self.beta_LEIR_inj = self.calculate_gamma_and_beta(self.Q_low, self.Linac3_E_per_u) # injection, same E_kin for all ions
        
        # LEIR extraction
        self.p_LEIR_extr = self.calcMomentum(self.LEIR_Brho, self.Q_low)
        self.p_LEIR_extr_proton_equiv = self.p_LEIR_extr / self.Q_low
        self.E_kin_per_u_LEIR_extr = self.calcKineticEnergyPerNucleon(self.p_LEIR_extr, self.Q_low)
        self.gamma_LEIR_extr, self.beta_LEIR_extr = self.calculate_gamma_and_beta(self.Q_low, self.E_kin_per_u_LEIR_extr)

        ############ PS - check if stripped ############
        # PS injection - p_PS_inj same as incoming momentum from LEIR extraction, but Brho might change if we have stripping  
        q_PS = self.Q_high if self.LEIR_PS_strip else self.Q_low
        self.p_PS_inj = self.p_LEIR_extr
        self.p_PS_inj_proton_equiv = self.p_PS_inj / q_PS
        self.Brho_PS_inj = self.calcBrho(self.p_PS_inj, q_PS) # same as LEIR extraction if no stripping, else will be different  
        B_PS_inj = self.Brho_PS_inj / self.PS_rho

        # Check if Brho in PS is too low or too high 
        if B_PS_inj < self.PS_MinB:
            print("\nA = {}, Q_low = {}, m_ion = {:.2f} u, Z = {}".format(self.A, self.Q_low, self.m_ion_in_u, self.Z))
            print('B = {:.4f} in PS at injection is too LOW!\n'.format(B_PS_inj))
            self.PS_B_field_is_too_low = True
        elif B_PS_inj > self.PS_MaxB:
            print("\nA = {}, Q_low = {}, m_ion = {:.2f} u, Z = {}".format(self.A, self.Q_low, self.m_ion_in_u, self.Z))
            print('B = {:.4f} in PS at injection is too HIGH!'.format(B_PS_inj))
        else:
            self.PS_B_field_is_too_low = False

        # PS injection
        self.E_kin_per_u_PS_inj = self.calcKineticEnergyPerNucleon(self.p_PS_inj, q_PS)
        self.gamma_PS_inj, self.beta_PS_inj = self.calculate_gamma_and_beta(q_PS, self.E_kin_per_u_PS_inj) 

        # PS extraction
        self.Brho_PS_extr = self.PS_B * self.PS_rho  # magnetic rigidity when magnets have ramped 
        self.p_PS_extr = self.calcMomentum(self.Brho_PS_extr, q_PS)
        self.p_PS_extr_proton_equiv = self.p_PS_extr / q_PS
        self.E_kin_per_u_PS_extr = self.calcKineticEnergyPerNucleon(self.p_PS_extr, q_PS)
        self.gamma_PS_extr, self.beta_PS_extr = self.calculate_gamma_and_beta(q_PS, self.E_kin_per_u_PS_extr)

        ########### SPS - ions are fully stripped ###########
        # SPS injection 
        q_SPS = self.Q_high  # ions are fully stripped
        self.p_SPS_inj = self.p_PS_extr
        self.p_SPS_inj_proton_equiv = self.p_SPS_inj / q_SPS
        self.Brho_SPS_inj = self.calcBrho(self.p_SPS_inj, q_SPS) # same as PS extraction if no PS-SPS stripping, else will be different
        self.E_kin_per_u_SPS_inj = self.calcKineticEnergyPerNucleon(self.p_SPS_inj, q_SPS)
        self.gamma_SPS_inj, self.beta_SPS_inj = self.calculate_gamma_and_beta(q_SPS, self.E_kin_per_u_SPS_inj) 

        # SPS extraction
        self.Brho_SPS_extr = self.SPS_Brho
        self.p_SPS_extr = self.calcMomentum(self.Brho_SPS_extr, q_SPS)
        self.p_SPS_extr_proton_equiv = self.p_SPS_extr / q_SPS
        self.E_kin_per_u_SPS_extr = self.calcKineticEnergyPerNucleon(self.p_SPS_extr, q_SPS)
        self.gamma_SPS_extr, self.beta_SPS_extr = self.calculate_gamma_and_beta(q_SPS, self.E_kin_per_u_SPS_extr)


    def print_all_gammas(self):
        """
        Print the relativistic gamma and all the injection energies
        """
        if not self.gammas_are_calculated:
            self.calculate_all_gammas()
        print("\nA = {}, Q_low = {}, m_ion = {:.2f} u, Z = {}\n".format(self.A, self.Q_low, self.m_ion_in_u, self.Z))
        print("Stripping happens between LEIR-PS\n") if self.LEIR_PS_strip else print("Stripping happens between PS-SPS\n")

        print("gamma LEIR inj = {:.3f}".format(self.gamma_LEIR_inj))
        print("gamma LEIR extr = {:.3f}".format(self.gamma_LEIR_extr))
        print("gamma PS inj = {:.3f}".format(self.gamma_PS_inj))
        print("gamma PS extr = {:.3f}".format(self.gamma_PS_extr))
        print("gamma SPS inj = {:.3f}".format(self.gamma_SPS_inj))
        print("gamma SPS extr = {:.3f}".format(self.gamma_SPS_extr))
        
        print("\nEkin_per_u LEIR inj = {:.3} GeV/u".format(1e-9 * self.E_kin_per_u_LEIR_inj))
        print("Ekin_per_u LEIR extr = {:.3f} GeV/u".format(1e-9 * self.E_kin_per_u_LEIR_extr))
        print("Ekin_per_u PS inj = {:.3f} GeV/u".format(1e-9 * self.E_kin_per_u_PS_inj))
        print("Ekin_per_u PS extr = {:.3f} GeV/u".format(1e-9 * self.E_kin_per_u_PS_extr))
        print("Ekin_per_u SPS inj = {:.3f} GeV/u".format(1e-9 * self.E_kin_per_u_SPS_inj))
        print("Ekin_per_u SPS extr = {:.3f} GeV/u".format(1e-9 * self.E_kin_per_u_SPS_extr))

        print("PS_B_field_is_too_low: {}".format(self.PS_B_field_is_too_low))


    def return_all_gammas(self): 
        """
        Returns dictionaries of all gammas, Ekin per nucleon [GeV/u] and Brho for all accelerators
        """
        if not self.gammas_are_calculated:
            self.calculate_all_gammas()
        
        gamma_dict = {
            "LEIR_gamma_inj": self.gamma_LEIR_inj,
            "LEIR_gamma_extr": self.gamma_LEIR_extr,
            "PS_gamma_inj": self.gamma_PS_inj,
            "PS_gamma_extr": self.gamma_PS_extr,
            "SPS_gamma_inj": self.gamma_SPS_inj,
            "SPS_gamma_extr": self.gamma_SPS_extr,
            "LEIR_Ekin_per_u_inj": 1e-9 * self.E_kin_per_u_LEIR_inj,
            "LEIR_Ekin_per_u_extr": 1e-9 * self.E_kin_per_u_LEIR_extr,
            "PS_Ekin_per_u_inj": 1e-9 * self.E_kin_per_u_PS_inj,
            "PS_Ekin_per_u_extr": 1e-9 * self.E_kin_per_u_PS_extr,
            "SPS_Ekin_per_u_inj": 1e-9 * self.E_kin_per_u_SPS_inj,
            "SPS_Ekin_per_u_extr": 1e-9 * self.E_kin_per_u_SPS_extr,
            "p_LEIR_extr_proton_equiv": 1e-9 *self.p_LEIR_extr_proton_equiv,
            "p_PS_inj_proton_equiv": 1e-9 *self.p_PS_inj_proton_equiv,
            "p_PS_extr_proton_equiv": 1e-9 *self.p_PS_extr_proton_equiv,
            "p_SPS_inj_proton_equiv": 1e-9 *self.p_SPS_inj_proton_equiv, 
            "p_SPS_extr_proton_equiv": 1e-9 *self.p_SPS_extr_proton_equiv,
            "LEIR_Brho": self.LEIR_Brho,
            "PS_Brho_inj": self.Brho_PS_inj,
            "PS_Brho_extr": self.Brho_PS_extr,
            "SPS_Brho_inj": self.Brho_SPS_inj,
            "SPS_Brho_extr": self.Brho_SPS_extr,
            "PS_B_field_is_too_low": self.PS_B_field_is_too_low
        }
        
        return gamma_dict
