#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First test of the CERN Injector Chain for different ions
- by Elias Waagaard 
"""
import pandas as pd
import numpy as np

class CERN_Injector_Chain:
    """
    Class to represent the CERN Injector Chain for different ions 
    """
    def __init__(self, ion_type, ion_data):
        self.ion_type = ion_type
        self.ion_data = ion_data[ion_type]
        self.mass_GeV = self.ion_data['mass [GeV]']
        self.Z = self.ion_data['Z']
        self.A = self.ion_data['A']
        self.Q = self.ion_data['Q before stripping']
        self.strip_ratio = self.ion_data['LEIR-PS Stripping Efficiency']
        self.linac3_current_uA = self.ion_data['Linac3 current [uA]']
        
        self.mass_GeV_Pb82_0 = 193.687 # GeV for reference case 
    
    
    def beta(self, gamma):
        """
        Relativistic beta factor from gamma factor 
        """
        return np.sqrt(1 - 1/gamma**2)
    
    
    def spaceChargeScalingFactor(self, Nb, Z, m, gamma, epsilon, sigma_z):
        """
        Approximate scaling equation for linear space charge tune shift, 
        from Eq (1) in Hannes' and Isabelle's space charge report
        at https://cds.cern.ch/record/2749453
        (assuming lattice integral is constant and that this scaling 
        stays constant, i.e. with  the same space charge tune shift 
        """
        beta = self.beta(gamma)
        return Nb*Z**2/(m *beta*gamma**2*epsilon*sigma_z)
    
    
    def linearIntensityLimit(self, Z, m, gamma, Nb_0, Z_0, m_0, gamma_0):
        """
        Linear intensity limit for new ion species for given bunch intensity 
        Nb_0 and parameters gamma_0, Z_0, m_0 from other species - assuming
        emittance and bunch length are constant for all ion species
        """
        beta_0 = self.beta(gamma_0)
        beta = self.beta(gamma)
        return Nb_0*(m/m_0)*(Z_0**2/Z**2)*(beta/beta_0)*(gamma**2/gamma_0**2)   
    
    
    def linac3(self):
        # Calculate the relativistic Lorentz factor at the entrance and exit of Linac3
        pass
    
    
    def leir(self):
        """
        Calculate gamma at entrance and exit of the LEIR and transmitted bunch intensity 
        using known Pb ion values from LIU report on 
        https://edms.cern.ch/ui/file/1420286/2/LIU-Ions_beam_parameter_table.pdf
        """
        # Estimate gamma at extraction
        E_kin_per_A_inj = 4.2e-3 # kinetic energy per nucleon in LEIR before RF capture, same for all species
        E_kin_per_A_extr = 7.22e-2 # kinetic energy per nucleon in LEIR at exit, same for all species
        self.gamma_leir_inj = (self.mass_GeV + E_kin_per_A_inj * self.A)/self.mass_GeV
        self.gamma_leir_extr = (self.mass_GeV + E_kin_per_A_extr * self.A)/self.mass_GeV
        
        # Calculate reference energy case for Pb54+
        self.gamma_leir_extr_Pb54_0 = (self.mass_GeV_Pb82_0 + E_kin_per_A_extr * 208)/self.mass_GeV_Pb82_0
        
        # Estimate number of charges at extraction - 10e10 charges for Pb54+, use this as scaling 
        Nb_Pb54_0_extr = 10e10/54.0
        self.Nb_leir_extr = self.linearIntensityLimit(Z=self.Z, 
                                       m = self.mass_GeV, 
                                       gamma = self.gamma_leir_extr,  
                                       Nb_0 = Nb_Pb54_0_extr, 
                                       Z_0 = 54, # partially stripped charged state 
                                       m_0 = self.mass_GeV_Pb82_0,  
                                       gamma_0 = self.gamma_leir_extr_Pb54_0  # use gamma at extraction
                                       )
        
        self.N_charges_leir_extr = self.Nb_leir_extr*self.Q  # number of outgoing charges, before any stripping
    
    
    def ps(self):
        """
        Calculate gamma at entrance and exit of the PS and transmitted bunch intensity 
        using known Pb ion values from LIU report on 
        https://edms.cern.ch/ui/file/1420286/2/LIU-Ions_beam_parameter_table.pdf
        """
        # Estimate gamma at extraction
        E_kin_per_A_inj = 7.22e-2 # GeV/nucleon according to LIU design report 
        E_kin_per_A_extr = 5.9 # GeV/nucleon according to LIU design report 
        self.gamma_ps_inj = (self.mass_GeV + E_kin_per_A_inj * self.A)/self.mass_GeV
        self.gamma_ps_extr = (self.mass_GeV + E_kin_per_A_extr * self.A)/self.mass_GeV
        
        # Calculate reference energy case for Pb54+
        self.gamma_ps_extr_Pb54_0 = (self.mass_GeV_Pb82_0 + E_kin_per_A_extr * 208)/self.mass_GeV_Pb82_0
        
        # Estimate number of charges at extraction - around 8e10 charges for Pb54+, use this as scaling 
        Nb_Pb54_0_extr = 8e10/54.0  # CHECK WITH NICOLO ! 
        self.Nb_ps_extr = self.linearIntensityLimit(Z=self.Z, 
                                       m = self.mass_GeV, 
                                       gamma = self.gamma_ps_extr,  
                                       Nb_0 = Nb_Pb54_0_extr, 
                                       Z_0 = 54, # partially stripped charged state 
                                       m_0 = self.mass_GeV_Pb82_0,  
                                       gamma_0 = self.gamma_ps_extr_Pb54_0  # use gamma at extraction
                                       )
        
        self.N_charges_ps_extr = self.Nb_ps_extr*self.Q  # number of outgoing charges, before any stripping
    
    
    def sps(self):
        """
        Calculate gamma at entrance and exit of the SPS, and transmitted bunch intensity 
        using known Pb ion values from LIU report on
        https://edms.cern.ch/ui/file/1420286/2/LIU-Ions_beam_parameter_table.pdf
        """
        # Calculate gamma at injection
        E_kin_per_A_inj = 5.9 # GeV/nucleon according to LIU design report 
        E_kin_per_A_extr = 176.4 # GeV/nucleon according to LIU design report 
        self.gamma_sps_inj = (self.mass_GeV + E_kin_per_A_inj * self.A)/self.mass_GeV
        self.gamma_sps_extr = (self.mass_GeV + E_kin_per_A_extr * self.A)/self.mass_GeV
        
        # Calculate reference energy case for Pb82+
        self.gamma_sps_extr_Pb82_0 = (self.mass_GeV_Pb82_0 + E_kin_per_A_extr * 208)/self.mass_GeV_Pb82_0
        
        # Calculate outgoing intensity from linear scaling 
        Nb_Pb82_0_extr = 2.21e8 # outgoing ions per bunch from SPS (2015 values), with 62% transmission
        self.Nb_sps_extr = self.linearIntensityLimit(Z = self.Z, 
                                       m = self.mass_GeV, 
                                       gamma = self.gamma_sps_extr,  # RODERIK USES GAMMA AT INJECTION, BUT SHOULD BE EXTRACTION?!  
                                       Nb_0 = Nb_Pb82_0_extr, 
                                       Z_0 = 82, 
                                       m_0 = self.mass_GeV_Pb82_0,  
                                       gamma_0 = self.gamma_sps_extr_Pb82_0  # use gamma at extraction
                                       )
    
        self.N_charges_sps_extr = self.Nb_sps_extr*self.Z  # number of outgoing charges
        
        #self.storedEnergy_at_7TeV_in_MJ =  ### WHAT IS THE GOOD CONVERSION? 
    
    def stripper_foil(self):
        # Calculate the stripping and transmission fraction for each ion type
        pass
    
    
    def space_charge_tune_shift(self):
        # Calculate the space charge tune shift for each accelerator
        pass
    
    
    def simulate_injection(self):
        # Call methods in sequence
        self.linac3()
        self.leir()
        self.stripper_foil()
        self.ps()
        self.sps()
        self.space_charge_tune_shift()


# Test the class 
if __name__ == '__main__':
    ion_type = 'O'
    ion_data = pd.read_csv("Data/Ion_species.csv", sep=';', header=0, index_col=0).T
    
    # Instantiate the Injector Chain and simulate injection
    cern_chain = CERN_Injector_Chain(ion_type, ion_data)
    cern_chain.simulate_injection()
