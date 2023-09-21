#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation model of the CERN Injector Chain for different ions
- by Elias Waagaard 
"""
import pandas as pd
import numpy as np
import math
#from scipy.constants import e
from scipy import constants
from collections import defaultdict

class InjectorChain:
    """
    Representation of the CERN Injector Chain for different ions with linear space charge effects 
    to calculte maximum intensity limits
    following Roderik Bruce's example from 2021
    """
    def __init__(self, ion_type, 
                 ion_data, 
                 ion_type_ref='Pb',
                 nPulsesLEIR = 1,
                 LEIR_bunches = 2,
                 PS_splitting = 2,
                 account_for_SPS_transmission=True,
                 LEIR_PS_strip=False,
                 consider_PS_space_charge_limit=True,
                 use_gammas_ref=False,
                 higher_brho_LEIR=False,
                 save_path_csv = 'output/csv_tables'
                 ):
        
        self.full_ion_data = ion_data
        self.LEIR_PS_strip = LEIR_PS_strip
        self.higher_brho_LEIR = higher_brho_LEIR
        self.brho_string = '_higher_brho_LEIR' if self.higher_brho_LEIR else ''

        # Check whether to load relativistic gamma data from injection_energies
        self.use_gammas_ref = use_gammas_ref
        
        self.init_ion(ion_type)
        self.debug_mode = False
        self.account_for_SPS_transmission = account_for_SPS_transmission
        self.consider_PS_space_charge_limit = consider_PS_space_charge_limit
        
        # Initiate values for PS B-field
        self.PS_MinB    = 383 * 1e-4 # [T] - minimum magnetic field in PS, (Gauss to Tesla) from Heiko Damerau
        self.PS_MaxB    = 1.26 # [T] - minimum magnetic field in PS, from reyes Alemany Fernandez
        self.PS_rho     = 70.1206 # [m] - PS bending radius 
        
        # Rules for splitting and bunches 
        self.nPulsesLEIR = nPulsesLEIR
        self.LEIR_bunches = LEIR_bunches
        self.PS_splitting = PS_splitting
        
        # Also initiate reference values
        self.ion_type_ref = ion_type_ref
        self.ion0_referenceValues() 

        # Save path
        self.save_path = save_path_csv

        #For printing some reference values
        if self.debug_mode:
            print(f"Initiate. Type: {self.ion_type}")
            print("Q = {}, Z = {}\n".format(self.Q, self.Z))


    def init_ion(self, ion_type):
        """
        Initialize ion species for a given type 
        """
        self.ion_type = ion_type
        self.ion_data = self.full_ion_data[ion_type]
        self.mass_GeV = self.ion_data['mass [GeV]']
        self.Z = self.ion_data['Z']
        self.A = self.ion_data['A']
        self.Q = self.ion_data['Q before stripping']
        
        # Values from first tables in Roderik's notebook
        self.linac3_current = self.ion_data['Linac3 current [uA]'] * 1e-6
        self.linac3_pulseLength = self.ion_data['Linac3 pulse length [us]'] * 1e-6

        # General rules - stripping and transmission
        self.LEIR_injection_efficiency = 0.5
        self.LEIR_transmission = 0.8
        self.LEIR_PS_stripping_efficiency = self.ion_data['LEIR-PS Stripping Efficiency']
        self.PS_transmission = 0.9
        self.PS_SPS_transmission_efficiency = 1.0 # 0.9 is what we see today, but Roderik uses 1.0
        self.PS_SPS_strip = not self.LEIR_PS_strip  # if we have LEIR-PS stripping, no stripping PS-SPS
        self.PS_SPS_stripping_efficiency = 0.9  # default value until we have other value
        self.SPS_transmission = 0.62
        self.SPS_slipstacking_transmission = 1.0
        
        # Use gammas from calculated reference module
        if self.use_gammas_ref:
            self.load_ion_energy()


    def load_ion_energy(self):
        """
        Loads calculated ion energies for each ion type from the ion_injection_energies module
        """
        # Load ion energy data depending on where stripping is made 
        if self.LEIR_PS_strip:
            self.ion_energy_data = pd.read_csv('../data/injection_energies/ion_injection_energies_LEIR_PS_strip{}.csv'.format(self.brho_string), index_col=0)
        else:
            self.ion_energy_data = pd.read_csv('../data/injection_energies/ion_injection_energies_PS_SPS_strip{}.csv'.format(self.brho_string), index_col=0)
        
        key = str(int(self.Q)) + self.ion_type + str(int(self.A))
        ion_energy = self.ion_energy_data.loc[key]
        
        # Load reference injection energies
        self.LEIR_gamma_inj_ref = ion_energy['LEIR_gamma_inj']
        self.LEIR_gamma_extr_ref = ion_energy['LEIR_gamma_extr']
        self.PS_gamma_inj_ref = ion_energy['PS_gamma_inj']
        self.PS_gamma_extr_ref = ion_energy['PS_gamma_extr']
        self.SPS_gamma_inj_ref = ion_energy['SPS_gamma_inj']
        self.SPS_gamma_extr_ref = ion_energy['SPS_gamma_extr']

    def beta(self, gamma):
        """
        Relativistic beta factor from gamma factor 
        """
        return np.sqrt(1 - 1/gamma**2)
    

    def calcMomentum_from_gamma(self, gamma, q):
        """
        Calculates mometum from relativistic gamma and charge (number of elementary charges) 
        considering the electrons that have been stripped 
        """
        # Subtract missing electron mass, also expressed in GeV
        mass_in_eV_stripped = 1e9 * self.mass_GeV - (self.Z - q) * 1e6 * constants.physical_constants['electron mass energy equivalent in MeV'][0]  
        beta = np.sqrt(1 - 1/gamma**2)
        p = gamma * mass_in_eV_stripped * beta # in eV/c, so as mass is already in eV/c^2 then a factor c is not needed 
        return p


    def calcBrho(self, p, q):
        """
        Calculates Brho from momentum [eV/c] and charge (number of elementary charges) 
        """
        Brho = p / (q * constants.c) # in Tm
        return Brho    
    
    
    def spaceChargeScalingFactor(self, Nb, m, gamma, epsilon, sigma_z, fully_stripped=True):
        """
        Approximate scaling equation for linear space charge tune shift, 
        from Eq (1) in Hannes' and Isabelle's space charge report
        at https://cds.cern.ch/record/2749453
        (assuming lattice integral is constant and that this scaling 
        stays constant, i.e. with  the same space charge tune shift 
        - for now we ignore constants for a simpler expression
        """
        # Specify if fully stripped ion or not
        if fully_stripped:
            charge = self.Z
        else:
            charge = self.Q
        beta = self.beta(gamma)
        return Nb*charge**2/(m *beta*gamma**2*epsilon*sigma_z)
    
    
    def linearIntensityLimit(self, m, gamma, Nb_0, charge_0, m_0, gamma_0, fully_stripped=True):
        """
        Linear intensity limit for new ion species for given bunch intensity 
        Nb_0 and parameters gamma_0, charge0, m_0 from reference ion species - assuming
        that space charge stays constant, and that
        emittance and bunch length are constant for all ion species
        """
        # Specify if fully stripped ion or not
        if fully_stripped:
            charge = self.Z
        else:
            charge = self.Q
        beta_0 = self.beta(gamma_0)
        beta = self.beta(gamma)
        linearIntensityFactor = (m/m_0)*(charge_0/charge)**2*(beta/beta_0)*(gamma/gamma_0)**2   
        
        if self.debug_mode:
            print(f"SPS intensity limit. Type: {self.ion_type}")
            print("Fully stripped: {}".format(fully_stripped))
            print("Q = {}, Z = {}".format(self.Q, self.Z))
            print("Nb_0 = {:.2e}".format(Nb_0))
            print("m = {:.2e} GeV, m0 = {:.2e} GeV".format(m, m_0))
            print("charge = {:.1f}, charge_0 = {:.1f}".format(charge, charge_0))
            print("beta = {:.5f}, beta_0 = {:.5f}".format(beta, beta_0))
            print("gamma = {:.3f}, gamma_0 = {:.3f}".format(gamma, gamma_0))
            print('Linear intensity factor: {:.3f}\n'.format(linearIntensityFactor))
        return Nb_0*linearIntensityFactor 
    
    
    def ion0_referenceValues(self):
        """
        Sets bunch intensity Nb, gamma factor and mass of reference ion species 
        As of now, use reference values from Pb from Hannes 
        For injection and extraction energy, use known Pb ion values from LIU report on 
        https://edms.cern.ch/ui/file/1420286/2/LIU-Ions_beam_parameter_table.pdf
        """
        # LEIR
        self.E_kin_per_A_LEIR_inj = 4.2e-3 # kinetic energy per nucleon in LEIR before RF capture, same for all species
        self.E_kin_per_A_LEIR_extr = 7.22e-2 # kinetic energy per nucleon in LEIR at exit, same for all species
        
        # PS
        self.E_kin_per_A_PS_inj = 7.22e-2 # GeV/nucleon according to LIU design report 
        self.E_kin_per_A_PS_extr = 5.9 # GeV/nucleon according to LIU design report 
        
        # SPS
        self.E_kin_per_A_SPS_inj = 5.9 # GeV/nucleon according to LIU design report 
        self.E_kin_per_A_SPS_extr = 176.4 # GeV/nucleon according to LIU design report 
        
        # As of now, reference data exists only for Pb54+
        if self.ion_type_ref == 'Pb':    
            
            # Pb ion values
            self.m0_GeV = 193.687 # rest mass in GeV for Pb reference case 
            self.Z0 = 82.0
            
            # LEIR - reference case for Pb54+ --> BEFORE stripping
            self.Nq0_LEIR_extr = 10e10  # number of observed charges extracted at LEIR
            self.Q0_LEIR = 54.0
            self.Nb0_LEIR_extr = self.Nq0_LEIR_extr/self.Q0_LEIR
            self.gamma0_LEIR_inj = (self.m0_GeV + self.E_kin_per_A_LEIR_inj * 208)/self.m0_GeV
            self.gamma0_LEIR_extr = (self.m0_GeV + self.E_kin_per_A_LEIR_extr * 208)/self.m0_GeV
            
            # PS - reference case for Pb54+ --> BEFORE stripping
            self.Nq0_PS_extr =  6e10 # from November 2022 ionlifetime MD, previously 8e10  # number of observed charges extracted at PS for nominal beam
            self.Q0_PS = 54.0
            self.Nb0_PS_extr = self.Nq0_PS_extr/self.Q0_PS
            self.gamma0_PS_inj = (self.m0_GeV + self.E_kin_per_A_PS_inj * 208)/self.m0_GeV
            self.gamma0_PS_extr = (self.m0_GeV + self.E_kin_per_A_PS_extr * 208)/self.m0_GeV
            
            # SPS - reference case for Pb82+ --> AFTER stripping
            if not self.account_for_SPS_transmission:
                self.SPS_transmission = 1.0
            self.Nb0_SPS_extr = 2.21e8/self.SPS_transmission # outgoing ions per bunch from SPS (2015 values), adjusted for 62% transmission
            self.Q0_SPS = 82.0
            self.Nq0_SPS_extr = self.Nb0_SPS_extr*self.Q0_SPS
            self.gamma0_SPS_inj = (self.m0_GeV + self.E_kin_per_A_SPS_inj * 208)/self.m0_GeV
            self.gamma0_SPS_extr = (self.m0_GeV + self.E_kin_per_A_SPS_extr * 208)/self.m0_GeV
    
        else:
            raise ValueError('Other reference ion type than Pb does not yet exist!')
   
    
    def leir(self):
        """
        Calculate gamma at entrance and exit of the LEIR and transmitted bunch intensity 
        using known Pb ion values from LIU report on 
        https://edms.cern.ch/ui/file/1420286/2/LIU-Ions_beam_parameter_table.pdf
        """
        # Estimate gamma at extraction
        if self.use_gammas_ref:
            self.gamma_LEIR_inj = self.LEIR_gamma_inj_ref
            self.gamma_LEIR_extr = self.LEIR_gamma_extr_ref
        else: 
            self.gamma_LEIR_inj = (self.mass_GeV + self.E_kin_per_A_LEIR_inj * self.A)/self.mass_GeV
            
            self.gamma_LEIR_extr =  np.sqrt(
                                    1 + ((self.Q / 54) / (self.mass_GeV/self.m0_GeV))**2
                                    * (self.gamma0_LEIR_extr**2 - 1)
                                    )
            
            #self.gamma_LEIR_extr = (self.mass_GeV + self.E_kin_per_A_LEIR_extr * self.A)/self.mass_GeV         
                
        # Estimate number of charges at extraction - 10e10 charges for Pb54+, use this as scaling 
        self.Nb_LEIR_extr = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_LEIR_inj,  
                                               Nb_0 = self.Nb0_LEIR_extr, 
                                               charge_0 = self.Q0_LEIR, # partially stripped charged state 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_LEIR_inj,  # use gamma at extraction
                                               fully_stripped=False
                                               )
        
        self.Nq_LEIR_extr = self.Nb_LEIR_extr*self.Q  # number of outgoing charges, before any stripping


    def stripper_foil_leir_ps(self):
        """
        Stripper foil between LEIR and PS 
        """
        pass    

    
    def ps(self):
        """
        Calculate gamma at entrance and exit of the PS and transmitted bunch intensity 
        """
        # Estimate gamma at extraction
        if self.use_gammas_ref:
            self.gamma_PS_inj = self.PS_gamma_inj_ref
            self.gamma_PS_extr = self.PS_gamma_extr_ref
        else:         

            self.gamma_PS_inj =  self.gamma_LEIR_extr
            self.gamma_PS_extr =  np.sqrt(
                                    1 + (((self.Z if self.LEIR_PS_strip else self.Q) / 54) / (self.mass_GeV/self.m0_GeV))**2
                                    * (self.gamma0_PS_extr**2 - 1)
                                    )
        
        # Estimate number of charges at extraction
        self.Nb_PS_extr = self.linearIntensityLimit(
                                                m = self.mass_GeV, 
                                                gamma = self.gamma_PS_inj,  
                                                Nb_0 = self.Nb0_PS_extr, 
                                                charge_0 = self.Q0_PS, # partially stripped charged state 
                                                m_0 = self.m0_GeV,  
                                                gamma_0 = self.gamma0_PS_inj,  # use gamma at extraction,
                                                fully_stripped=False
                                                )
        self.Nq_PS_extr = self.Nb_PS_extr*self.Q  # number of outgoing charges, before any stripping


    def stripper_foil_ps_sps(self):
        """
        Stripper foil between PS and SPS 
        """
        pass   
    
    
    def sps(self):
        """
        Calculate gamma at entrance and exit of the SPS, and transmitted bunch intensity 
        Space charge limit comes from gamma at injection
        """
        # Calculate gamma at injection, simply scaling with the kinetic energy per nucleon as of today
        if self.use_gammas_ref:
            self.gamma_SPS_inj = self.SPS_gamma_inj_ref
            self.gamma_SPS_extr = self.SPS_gamma_extr_ref
        else:     
            # In his notebook, Roderik considers same magnetic rigidity at PS extraction for all ion species: Brho = P/Q, P = m*gamma*beta*c
            self.gamma_SPS_inj =  np.sqrt(
                                    1 + (((self.Z if self.LEIR_PS_strip else self.Q) / 54) / (self.mass_GeV/self.m0_GeV))**2
                                    * (self.gamma0_SPS_inj**2 - 1)
                                )
            self.gamma_SPS_extr = (self.mass_GeV + self.E_kin_per_A_SPS_extr * self.A)/self.mass_GeV
        
        # Calculate outgoing intensity from linear scaling 
        self.Nb_SPS_extr = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_SPS_inj,  
                                               Nb_0 = self.Nb0_SPS_extr, # what we can successfully accelerate today
                                               charge_0 = self.Q0_SPS, 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_SPS_inj,  # use gamma at extraction
                                               fully_stripped=True
                                               )
    
        self.Nq_SPS_extr = self.Nb_SPS_extr*self.Z  # number of outgoing charges
      
    
    def simulate_injection_SpaceCharge_limit(self):
        """
        Simulate space charge limit in full injection through Linac3, LEIR, PS and SPS for a given ion type
        """
        self.leir()
        self.ps()
        self.sps()


    def simulate_SpaceCharge_intensity_limit_all_ions(self, return_dataframe=True):
        """
        Calculate intensity limits with linear space charge in 
        Linac3, LEIR, PS and SPS for all ions given in table
        """
        
        # Initiate row of ions per bunch (Nb) and charges per bunch (Nq)
        self.ion_Nb_data = pd.DataFrame(index=self.full_ion_data.transpose().index)
        self.ion_Nb_data["Nb_LEIR"] = np.NaN
        self.ion_Nb_data["Nq_LEIR"] = np.NaN
        self.ion_Nb_data["Nb_PS"] = np.NaN
        self.ion_Nb_data["Nq_PS"] = np.NaN
        self.ion_Nb_data["Nb_SPS"] = np.NaN
        self.ion_Nb_data["Nq_SPS"] = np.NaN
        
        # Create dataframes or gamma injection and extraction data 
        self.ion_gamma_inj_data = pd.DataFrame(columns=['LEIR', 'PS', 'SPS'], 
                                               index=self.full_ion_data.transpose().index)
        self.ion_gamma_extr_data = self.ion_gamma_inj_data.copy()
        
        # Iterate over all ions in data 
        for i, ion_type in enumerate(self.full_ion_data.columns):
            self.init_ion(ion_type)
            self.simulate_injection_SpaceCharge_limit()
            
            # Add the intensities into a table 
            self.ion_Nb_data["Nb_LEIR"][ion_type] = self.Nb_LEIR_extr
            self.ion_Nb_data["Nq_LEIR"][ion_type] = self.Nq_LEIR_extr
            self.ion_Nb_data["Nb_PS"][ion_type] = self.Nb_PS_extr
            self.ion_Nb_data["Nq_PS"][ion_type] = self.Nq_PS_extr
            self.ion_Nb_data["Nb_SPS"][ion_type] = self.Nb_SPS_extr
            self.ion_Nb_data["Nq_SPS"][ion_type] = self.Nq_SPS_extr

            # Add the gamma of injection and extraction into a table 
            self.ion_gamma_inj_data['LEIR'][ion_type] = self.gamma_LEIR_inj
            self.ion_gamma_extr_data['LEIR'][ion_type] = self.gamma_LEIR_extr
            self.ion_gamma_inj_data['PS'][ion_type] = self.gamma_PS_inj
            self.ion_gamma_extr_data['PS'][ion_type] = self.gamma_PS_extr
            self.ion_gamma_inj_data['SPS'][ion_type] = self.gamma_SPS_inj
            self.ion_gamma_extr_data['SPS'][ion_type] = self.gamma_SPS_extr

        if return_dataframe:
            return self.ion_Nb_data 


    def calculate_LHC_bunch_intensity(self):
        """
        Estimate LHC bunch intensity for a given ion species
        through Linac3, LEIR, PS and SPS considering all the limits of the injectors
        """
        self.simulate_injection_SpaceCharge_limit()
        
        ### LINAC3 ### 
        ionsPerPulseLinac3 = (self.linac3_current * self.linac3_pulseLength) / (self.Q * constants.e)
        
        ### LEIR ###
        spaceChargeLimitLEIR = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_LEIR_inj,  
                                               Nb_0 = self.Nb0_LEIR_extr, 
                                               charge_0 = self.Q0_LEIR, # partially stripped charged state 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_LEIR_inj,  # use gamma at LEIR extraction
                                               fully_stripped=False
                                               )
        
        nPulsesLEIR = (min(7, math.ceil(spaceChargeLimitLEIR / (ionsPerPulseLinac3 * self.LEIR_injection_efficiency))) if self.nPulsesLEIR == 0 else self.nPulsesLEIR)
        totalIntLEIR = ionsPerPulseLinac3 * nPulsesLEIR * self.LEIR_injection_efficiency
        ionsPerBunchExtractedLEIR = self.LEIR_transmission * np.min([totalIntLEIR, spaceChargeLimitLEIR]) / self.LEIR_bunches
        LEIR_space_charge_limit_hit = True if totalIntLEIR > spaceChargeLimitLEIR else False 
        
        #### PS ####
        ionsPerBunchInjectedPS = ionsPerBunchExtractedLEIR * (self.LEIR_PS_stripping_efficiency if self.LEIR_PS_strip else 1)
        spaceChargeLimitPS = self.linearIntensityLimit(
                                        m = self.mass_GeV, 
                                        gamma = self.gamma_PS_inj,  
                                        Nb_0 = self.Nb0_PS_extr, 
                                        charge_0 = self.Q0_PS, # partially stripped charged state 
                                        m_0 = self.m0_GeV,  
                                        gamma_0 = self.gamma0_PS_inj,  # use gamma at PS inj
                                        fully_stripped = self.LEIR_PS_strip # fully stripped if LEIR-PS strip
                                        )
        
        # Check that injected momentum is not too low for the PS B-field
        q_PS = self.Z if self.LEIR_PS_strip else self.Q
        self.p_PS_inj = self.calcMomentum_from_gamma(self.gamma_PS_inj, q_PS)
        self.Brho_PS_inj = self.calcBrho(self.p_PS_inj, q_PS) # same as LEIR extraction if no stripping, else will be different 
        B_PS_inj = self.Brho_PS_inj / self.PS_rho
        if B_PS_inj < self.PS_MinB:
            self.PS_B_field_is_too_low = True
        elif B_PS_inj > self.PS_MaxB:
            print("\nA = {}, Q_low = {}, m_ion = {:.2f} u, Z = {}".format(self.A, self.Q_low, self.m_ion_in_u, self.Z))
            print('B = {:.4f} in PS at injection is too HIGH!'.format(B_PS_inj))
            raise ValueError("B field in PS is too high!")
        else:
            self.PS_B_field_is_too_low = False
        
        # If space charge limit in PS is considered, choose the minimum between the SC limit and the extracted ionsPerBunchPS
        if self.consider_PS_space_charge_limit:
            ionsPerBunchPS = min(spaceChargeLimitPS, ionsPerBunchInjectedPS)
            #if spaceChargeLimitPS < ionsPerBunchInjectedPS:
            #    print("\nIon type: {}".format(self.ion_type))
            #    print("Space charge limit PS: {:.3e} vs max injected ions per bunch PS: {:.3e}".format(spaceChargeLimitPS, ionsPerBunchInjectedPS))
        else:
            ionsPerBunchPS = ionsPerBunchInjectedPS 
        PS_space_charge_limit_hit = True if ionsPerBunchInjectedPS > spaceChargeLimitPS else False 
        ionsPerBunchExtracted_PS = ionsPerBunchPS * self.PS_transmission / self.PS_splitting # maximum intensity without SC
        
        # Calculate ion transmission for SPS 
        ionsPerBunchSPSinj = ionsPerBunchExtracted_PS * (self.PS_SPS_transmission_efficiency if self.Z == self.Q or self.LEIR_PS_strip else self.PS_SPS_stripping_efficiency)
        spaceChargeLimitSPS = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_SPS_inj,  
                                               Nb_0 = self.Nb0_SPS_extr, 
                                               charge_0 = self.Q0_SPS, 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_SPS_inj, # use gamma at SPS inj
                                               fully_stripped=True
                                               )
        SPS_space_charge_limit_hit = True if ionsPerBunchSPSinj > spaceChargeLimitSPS else False
        ionsPerBunchLHC = min(spaceChargeLimitSPS, ionsPerBunchSPSinj) * self.SPS_transmission * self.SPS_slipstacking_transmission

        result = {
            "Ion": self.ion_type,
            "chargeBeforeStrip": int(self.Q),
            "atomicNumber": int(self.Z),
            "massNumber": int(self.A),
            "Linac3_current [A]": self.linac3_current,
            "Linac3_pulse_length [s]": self.linac3_pulseLength, 
            "LEIR_numberofPulses": nPulsesLEIR,
            "LEIR_injection_efficiency": self.LEIR_injection_efficiency, 
            "LEIR_splitting": self.LEIR_bunches,
            "LEIR_transmission": self.LEIR_transmission, 
            "PS_splitting": self.PS_splitting, 
            "PS_transmission": self.PS_transmission, 
            "PS_SPS_stripping_efficiency": self.PS_SPS_stripping_efficiency, 
            "SPS_transmission": self.SPS_transmission, 
            "Linac3_ionsPerPulse": ionsPerPulseLinac3,
            "LEIR_maxIntensity": totalIntLEIR,
            "LEIR_space_charge_limit": spaceChargeLimitLEIR,
            "LEIR_extractedIonPerBunch": ionsPerBunchExtractedLEIR,
            "PS_space_charge_limit": spaceChargeLimitPS,
            "PS_maxIntensity": ionsPerBunchInjectedPS,
            "PS_ionsExtractedPerBunch":  ionsPerBunchExtracted_PS,
            "SPS_maxIntensity": ionsPerBunchSPSinj,
            "SPS_spaceChargeLimit": spaceChargeLimitSPS,
            "LHC_ionsPerBunch": ionsPerBunchLHC,
            "LHC_chargesPerBunch": ionsPerBunchLHC * self.Z,
            "LEIR_gamma_inj": self.gamma_LEIR_inj,
            "LEIR_gamma_extr": self.gamma_LEIR_extr,
            "PS_gamma_inj": self.gamma_PS_inj,
            "PS_gamma_extr": self.gamma_PS_extr,
            "SPS_gamma_inj": self.gamma_SPS_inj,
            "SPS_gamma_extr": self.gamma_SPS_extr,
            "PS_B_field_is_too_low": self.PS_B_field_is_too_low,
            "LEIR_space_charge_limit_hit": LEIR_space_charge_limit_hit,
            "consider_PS_space_charge_limit": self.consider_PS_space_charge_limit,
            "PS_space_charge_limit_hit": PS_space_charge_limit_hit,
            "SPS_space_charge_limit_hit": SPS_space_charge_limit_hit,
            "LEIR_ratio_SC_limit_maxIntensity": spaceChargeLimitLEIR / totalIntLEIR,
            "PS_ratio_SC_limit_maxIntensity": spaceChargeLimitPS / ionsPerBunchInjectedPS,
            "SPS_ratio_SC_limit_maxIntensity": spaceChargeLimitSPS / ionsPerBunchSPSinj
        }
        

        # Add key of LEIR-PS stripping efficiency if this is done 
        if self.LEIR_PS_strip:
            result["LEIR_PS_strippingEfficiency"] = self.LEIR_PS_stripping_efficiency
    
        return result


    def calculate_LHC_bunch_intensity_all_ion_species(self, save_csv=False, output_name='output'):
        """
        Estimate LHC bunch intensity for all ion species provided in table
        through Linac3, LEIR, PS and SPS considering all the limits of the injectors
        """
        # Initialize full dicionary
        full_result = defaultdict(list)
        
        # Iterate over all ions in data 
        for i, ion_type in enumerate(self.full_ion_data.columns):
            # Initiate the correct ion
            self.init_ion(ion_type)
            result = self.calculate_LHC_bunch_intensity()
         
            # Append the values to the corresponding key 
            for key, value in result.items():
                full_result[key].append(value)
            
            del result
            
        # Convert dictionary to dataframe 
        df_all_ions = pd.DataFrame(full_result)
        df_all_ions = df_all_ions.set_index("Ion")
        
        # Save CSV file if desired 
        if save_csv:
            float_columns = df_all_ions.select_dtypes(include=['float']).columns
            df_save = df_all_ions.copy()
            df_save[float_columns]  = df_save[float_columns].applymap(self.format_large_numbers)
            df_save = df_save.T
            df_save.to_csv("{}/{}.csv".format(self.save_path, output_name))
            
        return df_all_ions
    
    
    def space_charge_limit_effect_on_LHC_bunch_intensity(self):
        """
        Propagate estimated bunch intensity into LHC for a given ion species
        considering space charge limit only in LEIR, PS or SPS 
        --> considering first default baseline case
        """
        self.simulate_injection_SpaceCharge_limit()
    
        #### LEIR SPACE CHARGE LIMIT ####
        spaceChargeLimitLEIR = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_LEIR_inj,  
                                               Nb_0 = self.Nb0_LEIR_extr, 
                                               charge_0 = self.Q0_LEIR, # partially stripped charged state 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_LEIR_inj,  # use gamma at LEIR extraction
                                               fully_stripped=False
                                               )
        
        ionsPerBunchLHC_LEIR_lim = spaceChargeLimitLEIR * self.LEIR_transmission \
                                *  (self.LEIR_PS_stripping_efficiency if self.LEIR_PS_strip else 1) \
                                * self.PS_transmission * (self.PS_SPS_transmission_efficiency if self.Z == self.Q or self.LEIR_PS_strip else self.PS_SPS_stripping_efficiency) \
                                * self.SPS_transmission / (self.LEIR_bunches * self.PS_splitting)  # divide by number of pulses into LEIR and PS splitting
        

        #### PS SPACE CHARGE LIMIT ####
        spaceChargeLimitPS = self.linearIntensityLimit(
                                        m = self.mass_GeV, 
                                        gamma = self.gamma_PS_inj,  
                                        Nb_0 = self.Nb0_PS_extr, 
                                        charge_0 = self.Q0_PS, # partially stripped charged state 
                                        m_0 = self.m0_GeV,  
                                        gamma_0 = self.gamma0_PS_inj,  # use gamma at PS inj
                                        fully_stripped = self.LEIR_PS_strip # fully stripped if LEIR-PS strip
                                        )

        ionsPerBunchLHC_PS_lim = spaceChargeLimitPS * self.PS_transmission \
                                * (self.PS_SPS_transmission_efficiency if self.Z == self.Q or self.LEIR_PS_strip else self.PS_SPS_stripping_efficiency) \
                                * self.SPS_transmission / self.PS_splitting


        #### SPS SPACE CHARGE LIMIT ####
        spaceChargeLimitSPS = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_SPS_inj,  
                                               Nb_0 = self.Nb0_SPS_extr, 
                                               charge_0 = self.Q0_SPS, 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_SPS_inj, # use gamma at SPS inj
                                               fully_stripped=True
                                               )
        ionsPerBunchLHC_SPS_lim = spaceChargeLimitSPS * self.SPS_transmission * self.SPS_slipstacking_transmission        
        
        # Array of Nb injected into LHC considering SC limits in LEIR, PS and SPS
        Nb_SC_limits = np.array([ionsPerBunchLHC_LEIR_lim, ionsPerBunchLHC_PS_lim, ionsPerBunchLHC_SPS_lim])

        return Nb_SC_limits
        
    
    
    
    def format_large_numbers(self, x):
        """
        Converts large or small floats to exponential form 
        """
        if abs(x) >= 1e6 or abs(x) < 1e-3:
            return f'{x:.5e}'
        return x