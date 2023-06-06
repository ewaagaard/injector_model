#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First test of the CERN Injector Chain for different ions
- by Elias Waagaard 
"""
import pandas as pd
import numpy as np
import math
from scipy.constants import e
import matplotlib.pyplot as plt
import matplotlib as mpl

class CERN_Injector_Chain:
    """
    Representation of the CERN Injector Chain for different ions with linear space charge effects 
    to calculte maximum intensity limits
    following Roderik Bruce's example from 2021
    """
    def __init__(self, ion_type, 
                 ion_data, 
                 ion_type_ref='Pb',
                 nPulsesLEIR = 1,
                 LEIR_bunches = 1,
                 PS_splitting = 1,
                 account_for_SPS_transmission=True,
                 LEIR_PS_strip=False
                 ):
        
        self.full_ion_data = ion_data
        self.LEIR_PS_strip = LEIR_PS_strip
        self.init_ion(ion_type)
        self.debug_mode = False
        self.account_for_SPS_transmission = account_for_SPS_transmission
        
        
        # Rules for splitting and bunches 
        self.nPulsesLEIR = nPulsesLEIR
        self.LEIR_bunches = LEIR_bunches
        self.PS_splitting = PS_splitting
        
        # Also initiate reference values
        self.ion_type_ref = ion_type_ref
        self.ion0_referenceValues() 

        ######### Reference mode for SPS space charge scaling ############
        self.use_Roderiks_gamma = True

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
        
        # Take Roderik's last linac3 values from Reyes table
        #self.linac3_pulseLength  = 200e-6
        #self.linac3_current = 70e-6

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


    def beta(self, gamma):
        """
        Relativistic beta factor from gamma factor 
        """
        return np.sqrt(1 - 1/gamma**2)
    
    
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
            self.Nq0_PS_extr = 8e10  # number of observed charges extracted at PS for nominal beam
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
        self.gamma_LEIR_inj = (self.mass_GeV + self.E_kin_per_A_LEIR_inj * self.A)/self.mass_GeV
        self.gamma_LEIR_extr = (self.mass_GeV + self.E_kin_per_A_LEIR_extr * self.A)/self.mass_GeV
                
        # Estimate number of charges at extraction - 10e10 charges for Pb54+, use this as scaling 
        self.Nb_LEIR_extr = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_LEIR_extr,  
                                               Nb_0 = self.Nb0_LEIR_extr, 
                                               charge_0 = self.Q0_LEIR, # partially stripped charged state 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_LEIR_extr,  # use gamma at extraction
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
        self.gamma_PS_inj = (self.mass_GeV + self.E_kin_per_A_PS_inj * self.A)/self.mass_GeV
        self.gamma_PS_extr = (self.mass_GeV + self.E_kin_per_A_PS_extr * self.A)/self.mass_GeV
        
        # Estimate number of charges at extraction
        self.Nb_PS_extr = self.linearIntensityLimit(
                                                m = self.mass_GeV, 
                                                gamma = self.gamma_PS_extr,  
                                                Nb_0 = self.Nb0_PS_extr, 
                                                charge_0 = self.Q0_PS, # partially stripped charged state 
                                                m_0 = self.m0_GeV,  
                                                gamma_0 = self.gamma0_PS_extr,  # use gamma at extraction,
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
        # Calculate gamma at injection
        self.gamma_SPS_inj = (self.mass_GeV + self.E_kin_per_A_SPS_inj * self.A)/self.mass_GeV
        self.gamma_SPS_extr = (self.mass_GeV + self.E_kin_per_A_SPS_extr * self.A)/self.mass_GeV
         
        # For comparison on how Roderik scales the gamma - scale directly with charge/mass before stripping 
        # (same magnetic rigidity at PS extraction for all ion species)
        if self.use_Roderiks_gamma:
            # old version not considering LEIR-PS stripping, approximate scaling of gamma 
            #self.gamma_SPS_inj =  self.gamma0_SPS_inj*(self.Q/54)/(self.mass_GeV/self.m0_GeV) 
            
            # In this version below, consider LEIR-PS stripping and thus higher magnetic rigidity 
            # exact gamma expression for given magnetic rigidity, see Roderik's notebook 
            # Brho = P/Q is constant at PS extraction and SPS injection
            # Use P = m*gamma*beta*c
            # gamma = np.sqrt(1 + ((Q/Q0)/(m/m0))**2 + (gamma0**2 - 1))
            self.gamma_SPS_inj =  np.sqrt(
                                    1 + (((self.Z if self.LEIR_PS_strip else self.Q) / 54) / (self.mass_GeV/self.m0_GeV))**2
                                    * (self.gamma0_SPS_inj**2 - 1)
                                   )
            #print("{}: gamma SPS inj: {}".format(self.ion_type, self.gamma_SPS_inj))
        
        # Calculate outgoing intensity from linear scaling 
        self.Nb_SPS_extr = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_SPS_inj,  
                                               Nb_0 = self.Nb0_SPS_extr, 
                                               charge_0 = self.Q0_SPS, 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_SPS_inj,  # use gamma at extraction
                                               fully_stripped=True
                                               )
    
        self.Nq_SPS_extr = self.Nb_SPS_extr*self.Z  # number of outgoing charges
    
    
    def space_charge_tune_shift(self):
        # Calculate the space charge tune shift for each accelerator
        pass
    
    
    def simulate_injection_SpaceCharge_limit(self):
        """
        Simulate space charge limit in full injection through Linac3, LEIR, PS and SPS for a given ion type
        """
        self.linac3()
        self.leir()
        self.stripper_foil_leir_ps()
        self.ps()
        self.stripper_foil_ps_sps()
        self.sps()
        self.space_charge_tune_shift()


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
        for i, ion_type in enumerate(ion_data.columns):
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
        # Calculate ion transmission for LEIR 
        ionsPerPulseLinac3 = (self.linac3_current * self.linac3_pulseLength) / (self.Q * e)
        spaceChargeLimitLEIR = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_LEIR_extr,  
                                               Nb_0 = self.Nb0_LEIR_extr, 
                                               charge_0 = self.Q0_LEIR, # partially stripped charged state 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_LEIR_extr,  # use gamma at extraction
                                               fully_stripped=False
                                               )
        
       #print("{}: SC limit LEIR: {}".format(self.ion_type,  spaceChargeLimitLEIR))
        nPulsesLEIR = (min(7, math.ceil(spaceChargeLimitLEIR / (ionsPerPulseLinac3 * self.LEIR_injection_efficiency))) if self.nPulsesLEIR == 0 else self.nPulsesLEIR)
        
        #print("{}: SC limit LEIR: {:.3e}, ions per pulse: {:.3e}, injection eff: {:.3e}".format(self.ion_type, 
        #                                                                 self.spaceChargeLimitLEIR, 
        #                                                                 self.ionsPerPulseLinac3,
        #                                                                 self.LEIR_injection_efficiency))
        #print("{}: npulses LEIR: {}".format(self.ion_type, nPulsesLEIR))
        
        totalIntLEIR = ionsPerPulseLinac3*nPulsesLEIR * self.LEIR_injection_efficiency
        #print("{}: tot int LEIR: {}".format(self.ion_type, totalIntLEIR))
        
        # Calculate extracted intensity per bunch
        ionsPerBunchExtractedLEIR = self.LEIR_transmission * min(totalIntLEIR, spaceChargeLimitLEIR) / self.LEIR_bunches
        #print("{}: extracted ions per bunch LEIR: {}".format(self.ion_type, ionsPerBunchExtractedLEIR))
        ionsPerBunchExtractedPS = ionsPerBunchExtractedLEIR *(self.LEIR_PS_stripping_efficiency if self.LEIR_PS_strip else 1) * self.PS_transmission / self.PS_splitting
        #print("{}: extracted ions per bunch PS: {}".format(self.ion_type, ionsPerBunchExtractedPS))
        ionsPerBunchSPSinj = ionsPerBunchExtractedPS * (self.PS_SPS_transmission_efficiency if self.Z == self.Q or self.LEIR_PS_strip else self.PS_SPS_stripping_efficiency)
        # OLD LINEL: # ionsPerBunchSPSinj = ionsPerBunchExtractedPS * (self.PS_SPS_stripping_efficiency if self.PS_SPS_strip else self.PS_SPS_transmission_efficiency)
        
        # Calculate ion transmission for SPS 
        spaceChargeLimitSPS = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_SPS_inj,  
                                               Nb_0 = self.Nb0_SPS_extr, 
                                               charge_0 = self.Q0_SPS, 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_SPS_inj,  # use gamma at extraction
                                               fully_stripped=True
                                               )
        #print("{}: SC limit SPS: {}".format(self.ion_type,  spaceChargeLimitSPS))
        ionsPerBunchLHC = min(spaceChargeLimitSPS, ionsPerBunchSPSinj) * self.SPS_transmission * self.SPS_slipstacking_transmission
    
        result = {
            "ionsPerPulseLinac3": ionsPerPulseLinac3,
            "nPulsesLEIR": nPulsesLEIR,
            "gammaLEIR": self.gamma_LEIR_inj,
            "LEIRmaxInt": totalIntLEIR,
            "LEIRspaceChargeLim": spaceChargeLimitLEIR,
            "extractedLEIRperBunch": ionsPerBunchExtractedLEIR,
            "extractedPSperBunch": ionsPerBunchExtractedPS,
            "gammaInjSPS": self.gamma_SPS_inj,
            "SPSmaxIntPerBunch": ionsPerBunchSPSinj,
            "SPSspaceChargeLim": spaceChargeLimitSPS,
            "SPSaccIntensity": min(spaceChargeLimitSPS, ionsPerBunchSPSinj),
            "LHCionsPerBunch": ionsPerBunchLHC,
            "strippingEfficiencyLEIRPS": self.LEIR_PS_stripping_efficiency,
            "LHCChargesPerBunch": ionsPerBunchLHC * self.Z,
            "chargeBeforeStrip": self.Q,
            "atomicNumber": self.Z,
            "massNumber": self.A,
            "LEIRsplitting": self.LEIR_bunches,
            "species": self.ion_type
        }
    
        return result


    def calculate_LHC_bunch_intensity_all_ion_species(self):
        """
        Estimate LHC bunch intensity for all ion species provided in table
        through Linac3, LEIR, PS and SPS considering all the limits of the injectors
        """
        full_result = {
            "Ion": [],
            "massNumber": [],
            "atomicNumber": [],
            "chargeBeforeStrip": [],
            "LEIRmaxInt": [],
            "LEIRspaceChargeLim": [],
            "SPSmaxIntPerBunch": [],
            "SPSspaceChargeLim": [],
            "LHCionsPerBunch": [],
            "LHCChargesPerBunch": [],
            }
        
        # Iterate over all ions in data 
        for i, ion_type in enumerate(ion_data.columns):
            # Initiate the correct ion
            self.init_ion(ion_type)
            result = self.calculate_LHC_bunch_intensity()
            
            # Append the result 
            full_result["Ion"].append(ion_type)
            full_result["massNumber"].append(self.A)
            full_result["atomicNumber"].append(result["atomicNumber"])
            full_result["chargeBeforeStrip"].append(result["chargeBeforeStrip"])
            full_result["LEIRmaxInt"].append(result["LEIRmaxInt"])
            full_result["LEIRspaceChargeLim"].append(result["LEIRspaceChargeLim"])
            full_result["SPSmaxIntPerBunch"].append(result["SPSmaxIntPerBunch"])
            full_result["SPSspaceChargeLim"].append(result["SPSspaceChargeLim"])
            full_result["LHCionsPerBunch"].append(result["LHCionsPerBunch"])
            full_result["LHCChargesPerBunch"].append(result["LHCChargesPerBunch"])
            
            del result
            
        # Convert dictionary to dataframe 
        df_all_ions = pd.DataFrame(full_result)
        df_all_ions = df_all_ions.set_index("Ion")
        
        return df_all_ions
        
            

# Test the class 
if __name__ == '__main__':
    #ion_type = 'Pb'
    ion_data = pd.read_csv("Data/Ion_species.csv", sep=';', header=0, index_col=0).T
    ion_type = 'Pb'
    injector_chain = CERN_Injector_Chain(ion_type, ion_data, account_for_SPS_transmission=False)
    df_Nb = injector_chain.simulate_SpaceCharge_intensity_limit_all_ions()
    
    # Compare to reference intensities
    ref_Table_SPS = pd.read_csv('Data/SPS_final_intensities_WG5_and_Hannes.csv', delimiter=';', index_col=0)
    ref_Table_LEIR = pd.read_csv('Data/LEIR_final_intensities_Nicolo.csv', delimiter=';', index_col=0)
    df_Nb['SPS_WG5_ratio'] = df_Nb['Nb_SPS']/ref_Table_SPS['WG5 Intensity']
    df_Nb['SPS_Hannes_ratio'] = df_Nb['Nb_SPS']/ref_Table_SPS['Hannes Intensity ']
    df_Nb['LEIR_Nicolo_ratio'] = df_Nb['Nb_LEIR']/ref_Table_LEIR['Nicolo Intensity']
    #print(df_Nb.head(10))

    # Calculate the bunch intensity going into the LHC - now Roderik accounts for SPS transmission
    # Roderik uses Reyes excel as input table for linac3: 200 us pulse length, 70 uA
    injector_chain2 = CERN_Injector_Chain(ion_type, 
                                          ion_data, 
                                          nPulsesLEIR = 0,
                                          LEIR_bunches = 2,
                                          PS_splitting = 2,
                                          account_for_SPS_transmission=True)
    result = injector_chain2.calculate_LHC_bunch_intensity()
    
    # Calculate LHC bunch intensity for all ions
    df = injector_chain2.calculate_LHC_bunch_intensity_all_ion_species()


    ## TRY WITHOUT PS SPLITTING
    injector_chain3 = CERN_Injector_Chain(ion_type, 
                                          ion_data, 
                                          nPulsesLEIR = 0,
                                          LEIR_bunches = 2,
                                          PS_splitting = 1,
                                          account_for_SPS_transmission=True)
    df3 = injector_chain3.calculate_LHC_bunch_intensity_all_ion_species()


    ## WITH PS SPLITTING AND LEIR-PS STRIPPING
    injector_chain4 = CERN_Injector_Chain(ion_type, 
                                          ion_data, 
                                          nPulsesLEIR = 0,
                                          LEIR_bunches = 2,
                                          PS_splitting = 2,
                                          account_for_SPS_transmission=True,
                                          LEIR_PS_strip=True
                                          )
    df4 = injector_chain4.calculate_LHC_bunch_intensity_all_ion_species()

    ## WITH NO SPLITTING AND LEIR-PS STRIPPING
    injector_chain5 = CERN_Injector_Chain(ion_type, 
                                          ion_data, 
                                          nPulsesLEIR = 0,
                                          LEIR_bunches = 2,
                                          PS_splitting = 1,
                                          account_for_SPS_transmission=True,
                                          LEIR_PS_strip=True
                                          )
    df5 = injector_chain5.calculate_LHC_bunch_intensity_all_ion_species()

    #### PLOT THE DATA #######
    SMALL_SIZE = 12
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20
    plt.rcParams["font.family"] = "serif"
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    colors = ['green', 'blue', 'purple', 'brown', 'teal', 'coral', 'cyan', 'darkred']
    
    # Colormap - Build the colour maps
    #colours = ["#2196f3", "#bbdefb"] # Colours - Choose the extreme colours of the colour map
    #cmap = mpl.colors.LinearSegmentedColormap.from_list("colour_map", colours, N=256)
    #norm = mpl.colors.Normalize(df['LHCChargesPerBunch'].min(), df['LHCChargesPerBunch'].max()) # linearly normalizes data into the [0.0, 1.0] interval
    bar_width = 0.35
    x = np.arange(len(df.index))

    fig, ax = plt.subplots(1, 1, figsize = (6,5))
    fig.suptitle("")
    bar2 = ax.bar(x - bar_width/2, ref_Table_SPS['WG5 Intensity']*df['atomicNumber'], bar_width, color='red', label='WG5') #
    bar1 = ax.bar(x + bar_width/2, df['LHCChargesPerBunch'], bar_width, color='blue', label='Baseline scenario') #
    #ax.bar_label(bar1, labels=[f'{e:,.1e}' for e in df['LHCChargesPerBunch']], padding=3, color='black', fontsize=9) 
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("LHC charges per bunch")
    ax.legend()
    
    # Baseline scenario
    fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
    bar22 = ax2.bar(x - bar_width/2, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width, color='red', label='WG5') #
    bar12 = ax2.bar(x + bar_width/2, df['LHCionsPerBunch']*df['massNumber'], bar_width, color='blue', label='Baseline scenario') #
    ax2.set_xticks(x)
    ax2.set_xticklabels(df.index)
    ax2.set_ylabel("Nucleons per bunch")
    ax2.legend()
    
    # No PS splitting 
    bar_width2 = 0.25
    fig3, ax3 = plt.subplots(1, 1, figsize = (6,5))
    bar31 = ax3.bar(x - bar_width2, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width2, color='red', label='WG5') #
    bar32 = ax3.bar(x, df['LHCionsPerBunch']*df['massNumber'], bar_width2, color='blue', label='Baseline scenario') #
    bar33 = ax3.bar(x + bar_width2, df3['LHCionsPerBunch']*df3['massNumber'], bar_width2, color='gold', label='No PS splitting') #
    ax3.set_xticks(x)
    ax3.set_xticklabels(df.index)
    ax3.set_ylabel("Nucleons per bunch")
    ax3.legend()
    
    # Interpretation - Ca and Xe higher intensity due to higher LEIR charge state 
    # for In, LEIR is the limitation, i.e. can only inject intensities in the SPS that are below space charge limit
    
    # LEIR-PS stripping
    bar_width4 = 0.2
    fig4, ax4 = plt.subplots(1, 1, figsize = (6,5))
    bar41 = ax4.bar(x - 1.5*bar_width4, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width4, color='red', label='WG5') #
    bar42 = ax4.bar(x - 0.5*bar_width4, df['LHCionsPerBunch']*df['massNumber'], bar_width4, color='blue', label='Baseline scenario') #
    bar43 = ax4.bar(x + 0.5*bar_width4, df3['LHCionsPerBunch']*df3['massNumber'], bar_width4, color='gold', label='No PS splitting') #
    bar44 = ax4.bar(x + 1.5*bar_width4, df4['LHCionsPerBunch']*df4['massNumber'], bar_width4, color='limegreen', label='LEIR-PS stripping') #
    ax4.set_xticks(x)
    ax4.set_xticklabels(df.index)
    ax4.set_ylabel("Nucleons per bunch")
    ax4.legend()
    
    # LEIR-PS stripping and NO PS splitting
    bar_width5 = 0.15
    fig5, ax5 = plt.subplots(1, 1, figsize = (6,5))
    bar51 = ax5.bar(x, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width5, color='red', label='WG5') #
    bar52 = ax5.bar(x + bar_width5, df['LHCionsPerBunch']*df['massNumber'], bar_width5, color='blue', label='Baseline scenario') #
    bar53 = ax5.bar(x + 2*bar_width5, df3['LHCionsPerBunch']*df3['massNumber'], bar_width5, color='gold', label='No PS splitting') #
    bar54 = ax5.bar(x + 3*bar_width5, df4['LHCionsPerBunch']*df4['massNumber'], bar_width5, color='limegreen', label='LEIR-PS stripping') #
    bar55 = ax5.bar(x + 4*bar_width5, df5['LHCionsPerBunch']*df5['massNumber'], bar_width5, color='magenta', label='LEIR-PS striping, \nno PS splitting') #
    ax5.set_xticks(x + 2*bar_width5)
    ax5.set_xticklabels(df.index)
    ax5.set_ylabel("Nucleons per bunch")
    ax5.legend()