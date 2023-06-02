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
    Representation of the CERN Injector Chain for different ions with linear space charge effects 
    """
    def __init__(self, ion_type, ion_data, ion_type_ref='Pb'):
        self.full_ion_data = ion_data
        self.init_ion(ion_type)
        self.debug_mode = False
       
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
        self.strip_ratio = self.ion_data['LEIR-PS Stripping Efficiency']
        self.linac3_current_uA = self.ion_data['Linac3 current [uA]']


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
            self.Nb0_SPS_extr = 2.21e8 # outgoing ions per bunch from SPS (2015 values), with 62% transmission
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
         
        # For comparison on how Roderik scales the gamma
        if self.use_Roderiks_gamma:
            self.gamma_SPS_inj =  self.gamma0_SPS_inj*(self.Q/54)/(self.mass_GeV/self.m0_GeV)
        
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
        
        #self.storedEnergy_at_7TeV_in_MJ =  ### WHAT IS THE GOOD CONVERSION? 
    
    def stripper_foil(self):
        # Calculate the stripping and transmission fraction for each ion type
        pass
    
    
    def space_charge_tune_shift(self):
        # Calculate the space charge tune shift for each accelerator
        pass
    
    
    def simulate_injection(self):
        """
        Simulate full injection with Linac3, LEIR, PS and SPS for a given ion type
        """
        self.linac3()
        self.leir()
        self.stripper_foil_leir_ps()
        self.ps()
        self.stripper_foil_ps_sps()
        self.sps()
        self.space_charge_tune_shift()


    def simulate_injection_all_ions(self, return_dataframe=True):
        """
        Simulate full injection with Linac3, LEIR, PS and SPS for all ions given in table
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
            self.simulate_injection()
            
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
            return self.ion_Nb_data #df_ions


# Test the class 
if __name__ == '__main__':
    ion_type = 'Pb'
    ion_data = pd.read_csv("Data/Ion_species.csv", sep=';', header=0, index_col=0).T

    injector_chain = CERN_Injector_Chain(ion_type, ion_data)
    df_Nb = injector_chain.simulate_injection_all_ions()
    
    # Compare to reference intensities
    ref_Table_SPS = pd.read_csv('Data/SPS_final_intensities_WG5_and_Hannes.csv', delimiter=';', index_col=0)
    ref_Table_LEIR = pd.read_csv('Data/LEIR_final_intensities_Nicolo.csv', delimiter=';', index_col=0)
    df_Nb['SPS_WG5_ratio'] = df_Nb['Nb_SPS']/ref_Table_SPS['WG5 Intensity']
    df_Nb['SPS_Hannes_ratio'] = df_Nb['Nb_SPS']/ref_Table_SPS['Hannes Intensity ']
    df_Nb['LEIR_Nicolo_ratio'] = df_Nb['Nb_LEIR']/ref_Table_LEIR['Nicolo Intensity']
    print(df_Nb.head(10))

"""
    # Iterate over the different ions 
    for ion_type in ion_data.columns:
        # Instantiate the Injector Chain and simulate injection
        injector_chain = CERN_Injector_Chain(ion_type, ion_data)
        injector_chain.simulate_injection()
        #print("\nGamma SPS inj: {:.3f}".format(injector_chain.gamma_SPS_inj))
        #print("\nGamma SPS extr: {:.3f}".format(injector_chain.gamma_SPS_extr))
        #print("SPS extraction for {}: \nNr of charges = {:.2e}, \nNr of ions = {:.2e}\n".format(injector_chain.ion_type, 
        #del injector_chain
"""