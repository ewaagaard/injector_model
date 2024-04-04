#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation model of the CERN Injector Chain for different ions
solving for full space charge (SC) lattice integral 
- by Elias Waagaard 
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
from scipy import constants
import xtrack as xt
import xpart as xp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict

from .parameters_and_helpers import Reference_Values, BeamParams_LEIR, BeamParams_PS, BeamParams_SPS
from .sequence_makers import Sequences
from .space_charge_and_ibs import SC_Tune_Shifts, IBS_Growth_Rates

#### PLOT SETTINGS #######
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 15,
        "figure.titlesize": 20,
    }
)

# Calculate the absolute path to the data folder relative to the module's location
data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()
energy_folder = Path(__file__).resolve().parent.joinpath('injection_energies/calculated_injection_energies').absolute()
output_folder = Path(__file__).resolve().parent.joinpath('../output').absolute()


class InjectorChain_full_SC:
    """
    Representation of the CERN Injector Chain for different ions with full space charge lattice integral. 
    This model accounts for
    - full space charge integrals in LEIR, PS and SPS 
    """
    def __init__(self, ion_type, 
                 nPulsesLEIR = 1,
                 LEIR_bunches = 2,
                 PS_splitting = 2,
                 LEIR_PS_strip=False,
                 save_path_csv = '{}/csv_tables'.format(output_folder)
                 ):
        
        # Import reference data and initiate ion
        self.full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), header=0, index_col=0).T
        self.LEIR_PS_strip = LEIR_PS_strip
        self.init_ion(ion_type)
                
        # Rules for splitting and bunches 
        self.nPulsesLEIR = nPulsesLEIR
        self.LEIR_bunches = LEIR_bunches
        self.PS_splitting = PS_splitting
        self.save_path = save_path_csv # Save path for data


    def init_ion(self, ion_type):
        """
        Initialize ion species for a given type 
        """
        self.ion_type = ion_type
        self.ion_str = ''.join(filter(str.isalpha, ion_type))
        self.ion_data = self.full_ion_data[ion_type]
        self.mass_GeV = self.ion_data['mass [GeV]']
        self.Z = self.ion_data['Z']
        self.A = self.ion_data['A']
        
        # Set charge state for diffent accelerators
        self.Q_LEIR = self.ion_data['Q before stripping']
        self.Q_PS = self.ion_data['Z'] if self.LEIR_PS_strip else self.ion_data['Q before stripping']
        self.Q_SPS = self.ion_data['Z']
        
        # Values from first tables in Roderik's notebook
        self.linac3_current = self.ion_data['Linac3 current [uA]'] * 1e-6
        self.linac3_pulseLength = self.ion_data['Linac3 pulse length [us]'] * 1e-6
        self.LEIR_PS_stripping_efficiency = self.ion_data['LEIR-PS Stripping Efficiency']
        self.load_ion_energy()

        print(f"Initiated ion type: {self.ion_type}")
        print("Q_LEIR = {}, Q_PS = {}, Q_SPS = {} (fully stripped)\nStrip LEIR-PS: {}".format(self.Q_LEIR, 
                                                                                              self.Q_PS, 
                                                                                              self.Q_SPS,
                                                                                              self.LEIR_PS_strip))


    def load_ion_energy(self):
        """
        Loads calculated ion energies for all stable ion isotopes from the ion_injection_energies module
        """
        # Load ion energy data depending on where stripping is made 
        if self.LEIR_PS_strip:
            self.ion_energy_data = pd.read_csv('{}/ion_injection_energies_LEIR_PS_strip.csv'.format(energy_folder), index_col=0)
        else:
            self.ion_energy_data = pd.read_csv('{}/ion_injection_energies_PS_SPS_strip.csv'.format(energy_folder), index_col=0)
        
        # Load correct entry in ion energy table - key composed of lower charge state Q before stripping, then ion string, then mass number A
        key = str(int(self.Q_LEIR)) + self.ion_str + str(int(self.A))
        ion_energy = self.ion_energy_data.loc[key]
        
        # Load reference injection energies
        self.LEIR_gamma_inj = ion_energy['LEIR_gamma_inj']
        self.LEIR_gamma_extr = ion_energy['LEIR_gamma_extr']
        self.PS_gamma_inj = ion_energy['PS_gamma_inj']
        self.PS_gamma_extr = ion_energy['PS_gamma_extr']
        self.SPS_gamma_inj = ion_energy['SPS_gamma_inj']
        self.SPS_gamma_extr = ion_energy['SPS_gamma_extr']
        print('Ion type {}: \nPS extr gamma: {:.3f}'.format(self.ion_type, self.PS_gamma_extr))


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
       
    def LEIR_SC_limit(self):
        """
        LEIR: Load Pb reference values and line, calculate present Pb space charge tune shift
        Solve for maximum bunch intensity Nb_max with new particles
        """

        # Calculate max tune shift for Pb ions today - first call Pb reference values
        ref_val = Reference_Values()
        line_LEIR_Pb0 = Sequences.get_LEIR_line(m0_GeV = ref_val.m0_GeV, 
                                            Q = ref_val.Q0_LEIR, 
                                            gamma = ref_val.gamma0_LEIR_inj)
        dQx0, dQy0 = SC_Tune_Shifts.calculate_SC_tuneshift(line=line_LEIR_Pb0, 
                                                           beamParams=BeamParams_LEIR)

        # Call LEIR sequence with new ion
        line_LEIR = Sequences.get_LEIR_line(m0_GeV = self.mass_GeV, 
                                            Q = self.Q_LEIR, 
                                            gamma = self.LEIR_gamma_inj)
 
        # Calculate max intensity from space charge tune shift, assuming we are at the maximum limit today
        Nb_spaceChargeLimitLEIR = SC_Tune_Shifts.maxIntensity_from_SC_integral(dQx_max=dQx0, dQy_max=dQy0,
                                                                   line=line_LEIR, beamParams=BeamParams_LEIR)
        
        return Nb_spaceChargeLimitLEIR

    
    def PS_SC_limit(self):
        """
        PS: Load Pb reference values and line, calculate present Pb space charge tune shift
        Solve for maximum bunch intensity Nb_max with new particles
        """

        # Calculate max tune shift for Pb ions today - first call Pb reference values
        ref_val = Reference_Values()
        line_PS_Pb0 = Sequences.get_PS_line(m0_GeV = ref_val.m0_GeV, 
                                            Q = ref_val.Q0_PS, 
                                            gamma = ref_val.gamma0_PS_inj)
        dQx0, dQy0 = SC_Tune_Shifts.calculate_SC_tuneshift(line=line_PS_Pb0, 
                                                           beamParams=BeamParams_PS)

        # Call PS sequence with new ion
        line_PS = Sequences.get_PS_line(m0_GeV = self.mass_GeV, 
                                            Q = self.Q_PS, 
                                            gamma = self.PS_gamma_inj)
 
        # Calculate max intensity from space charge tune shift, assuming we are at the maximum limit today
        Nb_spaceChargeLimitPS = SC_Tune_Shifts.maxIntensity_from_SC_integral(dQx_max=dQx0, dQy_max=dQy0,
                                                                   line=line_PS, beamParams=BeamParams_PS)
        
        return Nb_spaceChargeLimitPS


    def SPS_SC_limit(self):
        """
        SPS: Load Pb reference values and line, calculate present Pb space charge tune shift
        Solve for maximum bunch intensity Nb_max with new particles
        """

        # Calculate max tune shift for Pb ions today - first call Pb reference values
        ref_val = Reference_Values()
        line_SPS_Pb0 = Sequences.get_SPS_line(m0_GeV = ref_val.m0_GeV, 
                                            Q = ref_val.Q0_SPS, 
                                            gamma = ref_val.gamma0_SPS_inj)
        dQx0, dQy0 = SC_Tune_Shifts.calculate_SC_tuneshift(line=line_SPS_Pb0, 
                                                           beamParams=BeamParams_SPS)

        # Call SPS sequence with new ion
        line_SPS = Sequences.get_SPS_line(m0_GeV = self.mass_GeV, 
                                            Q = self.Q_SPS, 
                                            gamma = self.SPS_gamma_inj)
 
        # Calculate max intensity from space charge tune shift, assuming we are at the maximum limit today
        Nb_spaceChargeLimitSPS = SC_Tune_Shifts.maxIntensity_from_SC_integral(dQx_max=dQx0, dQy_max=dQy0,
                                                                   line=line_SPS, beamParams=BeamParams_SPS)
        
        return Nb_spaceChargeLimitSPS 
    

    def calculate_LHC_bunch_intensity(self):
        """
        Estimate LHC bunch intensity for a given ion species
        through Linac3, LEIR, PS and SPS considering full lattice integral space charge limits of the injectors
        """        
        ### LINAC3 ### 
        ionsPerPulseLinac3 = (self.linac3_current * self.linac3_pulseLength) / (self.Q_LEIR * constants.e)
        
        ### LEIR ###
        spaceChargeLimitLEIR = self.LEIR_SC_limit()
        
        nPulsesLEIR = (min(7, math.ceil(spaceChargeLimitLEIR / (ionsPerPulseLinac3 * Reference_Values.LEIR_injection_efficiency))) if self.nPulsesLEIR == 0 else self.nPulsesLEIR)
        totalIntLEIR = ionsPerPulseLinac3 * nPulsesLEIR * Reference_Values.LEIR_injection_efficiency
        ionsPerBunchExtractedLEIR = Reference_Values.LEIR_transmission * np.min([totalIntLEIR, spaceChargeLimitLEIR]) / self.LEIR_bunches
        LEIR_space_charge_limit_hit = True if totalIntLEIR > spaceChargeLimitLEIR else False 
        
        #### PS ####
        ionsPerBunchInjectedPS = ionsPerBunchExtractedLEIR * (self.LEIR_PS_stripping_efficiency if self.LEIR_PS_strip else 1)
        spaceChargeLimitPS = self.PS_SC_limit()
        
        # Check that injected momentum is not too low for the PS B-field
        self.p_PS_inj = self.calcMomentum_from_gamma(self.PS_gamma_inj, self.Q_PS)
        self.Brho_PS_inj = self.calcBrho(self.p_PS_inj, self.Q_PS) # same as LEIR extraction if no stripping, else will be different 
        B_PS_inj = self.Brho_PS_inj / self.PS_rho
        if B_PS_inj < self.PS_MinB:
            self.PS_B_field_is_too_low = True
        elif B_PS_inj > self.PS_MaxB:
            print("\nA = {}, Q_PS = {}, m_ion = {:.2f} GeV, Z = {}".format(self.A, self.Q_PS, self.mass_GeV, self.Z))
            print('B = {:.4f} in PS at injection is too HIGH!'.format(B_PS_inj))
            raise ValueError("B field in PS is too high!")
        else:
            self.PS_B_field_is_too_low = False
        
        # Select minimum between maxiumum possible injected intensity and PS space charge limit
        ionsPerBunchPS = min(spaceChargeLimitPS, ionsPerBunchInjectedPS)
        PS_space_charge_limit_hit = True if ionsPerBunchInjectedPS > spaceChargeLimitPS else False 
        ionsPerBunchExtracted_PS = ionsPerBunchPS * Reference_Values.PS_transmission / self.PS_splitting # maximum intensity without SC
        
        # Calculate ion transmission for SPS 
        ionsPerBunchSPSinj = ionsPerBunchExtracted_PS * (Reference_Values.PS_SPS_transmission_efficiency if self.Z == self.Q_SPS or self.LEIR_PS_strip else Reference_Values.PS_SPS_stripping_efficiency)
        spaceChargeLimitSPS = self.SPS_SC_limit()
        SPS_space_charge_limit_hit = True if ionsPerBunchSPSinj > spaceChargeLimitSPS else False
        ionsPerBunchLHC = min(spaceChargeLimitSPS, ionsPerBunchSPSinj) * Reference_Values.SPS_transmission * Reference_Values.SPS_slipstacking_transmission

        result = {
            "Ion": self.ion_type,
            "chargeBeforeStrip": int(self.Q),
            "atomicNumber": int(self.Z),
            "massNumber": int(self.A),
            "Linac3_current [A]": self.linac3_current,
            "Linac3_pulse_length [s]": self.linac3_pulseLength, 
            "LEIR_numberofPulses": nPulsesLEIR,
            "LEIR_injection_efficiency": Reference_Values.LEIR_injection_efficiency, 
            "LEIR_splitting": self.LEIR_bunches,
            "LEIR_transmission": Reference_Values.LEIR_transmission, 
            "PS_splitting": self.PS_splitting, 
            "PS_transmission": Reference_Values.PS_transmission, 
            "PS_SPS_stripping_efficiency": Reference_Values.PS_SPS_stripping_efficiency, 
            "SPS_transmission": Reference_Values.SPS_transmission, 
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
        # Check that output directory exists
        os.makedirs(self.save_path, exist_ok=True)
        
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