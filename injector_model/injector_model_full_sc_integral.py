#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation model of the CERN Injector Chain for different ions
solving for full space charge (SC) lattice integral 
- by Elias Waagaard 
"""
from pathlib import Path
import pandas as pd
import numpy as np
import math
from scipy import constants
import xtrack as xt
import xpart as xp
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .parameters_and_helpers import Inj_Parameters, IBS_Growth_Rates

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
output_folder = Path(__file__).resolve().parent.joinpath('../output').absolute()


 #ibs_folder = Path(__file__).resolve().parent.joinpath('../IBS_for_Xsuite').absolute()
# Import IBS module after "pip install -e IBS_for_Xsuite" has been executed 
# from lib.IBSfunctions import NagaitsevIBS

@dataclass
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
                 account_for_SPS_transmission=True,
                 LEIR_PS_strip=False,
                 save_path_csv = '{}/csv_tables'.format(output_folder)
                 ):
        
        self.full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), header=0, index_col=0).T
        self.LEIR_PS_strip = LEIR_PS_strip
        self.account_for_SPS_transmission = account_for_SPS_transmission
        
        ####################################
        
        # Initiate ion and reference Pb ion type 
        self.init_ion(ion_type)
        self.ion0_referenceValues() 
        
        self.debug_mode = False
                
        # Rules for splitting and bunches 
        self.nPulsesLEIR = nPulsesLEIR
        self.LEIR_bunches = LEIR_bunches
        self.PS_splitting = PS_splitting
    
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
        
        # Set charge state for diffent accelerators
        self.Q_LEIR = self.ion_data['Q before stripping']
        self.Q_PS = self.ion_data['Z'] if self.LEIR_PS_strip else self.ion_data['Q before stripping']
        self.Q_SPS = self.ion_data['Z']
        
        # Values from first tables in Roderik's notebook
        self.linac3_current = self.ion_data['Linac3 current [uA]'] * 1e-6
        self.linac3_pulseLength = self.ion_data['Linac3 pulse length [us]'] * 1e-6
        self.LEIR_PS_stripping_efficiency = self.ion_data['LEIR-PS Stripping Efficiency']


        
    def ion0_referenceValues(self):
        """
        Sets bunch intensity Nb, gamma factor and mass of reference ion species 
        As of now, use reference values from Pb from Hannes 
        For injection and extraction energy, use known Pb ion values from LIU report on 
        https://edms.cern.ch/ui/file/1420286/2/LIU-Ions_beam_parameter_table.pdf
        """

    
        ################## Find current tune shifts from xtrack sequences ##################    
        # LEIR 
        self.particle0_LEIR = xp.Particles(mass0 = 1e9 * self.m0_GeV, q0 = self.Q0_LEIR, gamma0 = self.gamma0_LEIR_inj)
        self.line_LEIR_Pb = xt.Line.from_json('{}/xtrack_sequences/LEIR_2021_Pb_ions_with_RF.json'.format(data_folder))
        self.line_LEIR_Pb.reference_particle = self.particle0_LEIR
        self.line_LEIR_Pb.build_tracker()
        self.twiss0_LEIR = self.line_LEIR_Pb.twiss()
        self.twiss0_LEIR_interpolated, self.sigma_x0_LEIR, self.sigma_y0_LEIR = self.interpolate_Twiss_table(self.twiss0_LEIR, 
                                                                                                             self.line_LEIR_Pb, 
                                                                                                             self.particle0_LEIR, 
                                                                                                             self.ex_LEIR, 
                                                                                                             self.ey_LEIR,
                                                                                                             self.delta_LEIR,
                                                                                                             )
        self.dQx0_LEIR, self.dQy0_LEIR = self.calculate_SC_tuneshift(self.Nb0_LEIR, self.particle0_LEIR, self.sigma_z_LEIR, 
                                                 self.twiss0_LEIR_interpolated, self.sigma_x0_LEIR, self.sigma_y0_LEIR)
        
        # PS 
        self.particle0_PS = xp.Particles(mass0 = 1e9 * self.m0_GeV, q0 = self.Q0_PS, gamma0 = self.gamma0_PS_inj)
        self.line_PS_Pb = xt.Line.from_json('{}/xtrack_sequences/PS_2022_Pb_ions_matched_with_RF.json'.format(data_folder))
        self.line_PS_Pb.reference_particle = self.particle0_PS
        self.line_PS_Pb.build_tracker()
        self.twiss0_PS = self.line_PS_Pb.twiss()
        self.twiss0_PS_interpolated, self.sigma_x0_PS, self.sigma_y0_PS = self.interpolate_Twiss_table(self.twiss0_PS, 
                                                                                                             self.line_PS_Pb, 
                                                                                                             self.particle0_PS, 
                                                                                                             self.ex_PS, 
                                                                                                             self.ey_PS,
                                                                                                             self.delta_PS,
                                                                                                             )
        self.dQx0_PS, self.dQy0_PS = self.calculate_SC_tuneshift(self.Nb0_PS, self.particle0_PS, self.sigma_z_PS, 
                                                 self.twiss0_PS_interpolated, self.sigma_x0_PS, self.sigma_y0_PS)
        
        # SPS 
        self.particle0_SPS = xp.Particles(mass0 = 1e9 * self.m0_GeV, q0 = self.Q0_SPS, gamma0 = self.gamma0_SPS_inj)
        self.line_SPS_Pb = xt.Line.from_json('{}/xtrack_sequences/SPS_2021_Pb_ions_matched_with_RF.json'.format(data_folder))
        self.line_SPS_Pb.reference_particle = self.particle0_SPS
        self.line_SPS_Pb.build_tracker()
        self.twiss0_SPS = self.line_SPS_Pb.twiss()
        self.twiss0_SPS_interpolated, self.sigma_x0_SPS, self.sigma_y0_SPS = self.interpolate_Twiss_table(self.twiss0_SPS, 
                                                                                                             self.line_SPS_Pb, 
                                                                                                             self.particle0_SPS, 
                                                                                                             self.ex_SPS, 
                                                                                                             self.ey_SPS,
                                                                                                             self.delta_SPS,
                                                                                                             )
        self.dQx0_SPS, self.dQy0_SPS = self.calculate_SC_tuneshift(self.Nb0_SPS, self.particle0_SPS, self.sigma_z_SPS, 
                                                 self.twiss0_SPS_interpolated, self.sigma_x0_SPS, self.sigma_y0_SPS)
        
    
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
       
    
    def LEIR(self):
        """
        Calculate gamma at entrance and exit of the LEIR and transmitted bunch intensity 
        using known Pb ion values from LIU report on 
        https://edms.cern.ch/ui/file/1420286/2/LIU-Ions_beam_parameter_table.pdf
        """
        # Estimate gamma at extraction
        self.gamma_LEIR_inj = (self.mass_GeV + self.E_kin_per_A_LEIR_inj * self.A)/self.mass_GeV
        self.gamma_LEIR_extr =  np.sqrt(
                                1 + ((self.Q_LEIR / 54) / (self.mass_GeV/self.m0_GeV))**2
                                * (self.gamma0_LEIR_extr**2 - 1)
                                )
                 
        # Define particle object for LEIR 
        self.particle_LEIR = xp.Particles(mass0 = 1e9 * self.mass_GeV, 
                                     q0 = self.Q_LEIR, 
                                     gamma0 = self.gamma_LEIR_inj
                                     )
        
        # Twiss and beam sizes sigma 
        self.line_LEIR = self.line_LEIR_Pb.copy()
        self.line_LEIR.reference_particle = self.particle_LEIR
        self.line_LEIR.build_tracker()
        self.twiss_LEIR = self.line_LEIR.twiss()
        self.twiss_LEIR_interpolated, self.sigma_x_LEIR, self.sigma_y_LEIR = self.interpolate_Twiss_table(self.twiss_LEIR, 
                                                                                                             self.line_LEIR, 
                                                                                                             self.particle_LEIR, 
                                                                                                             self.ex_LEIR, 
                                                                                                             self.ey_LEIR,
                                                                                                             self.delta_LEIR,
                                                                                                             )
        # Maximum intensity for space charge limit - keep same tune shift as today
        self.Nb_x_max_LEIR, self.Nb_y_max_LEIR = self.maxIntensity_from_SC_integral(self.dQx0_LEIR, self.dQy0_LEIR,
                                                                                    self.particle_LEIR, self.sigma_z_LEIR,
                                                                                    self.twiss_LEIR_interpolated, self.sigma_x_LEIR, 
                                                                                    self.sigma_y_LEIR)

        self.Nb_LEIR_extr = min(self.Nb_x_max_LEIR, self.Nb_y_max_LEIR)  # pick the limiting intensity
        self.Nq_LEIR_extr = self.Nb_LEIR_extr*self.Q_LEIR  # number of outgoing charges, before any stripping
        self.limiting_plane_LEIR = 'X' if self.Nb_x_max_LEIR < self.Nb_y_max_LEIR else 'Y' # flag to identify if x or y plane is limiting

        # Also find IBS growth rates after first turn
        self.Ixx_LEIR, self.Iyy_LEIR, self.Ipp_LEIR = self.find_analytical_IBS_growth_rates(self.particle_LEIR,
                                                                                            self.twiss_LEIR, 
                                                                                            self.line_LEIR,
                                                                                            self.Nb_LEIR_extr, 
                                                                                            self.sigma_z_LEIR,
                                                                                            self.ex_LEIR, 
                                                                                            self.ey_LEIR,
                                                                                            self.delta_LEIR)

    
    def PS(self):
        """
        Calculate gamma at entrance and exit of the PS and transmitted bunch intensity 
        """
        # Estimate gamma at extraction
        self.gamma_PS_inj =  self.gamma_LEIR_extr
        self.gamma_PS_extr =  np.sqrt(
                                1 + ((self.Q_PS / 54) / (self.mass_GeV/self.m0_GeV))**2
                                * (self.gamma0_PS_extr**2 - 1)
                                )
        
        # Define particle object for PS 
        self.particle_PS = xp.Particles(mass0 = 1e9 * self.mass_GeV, 
                                     q0 = self.Q_PS, 
                                     gamma0 = self.gamma_PS_inj
                                     )
        
        # Twiss and beam sizes sigma 
        self.line_PS = self.line_PS_Pb.copy()
        self.line_PS.reference_particle = self.particle_PS
        self.line_PS.build_tracker()
        self.twiss_PS = self.line_PS.twiss()
        self.twiss_PS_interpolated, self.sigma_x_PS, self.sigma_y_PS = self.interpolate_Twiss_table(self.twiss_PS, 
                                                                                                             self.line_PS, 
                                                                                                             self.particle_PS, 
                                                                                                             self.ex_PS, 
                                                                                                             self.ey_PS,
                                                                                                             self.delta_PS,
                                                                                                             )
        # Maximum intensity for space charge limit - keep same tune shift as today
        self.Nb_x_max_PS, self.Nb_y_max_PS = self.maxIntensity_from_SC_integral(self.dQx0_PS, self.dQy0_PS,
                                                                                    self.particle_PS, self.sigma_z_PS,
                                                                                    self.twiss_PS_interpolated, self.sigma_x_PS, 
                                                                                    self.sigma_y_PS)

        self.Nb_PS_extr = min(self.Nb_x_max_PS, self.Nb_y_max_PS)  # pick the limiting intensity
        self.Nq_PS_extr = self.Nb_PS_extr*self.Q_PS  # number of outgoing charges, before any stripping
        self.limiting_plane_PS = 'X' if self.Nb_x_max_PS < self.Nb_y_max_PS else 'Y' # flag to identify if x or y plane is limiting

        # Also find IBS growth rates after first turn
        self.Ixx_PS, self.Iyy_PS, self.Ipp_PS = self.find_analytical_IBS_growth_rates(self.particle_PS,
                                                                                            self.twiss_PS,
                                                                                            self.line_PS,
                                                                                            self.Nb_PS_extr, 
                                                                                            self.sigma_z_PS,
                                                                                            self.ex_PS, 
                                                                                            self.ey_PS,
                                                                                            self.delta_PS)
    
    def SPS(self):
        """
        Calculate gamma at entrance and exit of the SPS, and transmitted bunch intensity 
        Space charge limit comes from gamma at injection
        """
        # Calculate gamma at injection, simply scaling with the kinetic energy per nucleon as of today
         # consider same magnetic rigidity at PS extraction for all ion species: Brho = P/Q, P = m*gamma*beta*c
        self.gamma_SPS_inj =  np.sqrt(
                                1 + ((self.Q_PS / 54) / (self.mass_GeV/self.m0_GeV))**2
                                * (self.gamma0_SPS_inj**2 - 1)
                            )
        self.gamma_SPS_extr = (self.mass_GeV + self.E_kin_per_A_SPS_extr * self.A)/self.mass_GeV
        
        # Define particle object for SPS 
        self.particle_SPS = xp.Particles(mass0 = 1e9 * self.mass_GeV, 
                                     q0 = self.Q_SPS, 
                                     gamma0 = self.gamma_SPS_inj
                                     )
        
        # Twiss and beam sizes sigma 
        self.line_SPS = self.line_SPS_Pb.copy()
        self.line_SPS.reference_particle = self.particle_SPS
        self.line_SPS.build_tracker()
        self.twiss_SPS = self.line_SPS.twiss()
        self.twiss_SPS_interpolated, self.sigma_x_SPS, self.sigma_y_SPS = self.interpolate_Twiss_table(self.twiss_SPS, 
                                                                                                             self.line_SPS, 
                                                                                                             self.particle_SPS, 
                                                                                                             self.ex_SPS, 
                                                                                                             self.ey_SPS,
                                                                                                             self.delta_SPS,
                                                                                                             )
        # Maximum intensity for space charge limit - keep same tune shift as today
        self.Nb_x_max_SPS, self.Nb_y_max_SPS = self.maxIntensity_from_SC_integral(self.dQx0_SPS, self.dQy0_SPS,
                                                                                    self.particle_SPS, self.sigma_z_SPS,
                                                                                    self.twiss_SPS_interpolated, self.sigma_x_SPS, 
                                                                                    self.sigma_y_SPS)

        self.Nb_SPS_extr = min(self.Nb_x_max_SPS, self.Nb_y_max_SPS)  # pick the limiting intensity
        self.Nq_SPS_extr = self.Nb_SPS_extr*self.Q_SPS  # number of outgoing charges, before any stripping
        self.limiting_plane_SPS = 'X' if self.Nb_x_max_SPS < self.Nb_y_max_SPS else 'Y' # flag to identify if x or y plane is limiting

        # Also find IBS growth rates after first turn
        self.Ixx_SPS, self.Iyy_SPS, self.Ipp_SPS = self.find_analytical_IBS_growth_rates(self.particle_SPS,
                                                                                            self.twiss_SPS,
                                                                                            self.line_SPS,
                                                                                            self.Nb_SPS_extr, 
                                                                                            self.sigma_z_SPS,
                                                                                            self.ex_SPS, 
                                                                                            self.ey_SPS,
                                                                                            self.delta_SPS)    
    
    def simulate_injection(self):
        """
        Simulate injection and calculate injection energies for LEIR, PS and SPS 
        """
        self.LEIR()
        self.PS()
        self.SPS()
    

    def calculate_LHC_bunch_intensity(self):
        """
        Estimate LHC bunch intensity for a given ion species
        through Linac3, LEIR, PS and SPS considering all the limits of the injectors
        """
        self.simulate_injection()
        
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