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
from collections import defaultdict

from .parameters_and_helpers import Reference_Values, BeamParams_LEIR, BeamParams_PS, BeamParams_SPS
from .sequence_makers import Sequences
from .space_charge_and_ibs import SC_Tune_Shifts, IBS_Growth_Rates
from .injection_energies import InjectionEnergies

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


class InjectorChain:
    """
    Representation of the CERN Injector Chain for different ions with full space charge lattice integral. 
    This model accounts for
    - full space charge integrals in LEIR, PS and SPS 

    Parameters:
    -----------
    nPulsesLEIR : int
        number of pulses injected into LEIR (maximum 7 today with Pb). If "None", then calculate max from SC limit
    LEIR_bunches : int
        after RF capture, how many bunches that will circulate in LEIR
    PS_splitting : int
        number of bunches that injected bunch in PS will be split into. E.g. "2" means that bunch will be split into 2 bunches
    account_for_LEIR_ecooling : bool
        whether to factor electron cooling time in LEIR, which is longer for lighter ions
    LEIR_PS_strip : bool
        whether stripping foil should be placed between LEIR and PS. If "False", then default stripping between PS and SPS
    PS_factor_SC : float
        acceptance factor, number of times by which we accept the SC limit in PS to be exceeded compared to Pb. Relies on
        assumption that we are not limited by space charge in PS today. For example, a PS_factor_SC = 1.5 will allow SC limit to
        cause tune shifts 1.5 higher than today. np.inf removes space charge limit entirely
    """
    def __init__(self, ion_type='Pb', 
                 nPulsesLEIR = None,
                 LEIR_bunches = 2,
                 PS_splitting = 2,
                 account_for_LEIR_ecooling=False,
                 LEIR_PS_strip=False,
                 PS_factor_SC = 1.0
                 ):
        
        # Import reference data and initiate ion
        self.full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), header=0, index_col=0).T
        self.LEIR_PS_strip = LEIR_PS_strip
        self.account_for_LEIR_ecooling = account_for_LEIR_ecooling
        self.init_ion(ion_type)
        self.load_Pb_lines()
                
        # Rules for splitting and bunches 
        if nPulsesLEIR is None:
            self.nPulsesLEIR_default = Reference_Values.max_injections_into_LEIR
        else:
            self.nPulsesLEIR_default = nPulsesLEIR
        self.LEIR_bunches = LEIR_bunches
        self.PS_splitting = PS_splitting
        self.PS_factor_SC = PS_factor_SC


    def load_Pb_lines(self) -> None:
        """
        Load xtrack lines for LEIR, PS and SPS - retrieve Twiss tables and lengths
        Calculate present space charge tune shift dQx and dQy for Pb, as a reference
        """
        # Instantiate reference values and class for tune shifts
        ref_val = Reference_Values()
        sc = SC_Tune_Shifts()

        ##### LEIR #####
        self.line_LEIR_Pb0 = Sequences.get_LEIR_line(m0_GeV = ref_val.m0_GeV, 
                                            Q = ref_val.Q0_LEIR, 
                                            gamma = ref_val.gamma0_LEIR_inj)
        self.twiss_LEIR_Pb0 = self.line_LEIR_Pb0.twiss()
        self.line_LEIR_length = self.line_LEIR_Pb0.get_length()
        self.dQx0_LEIR_Pb, self.dQy0_LEIR_Pb = sc.calculate_SC_tuneshift(twissTableXsuite=self.twiss_LEIR_Pb0,
                                                    particle_ref=self.line_LEIR_Pb0.particle_ref,
                                                    line_length=self.line_LEIR_length,
                                                    Nb=BeamParams_LEIR.Nb,
                                                    beamParams=BeamParams_LEIR)

        ##### PS #####
        self.line_PS_Pb0 = Sequences.get_PS_line(m0_GeV = ref_val.m0_GeV, 
                                            Q = ref_val.Q0_PS, 
                                            gamma = ref_val.gamma0_PS_inj)
        self.twiss_PS_Pb0 = self.line_PS_Pb0.twiss()
        self.line_PS_length = self.line_PS_Pb0.get_length()
        self.dQx0_PS_Pb, self.dQy0_PS_Pb = sc.calculate_SC_tuneshift(twissTableXsuite=self.twiss_PS_Pb0,
                                                    particle_ref=self.line_PS_Pb0.particle_ref,
                                                    line_length=self.line_PS_length,
                                                    Nb=BeamParams_PS.Nb,
                                                    beamParams=BeamParams_PS)

        ##### SPS #####
        self.line_SPS_Pb0 = Sequences.get_SPS_line(m0_GeV = ref_val.m0_GeV, 
                                            Q = ref_val.Q0_SPS, 
                                            gamma = ref_val.gamma0_SPS_inj)
        self.twiss_SPS_Pb0 = self.line_SPS_Pb0.twiss()
        self.line_SPS_length = self.line_SPS_Pb0.get_length()
        self.dQx0_SPS_Pb, self.dQy0_SPS_Pb = sc.calculate_SC_tuneshift(twissTableXsuite=self.twiss_SPS_Pb0,
                                                    particle_ref=self.line_SPS_Pb0.particle_ref,
                                                    line_length=self.line_SPS_length,
                                                    Nb=BeamParams_SPS.Nb,
                                                    beamParams=BeamParams_SPS)
        print('Pb tune shifts:')
        print('LEIR: dQx_Pb = {:.3f}, dQy_Pb = {:.3f}'.format(self.dQx0_LEIR_Pb, self.dQy0_LEIR_Pb))
        print('PS:   dQx_Pb = {:.3f}, dQy_Pb = {:.3f}'.format(self.dQx0_PS_Pb, self.dQy0_PS_Pb))
        print('SPS:  dQx_Pb = {:.3f}, dQy_Pb = {:.3f}'.format(self.dQx0_SPS_Pb, self.dQy0_SPS_Pb))
        print('Loaded all sequences.\n')


    def init_ion(self, ion_type, ion_data_custom=None) -> None:
        """
        Initialize ion species for a given type - can be a customized pd.Dataframe with ion_type already chosen
        """
        self.ion_type = ion_type
        self.ion_str = ''.join(filter(str.isalpha, ion_type))

        # Provide custom ion data if desired
        if ion_data_custom is None:
            self.ion_data = self.full_ion_data[ion_type].copy()
        else:
            self.ion_data = ion_data_custom

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
        
        # LEIR ecooling ratios - if new charge states, scale with reference data
        if self.account_for_LEIR_ecooling:
            
            # Check if we are testing a different charge state
            Q_default = self.full_ion_data[ion_type]['Q before stripping']
            A_default = self.full_ion_data[ion_type]['A']
            Q = self.ion_data['Q before stripping']
            A = self.ion_data['A']
            
            # If new charge state, determine which factor by which to change the ecooling time
            if Q == Q_default:
                ecooling_factor = 1.0
            else:
                ecooling_factor = (Q_default/Q)**2 * (A/A_default)
                print('New charge state! --> recalculate e-cooling time factor to {:.3f}'.format(ecooling_factor))
                
            self.relative_ecooling_time_leir = self.ion_data['Relative_LEIR_ecooling_time'] * ecooling_factor
            print('E-cooling time w.r.t to Pb: {:.3f}\n'.format(self.relative_ecooling_time_leir * ecooling_factor))


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

        # See if ion energy has been calculated, otherwise calculate it for this svenario
        try:
            ion_energy = self.ion_energy_data.loc[key]
        except KeyError:
            print('Calculating new injection energies...')
            A, Q_low, Z = self.A, self.ion_data['Q before stripping'], self.Z
            m_in_eV = self.ion_data['mass [GeV]']  * 1e9
            m_ion_in_u = m_in_eV / constants.physical_constants['atomic mass unit-electron volt relationship'][0] 
            inj_energies0 = InjectionEnergies(A, Q_low, m_ion_in_u, Z, LEIR_PS_strip=self.LEIR_PS_strip)
            ion_energy = inj_energies0.return_all_gammas()
        
        # Load reference injection energies
        self.LEIR_gamma_inj = ion_energy['LEIR_gamma_inj']
        self.LEIR_gamma_extr = ion_energy['LEIR_gamma_extr']
        self.PS_gamma_inj = ion_energy['PS_gamma_inj']
        self.PS_gamma_extr = ion_energy['PS_gamma_extr']
        self.SPS_gamma_inj = ion_energy['SPS_gamma_inj']
        self.SPS_gamma_extr = ion_energy['SPS_gamma_extr']
        print('\nIon type {} with Q_PS = {}: \nPS extr gamma: {:.3f}'.format(self.ion_type, self.Q_PS, self.PS_gamma_extr))


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

        Parameters:
        -----------
        None

        Returns:
        --------
        Nb_spaceChargeLimitLEIR : float
            maximum bunch intensity from space charge limit
        dQx, dQy : float
            space charge tune shift related to this space charge limit
        """

        # Generate new ion
        LEIR_particle = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_LEIR, gamma0 = self.LEIR_gamma_inj)

        # Calculate max intensity from space charge tune shift, assuming we are at the maximum limit today
        leir_sc = SC_Tune_Shifts()
        Nb_spaceChargeLimitLEIR = leir_sc.maxIntensity_from_SC_integral(dQx_max=self.dQx0_LEIR_Pb, dQy_max=self.dQy0_LEIR_Pb,
                                                                        twissTableXsuite=self.twiss_LEIR_Pb0,
                                                                        particle_ref=LEIR_particle,
                                                                        line_length=self.line_LEIR_length, 
                                                                        beamParams=BeamParams_LEIR)
        
        # Find tune shifts of space charge limits
        dQx, dQy = leir_sc.calculate_SC_tuneshift(twissTableXsuite=self.twiss_LEIR_Pb0,
                                            particle_ref=LEIR_particle,
                                            line_length=self.line_LEIR_length, 
                                            Nb=Nb_spaceChargeLimitLEIR,
                                            beamParams=BeamParams_LEIR)
        
        return Nb_spaceChargeLimitLEIR, dQx, dQy


    def LEIR_tune_shifts(self, Nb_max=None):
        """
        LEIR: Load Pb reference values and line, calculate space charge tune shift from
        given bunch intensity

        Parameters:
        -----------
        Nb_max : float
            maximum injected bunch intensity from previous accelerators, to calculate tune shift        

        Returns:
        --------
        dQx, dQy : float
            space charge tune shift assuming maximum possible intensity from previous injectors
        """
        # If Nb_max is None, set to default value
        if Nb_max is None:
            Nb_max = BeamParams_LEIR.Nb

        # Generate new ion
        LEIR_particle = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_LEIR, gamma0 = self.LEIR_gamma_inj)

        # Find tune shifts of space charge limits
        leir_sc = SC_Tune_Shifts()
        dQx, dQy = leir_sc.calculate_SC_tuneshift(twissTableXsuite=self.twiss_LEIR_Pb0,
                                            particle_ref=LEIR_particle,
                                            line_length=self.line_LEIR_length, 
                                            Nb=Nb_max,
                                            beamParams=BeamParams_LEIR)
        
        return dQx, dQy
        
        
    def PS_SC_limit(self):
        """
        PS: Load Pb reference values and line, calculate present Pb space charge tune shift
        Solve for maximum bunch intensity Nb_max with new particles

        Parameters:
        -----------
        None

        Returns:
        --------
        Nb_spaceChargeLimitLEIR : float
            maximum possible injected bunch intensity from previous accelerators, to calculate new ion tune shift
        dQx, dQy : float
            space charge tune shift related to this space charge limit
        """
        # Generate new ion
        PS_particle = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_PS, gamma0 = self.PS_gamma_inj)   
        
        # Calculate max intensity from space charge tune shift, assuming we are at the maximum limit today
        ps_sc = SC_Tune_Shifts()
        Nb_spaceChargeLimitPS = ps_sc.maxIntensity_from_SC_integral(dQx_max=self.dQx0_PS_Pb, dQy_max=self.dQy0_PS_Pb,
                                                                    twissTableXsuite=self.twiss_PS_Pb0,
                                                                    particle_ref=PS_particle,
                                                                    line_length=self.line_PS_length, 
                                                                    beamParams=BeamParams_PS)
        
        # Find tune shifts of space charge limit
        dQx, dQy = ps_sc.calculate_SC_tuneshift(twissTableXsuite=self.twiss_PS_Pb0,
                                            particle_ref=PS_particle,
                                            line_length=self.line_PS_length,
                                            Nb=Nb_spaceChargeLimitPS,
                                            beamParams=BeamParams_PS)
        
        return Nb_spaceChargeLimitPS, dQx, dQy


    def PS_tune_shifts(self, Nb_max=None):
        """
        PS: Load Pb reference values and line, calculate space charge tune shift from
        given bunch intensity

        Parameters:
        -----------
        Nb_max : float
            maximum injected bunch intensity from previous accelerators, to calculate tune shift        

        Returns:
        --------
        dQx, dQy : float
            space charge tune shift assuming maximum possible intensity from previous injectors
        """
        # If Nb_max is None, set to default value
        if Nb_max is None:
            Nb_max = BeamParams_PS.Nb

        # Generate new ion
        PS_particle = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_PS, gamma0 = self.PS_gamma_inj)   
        
        # Calculate max intensity from space charge tune shift, assuming we are at the maximum limit today
        ps_sc = SC_Tune_Shifts()
        dQx, dQy = ps_sc.calculate_SC_tuneshift(twissTableXsuite=self.twiss_PS_Pb0,
                                            particle_ref=PS_particle,
                                            line_length=self.line_PS_length,
                                            Nb=Nb_max,
                                            beamParams=BeamParams_PS)
        
        return dQx, dQy


    def SPS_SC_limit(self):
        """
        SPS: Load Pb reference values and line, calculate present Pb space charge tune shift
        Solve for maximum bunch intensity Nb_max with new particles

        Parameters:
        -----------
        None

        Returns:
        --------
        Nb_spaceChargeLimitLEIR : float
            maximum bunch intensity from space charge limit
        dQx, dQy : float
            space charge tune shift related to this space charge limit
        """
        # Generate new ion
        SPS_particle = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_SPS, gamma0 = self.SPS_gamma_inj)

        # Calculate max intensity from space charge tune shift, assuming we are at the maximum limit today
        sps_sc = SC_Tune_Shifts()
        Nb_spaceChargeLimitSPS = sps_sc.maxIntensity_from_SC_integral(dQx_max=self.dQx0_SPS_Pb, dQy_max=self.dQy0_SPS_Pb,
                                                                      twissTableXsuite=self.twiss_SPS_Pb0,
                                                                      particle_ref=SPS_particle,
                                                                      line_length=self.line_SPS_length,
                                                                      beamParams=BeamParams_SPS)
        
        dQx, dQy = sps_sc.calculate_SC_tuneshift(twissTableXsuite=self.twiss_SPS_Pb0,
                                                   particle_ref=SPS_particle,
                                                   line_length=self.line_SPS_length,
                                                   Nb=Nb_spaceChargeLimitSPS,
                                                   beamParams=BeamParams_SPS)
        
        return Nb_spaceChargeLimitSPS, dQx, dQy
    

    def SPS_tune_shifts(self, Nb_max=None):
        """
        SPS: Load Pb reference values and line, calculate space charge tune shift from
        given bunch intensity

        Parameters:
        -----------
        Nb_max : float
            maximum injected bunch intensity from previous accelerators, to calculate tune shift        

        Returns:
        --------
        dQx, dQy : float
            space charge tune shift assuming maximum possible intensity from previous injectors
        """
        # If Nb_max is None, set to default value
        if Nb_max is None:
            Nb_max = BeamParams_SPS.Nb

        # Generate new ion
        SPS_particle = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_SPS, gamma0 = self.SPS_gamma_inj)

        # Calculate max intensity from space charge tune shift, assuming we are at the maximum limit today
        sps_sc = SC_Tune_Shifts()
        dQx, dQy = sps_sc.calculate_SC_tuneshift(twissTableXsuite=self.twiss_SPS_Pb0,
                                                   particle_ref=SPS_particle,
                                                   line_length=self.line_SPS_length,
                                                   Nb=Nb_max,
                                                   beamParams=BeamParams_SPS)
        return dQx, dQy


    def calculate_LHC_bunch_intensity(self):
        """
        Estimate LHC bunch intensity for a given ion species
        through Linac3, LEIR, PS and SPS considering full lattice integral space charge limits of the injectors
        """        
        ### LINAC3 ### 
        ionsPerPulseLinac3 = (self.linac3_current * self.linac3_pulseLength) / (self.Q_LEIR * constants.e)
        
        ### LEIR ###
        spaceChargeLimitLEIR, dQx0_LEIR, dQy0_LEIR = self.LEIR_SC_limit()  # space charge limit assuming same tune shifts as today
                             
        # Calculate number of bunches to inject if we consider electron cooling
        if self.account_for_LEIR_ecooling:
            num_injections_LEIR_with_ecooling = math.floor(Reference_Values.max_injections_into_LEIR / self.relative_ecooling_time_leir)
            if num_injections_LEIR_with_ecooling > 1:
                self.nPulsesLEIR = np.min([num_injections_LEIR_with_ecooling, self.nPulsesLEIR_default])
            else:
                self.nPulsesLEIR = 1
            print('Number of LEIR inejctions considering e-cooling: {}'.format(self.nPulsesLEIR))
        else:
            self.nPulsesLEIR = self.nPulsesLEIR_default
        
        totalIntLEIR = ionsPerPulseLinac3 * self.nPulsesLEIR * Reference_Values.LEIR_injection_efficiency
        ionsPerBunchExtractedLEIR = Reference_Values.LEIR_transmission * np.min([totalIntLEIR, spaceChargeLimitLEIR]) / self.LEIR_bunches
        LEIR_space_charge_limit_hit = True if totalIntLEIR > spaceChargeLimitLEIR else False 

        dQx_LEIR, dQy_LEIR = self.LEIR_tune_shifts(Nb_max=totalIntLEIR) # calculate tune shifts for LEIR if max new intensity is injected
        
        #### PS ####
        ionsPerBunchInjectedPS = ionsPerBunchExtractedLEIR * (self.LEIR_PS_stripping_efficiency if self.LEIR_PS_strip else 1)
        dQx_PS, dQy_PS = self.PS_tune_shifts(Nb_max=ionsPerBunchInjectedPS ) # calculate tune shifts for PS if max new intensity is injected
        spaceChargeLimitPS, dQx0_PS, dQy0_PS = self.PS_SC_limit()
        
        # Check that injected momentum is not too low for the PS B-field
        self.p_PS_inj = self.calcMomentum_from_gamma(self.PS_gamma_inj, self.Q_PS)
        self.Brho_PS_inj = self.calcBrho(self.p_PS_inj, self.Q_PS) # same as LEIR extraction if no stripping, else will be different 
        B_PS_inj = self.Brho_PS_inj / Reference_Values.PS_rho
        if B_PS_inj < Reference_Values.PS_MinB:
            self.PS_B_field_is_too_low = True
        elif B_PS_inj > Reference_Values.PS_MaxB:
            print("\nA = {}, Q_PS = {}, m_ion = {:.2f} GeV, Z = {}".format(self.A, self.Q_PS, self.mass_GeV, self.Z))
            print('B = {:.4f} in PS at injection is too HIGH!'.format(B_PS_inj))
            raise ValueError("B field in PS is too high!")
        else:
            self.PS_B_field_is_too_low = False
        
        # Select minimum between maxiumum possible injected intensity and PS space charge limit
        ionsPerBunchPS = min(self.PS_factor_SC * spaceChargeLimitPS, ionsPerBunchInjectedPS)
        PS_space_charge_limit_hit = True if ionsPerBunchInjectedPS > spaceChargeLimitPS else False # check if we are above SC tune shifts for Pb in PS today, without factor 
        ionsPerBunchExtracted_PS = ionsPerBunchPS * Reference_Values.PS_transmission / self.PS_splitting # maximum intensity without SC
        
        # Calculate ion transmission for SPS 
        ionsPerBunchSPSinj = ionsPerBunchExtracted_PS * (Reference_Values.PS_SPS_transmission_efficiency if self.LEIR_PS_strip else Reference_Values.PS_SPS_stripping_efficiency)
        dQx_SPS, dQy_SPS = self.SPS_tune_shifts(Nb_max=ionsPerBunchSPSinj) # calculate tune shifts for SPS if max new intensity is injected
        spaceChargeLimitSPS, dQx0_SPS, dQy0_SPS = self.SPS_SC_limit()
        SPS_space_charge_limit_hit = True if ionsPerBunchSPSinj > spaceChargeLimitSPS else False
        ionsPerBunchLHC = min(spaceChargeLimitSPS, ionsPerBunchSPSinj) * Reference_Values.SPS_transmission * Reference_Values.SPS_slipstacking_transmission

        result = {
            "Ion": self.ion_type,
            "atomicNumber": int(self.Z),
            "massNumber": int(self.A),
            "Q_LEIR": int(self.Q_LEIR),
            "Q_PS": int(self.Q_PS),
            "Q_SPS": int(self.Q_SPS),
            "Linac3_current [A]": self.linac3_current,
            "Linac3_pulse_length [s]": self.linac3_pulseLength, 
            "LEIR_numberofPulses": self.nPulsesLEIR,
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
            "SPS_space_charge_limit": spaceChargeLimitSPS,
            "LHC_ionsPerBunch": ionsPerBunchLHC,
            "LHC_chargesPerBunch": ionsPerBunchLHC * self.Z,
            "LEIR_gamma_inj": self.LEIR_gamma_inj,
            "LEIR_gamma_extr": self.LEIR_gamma_extr,
            "PS_gamma_inj": self.PS_gamma_inj,
            "PS_gamma_extr": self.PS_gamma_extr,
            "SPS_gamma_inj": self.SPS_gamma_inj,
            "SPS_gamma_extr": self.SPS_gamma_extr,
            "PS_B_field_is_too_low": self.PS_B_field_is_too_low,
            "LEIR_space_charge_limit_hit": LEIR_space_charge_limit_hit,
            "PS_space_charge_limit_hit": PS_space_charge_limit_hit,
            "SPS_space_charge_limit_hit": SPS_space_charge_limit_hit,
            "LEIR_ratio_SC_limit_maxIntensity": spaceChargeLimitLEIR / totalIntLEIR,
            "PS_ratio_SC_limit_maxIntensity": spaceChargeLimitPS / ionsPerBunchInjectedPS,
            "SPS_ratio_SC_limit_maxIntensity": spaceChargeLimitSPS / ionsPerBunchSPSinj,
            "dQx LEIR SC limit tune shift": dQx0_LEIR,
            "dQy LEIR SC limit tune shift": dQy0_LEIR,
            "dQx PS SC limit tune shift": dQx0_PS,
            "dQy PS SC limit tune shift": dQy0_PS,
            "dQx SPS SC limit tune shift": dQx0_SPS,
            "dQy SPS SC limit tune shift": dQy0_SPS,
            "dQx LEIR tune shift with Nb_max": dQx_LEIR,
            "dQy LEIR tune shift with Nb_max": dQy_LEIR,
            "dQx PS tune shift with Nb_max": dQx_PS,
            "dQy PS tune shift with Nb_max": dQy_PS,
            "dQx SPS tune shift with Nb_max": dQx_SPS,
            "dQy SPS tune shift with Nb_max": dQy_SPS,
            "PS space charge acceptance factor": self.PS_factor_SC
        }

        # Add key of LEIR-PS stripping efficiency if this is done 
        if self.LEIR_PS_strip:
            result["LEIR_PS_strippingEfficiency"] = self.LEIR_PS_stripping_efficiency
            
        if self.account_for_LEIR_ecooling:
            result["LEIR_relative_ecooling_time"] = self.relative_ecooling_time_leir
            result["LEIR_numberofPulses_ecooling"] = self.nPulsesLEIR
    
        return result

        
    def calculate_IBS_growth_rates(self, Nb_LEIR, Nb_PS, Nb_SPS):
        """
        Calculate IBS growth rates for initiated ion species and beam parameters

        Parameters:
        -----------
        Nb : float
            bunch intensity

        Returns:
        --------
        self.ion_str : str
            ion type in string format
        growth_rates_dict : dict
            IBS Nagaitsev growth rates dictionary
        """

        # Instantiate IBS analytical model
        IBS = IBS_Growth_Rates()
        
        # LEIR growth rates
        leir_line = self.line_LEIR_Pb0.copy()
        leir_line.particle_ref = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_LEIR, gamma0 = self.LEIR_gamma_inj)
        beamParams_LEIR = BeamParams_LEIR()
        beamParams_LEIR.Nb = Nb_LEIR
        if self.ion_type == 'Ca':
            beamParams_LEIR.sigma_delta = beamParams_LEIR.sigma_delta_Ca
        growth_rates_leir = IBS.get_growth_rates(leir_line, beamParams_LEIR)
        
        # PS growth rates
        ps_line = self.line_PS_Pb0.copy()
        ps_line.particle_ref = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_PS, gamma0 = self.PS_gamma_inj)
        beamParams_PS = BeamParams_PS()
        beamParams_PS.Nb = Nb_PS
        if self.ion_type == 'Ca':
            beamParams_PS.sigma_delta = beamParams_PS.sigma_delta_Ca
            print('Updating sigma delta for Ca special case!')
        growth_rates_ps = IBS.get_growth_rates(ps_line, beamParams_PS)
        
        # SPS growth rates
        sps_line = self.line_SPS_Pb0.copy()
        sps_line.particle_ref = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_SPS, gamma0 = self.SPS_gamma_inj)
        beamParams_SPS = BeamParams_SPS()
        beamParams_SPS.Nb = Nb_SPS
        if self.ion_type == 'Ca':
            beamParams_SPS.sigma_delta = beamParams_SPS.sigma_delta_Ca
        growth_rates_sps = IBS.get_growth_rates(sps_line, beamParams_SPS)
        
        
        print('\nSPS beam params for {}: {}'.format(self.ion_type, beamParams_SPS))
        print('IBS growth rates calculated for {}'.format(self.ion_str))
        print(growth_rates_sps)
        print('Particle:')
        sps_line.particle_ref.show()

        growth_rates_dict = {'LEIR Tx': growth_rates_leir[0],
                             'LEIR Ty': growth_rates_leir[1],
                             'LEIR Tz': growth_rates_leir[2],
                             'PS Tx': growth_rates_ps[0],
                             'PS Ty': growth_rates_ps[1],
                             'PS Tz': growth_rates_ps[2],
                             'SPS Tx': growth_rates_sps[0],
                             'SPS Ty': growth_rates_sps[1], 
                             'SPS Tz': growth_rates_sps[2]}
        
        return self.ion_str, growth_rates_dict


    def calculate_IBS_growth_rates_all_ion_species(self, output_name='IBS_growth_rates.csv'):
        """
        Calculate analytical Nagaitsev IBS growth rates for all ions in LEIR, PS and SPS

        Parameters
        ----------
        output_name : str
            name of csv file to be generated

        Returns
        -------
        df_IBS : pd.DataFrame
            dataframe containing all IBS growth rates
        """        
        # Check that output directory exists
        os.makedirs('output_csv', exist_ok=True)

        # Iterate over all ions in data 
        all_ion_IBS_dict = {}
        for ion_type in self.full_ion_data.columns:
            
            # Initiate the correct ion and calculate growth rates
            self.init_ion(ion_type)
            
            # calculate propagated bunch intensity
            result = self.calculate_LHC_bunch_intensity() 
            Nb_LEIR = result['LEIR_space_charge_limit']
            Nb_PS = result['PS_space_charge_limit']
            Nb_SPS = result['SPS_space_charge_limit']
            
            ion_str, growth_rates_dict = self.calculate_IBS_growth_rates(Nb_LEIR, Nb_PS, Nb_SPS)
            all_ion_IBS_dict[ion_str] = growth_rates_dict

        df_IBS = pd.DataFrame(all_ion_IBS_dict)
        df_IBS.to_csv('output_csv/{}'.format(output_name))

        return df_IBS


    def calculate_LHC_bunch_intensity_all_ion_species(self, save_csv=True, output_name='output'):
        """
        Estimate LHC bunch intensity for all ion species provided in table
        through Linac3, LEIR, PS and SPS considering all the limits of the injectors

        Returns:
        --------
        df_all_ions : pd.DataFrame
            pandas dataframe containing result of all ions
        """
        # Check that output directory exists
        os.makedirs('output_csv', exist_ok=True)
        
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
            
            # First save full CSV
            df_save = df_all_ions.copy()
            df_save0 = df_save.T
            df_save0.to_csv("output_csv/{}.csv".format(output_name))
            
            # Then save copy in exponential form with decimal number for paper - only some columns
            float_columns = df_all_ions.select_dtypes(include=['float']).columns
            for col in float_columns:
                df_save[col] = df_save[col].apply(lambda x: '{:.1e}'.format(x))
            df_SC_and_max_intensity = df_save[['LEIR_maxIntensity', 'LEIR_space_charge_limit', 'PS_maxIntensity', 'PS_space_charge_limit', 
                        'SPS_maxIntensity', 'SPS_space_charge_limit', 'LHC_ionsPerBunch', 'LHC_chargesPerBunch']]
            df_SC_and_max_intensity.to_csv("output_csv/{}_for_paper.csv".format(output_name), index=True)
            
        return df_all_ions