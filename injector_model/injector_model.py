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
from .sequence_makers import Sequences

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
    round_number_of_ecool_injections_up : bool
        whether to round up the number of Pb LEIR injections, and not down. Default is False, i.e. round down 
    """
    def __init__(self, ion_type='Pb', 
                 nPulsesLEIR = None,
                 LEIR_bunches = 2,
                 PS_splitting = 2,
                 account_for_LEIR_ecooling=True,
                 LEIR_PS_strip=False,
                 account_for_PS_rest_gas=True,
                 round_number_of_LEIR_inj_up=False
                 ):
        
        # Import reference data and initiate ion
        self.full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), header=0, index_col=0).T
        self.LEIR_PS_strip = LEIR_PS_strip
        self.account_for_PS_rest_gas = account_for_PS_rest_gas
        self.account_for_LEIR_ecooling = account_for_LEIR_ecooling
        self.round_number_of_ecool_injections_up = round_number_of_LEIR_inj_up
        self.init_ion(ion_type)
                
        # Rules for splitting and bunches 
        if nPulsesLEIR is None:
            self.nPulsesLEIR_default = Reference_Values.max_injections_into_LEIR
        else:
            self.nPulsesLEIR_default = nPulsesLEIR
        self.LEIR_bunches = LEIR_bunches
        self.PS_splitting = PS_splitting

    def Lambda(self, charge, m, gamma, charge_0, m_0, gamma_0):
        """
        Compute ratio for space charge intensity limit for new ion species for given bunch intensity 
        Nb_0 and parameters gamma_0, charge0, m_0 from reference ion species - assuming
        that space charge stays constant, and that geometric
        emittance and bunch length are constant for all ion species
        """
        Lambda = (m/m_0)*(charge_0/charge)**2*( (gamma*(gamma**2-1)) / (gamma_0*(gamma_0**2-1)))  
        return Lambda 

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
            print('Setting custom ion data')

        self.mass_GeV = self.ion_data['mass [GeV]']
        self.Z = self.ion_data['Z']
        self.A = self.ion_data['A']
        
        # Set charge state for diffent accelerators
        self.Q_LEIR = self.ion_data['Q before stripping']
        self.Q_PS = self.ion_data['Z'] if self.LEIR_PS_strip else self.ion_data['Q before stripping']
        self.Q_SPS = self.ion_data['Z']
        
        # Values from first tables in Roderik's notebook
        self.linac3_current = self.ion_data['Linac3 current [uA]'] * 1e-6
        self.linac3_pulseLength = 300e-6 # 2024 values  # self.ion_data['Linac3 pulse length [us]'] * 1e-6
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
            if Q == Q_default and A == A_default:
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


    def calculate_LHC_bunch_intensity(self):
        """
        Estimate LHC bunch intensity for a given ion species
        through Linac3, LEIR, PS and SPS considering full lattice integral space charge limits of the injectors
        """        
        # Instantiate reference values
        ref = Reference_Values(self.ion_type, self.Q_PS, self.PS_splitting, self.LEIR_PS_strip, self.account_for_PS_rest_gas)
        
        ### LINAC3 ### 
        ionsPerPulseLinac3 = (self.linac3_current * self.linac3_pulseLength) / (self.Q_LEIR * constants.e)
        
        ### LEIR ###
        Lambda_LEIR = self.Lambda(charge = self.Q_LEIR, 
                                           m = self.mass_GeV, 
                                           gamma = self.LEIR_gamma_inj,
                                           charge_0 = ref.Q0_LEIR, 
                                           m_0 = ref.m0_GeV, 
                                           gamma_0 = ref.gamma0_LEIR_inj) 
        spaceChargeLimitLEIR = ref.Nb0_LEIR_inj * Lambda_LEIR # what we can successfully inject
                             
        # Calculate number of bunches to inject if we consider electron cooling
        if self.account_for_LEIR_ecooling:
            
            # Decide whether to round the number of pulses up or down
            num_LEIR_injections_float = ref.max_injections_into_LEIR / self.relative_ecooling_time_leir
            if self.round_number_of_ecool_injections_up:
                num_injections_LEIR_with_ecooling = math.ceil(num_LEIR_injections_float)
                print('Rounding number of LEIR injections UP, from {:.3f}!'.format(num_LEIR_injections_float))
            else:
                num_injections_LEIR_with_ecooling = math.floor(num_LEIR_injections_float)
                print('Rounding number of LEIR injections DOWN, from {:.3f}!'.format(num_LEIR_injections_float))
            
            # Ensure that value does not exceed 7 or is inferior to 1
            if num_injections_LEIR_with_ecooling > 1:
                self.nPulsesLEIR = np.min([num_injections_LEIR_with_ecooling, self.nPulsesLEIR_default])
            else:
                self.nPulsesLEIR = 1
            print('Number of LEIR inejctions considering e-cooling: {}'.format(self.nPulsesLEIR))
        else:
            self.nPulsesLEIR = self.nPulsesLEIR_default
        
        totalIntLEIR = ionsPerPulseLinac3 * self.nPulsesLEIR * ref.LEIR_injection_efficiency
        ionsPerBunchExtractedLEIR = ref.LEIR_transmission * np.min([totalIntLEIR, spaceChargeLimitLEIR]) / self.LEIR_bunches
        LEIR_space_charge_limit_hit = True if totalIntLEIR > spaceChargeLimitLEIR else False 
        
        #### PS ####
        ionsPerBunchInjectedPS = ionsPerBunchExtractedLEIR * (self.LEIR_PS_stripping_efficiency if self.LEIR_PS_strip else ref.LEIR_PS_Transmission)
        
        # Hypothetical space charge limit - calculate just to project what it would be
        Lambda_PS = self.Lambda(charge = self.Q_PS, 
                                           m = self.mass_GeV, 
                                           gamma = self.PS_gamma_inj,
                                           charge_0 = ref.Q0_PS, 
                                           m_0 = ref.m0_GeV, 
                                           gamma_0 = ref.gamma0_PS_inj) 
        spaceChargeLimitPS = ref.Nb0_PS_inj * Lambda_PS # what we can successfully inject
        
        # Check that injected momentum is not too low for the PS B-field
        self.p_PS_inj = self.calcMomentum_from_gamma(self.PS_gamma_inj, self.Q_PS)
        self.Brho_PS_inj = self.calcBrho(self.p_PS_inj, self.Q_PS) # same as LEIR extraction if no stripping, else will be different 
        B_PS_inj = self.Brho_PS_inj / ref.PS_rho
        if B_PS_inj < ref.PS_MinB:
            self.PS_B_field_is_too_low = True
        elif B_PS_inj > ref.PS_MaxB:
            print("\nA = {}, Q_PS = {}, m_ion = {:.2f} GeV, Z = {}".format(self.A, self.Q_PS, self.mass_GeV, self.Z))
            print('B = {:.4f} in PS at injection is too HIGH!'.format(B_PS_inj))
            raise ValueError("B field in PS is too high!")
        else:
            self.PS_B_field_is_too_low = False
        
        # Apply transmission and splitting map on injected PS bunches
        ionsPerBunchExtracted_PS = ionsPerBunchInjectedPS * ref.PS_transmission / self.PS_splitting # maximum intensity without SC
        
        #### SPS #### 
        
        # Calculate maximum injected bunch intensity for SPS
        ionsPerBunchSPSinj = ionsPerBunchExtracted_PS * (ref.PS_SPS_transmission_efficiency if self.LEIR_PS_strip else ref.PS_SPS_stripping_efficiency)
        
        # Calculate SPS space charge limit
        Lambda_SPS = self.Lambda(charge = self.Q_SPS, 
                                m = self.mass_GeV, 
                                gamma = self.SPS_gamma_inj,
                                charge_0 = ref.Q0_SPS, 
                                m_0 = ref.m0_GeV, 
                                gamma_0 = ref.gamma0_SPS_inj) 
        spaceChargeLimitSPS = ref.Nb0_SPS_inj * Lambda_SPS 
        SPS_space_charge_limit_hit = True if ionsPerBunchSPSinj > spaceChargeLimitSPS else False
        ionsPerBunchLHC = min(spaceChargeLimitSPS, ionsPerBunchSPSinj) * ref.SPS_transmission * ref.SPS_to_LHC_transmission

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
            "LEIR_injection_efficiency": ref.LEIR_injection_efficiency, 
            "LEIR_no_bunches": self.LEIR_bunches,
            "LEIR_transmission": ref.LEIR_transmission, 
            "PS_splitting": self.PS_splitting, 
            "PS_transmission": ref.PS_transmission, 
            "PS_SPS_stripping_efficiency": ref.PS_SPS_stripping_efficiency, 
            "SPS_transmission": ref.SPS_transmission, 
            "Linac3_ionsPerPulse": ionsPerPulseLinac3,
            "LEIR_maxIntensity": totalIntLEIR,
            "LEIR_space_charge_limit": spaceChargeLimitLEIR,
            "LEIR_extractedIonPerBunch": ionsPerBunchExtractedLEIR,
            "PS_maxIntensity": ionsPerBunchInjectedPS,
            "PS_ionsExtractedPerBunch":  ionsPerBunchExtracted_PS,
            "SPS_maxIntensity": ionsPerBunchSPSinj,
            "SPS_space_charge_limit": spaceChargeLimitSPS,
            "SPS_extracted_ions_per_bunch": min(spaceChargeLimitSPS, ionsPerBunchSPSinj) * ref.SPS_transmission,
            "LHC_ionsPerBunch": ionsPerBunchLHC,
            "LHC_chargesPerBunch": ionsPerBunchLHC * self.Z,
            "LEIR_gamma_inj": self.LEIR_gamma_inj,
            "LEIR_gamma_extr": self.LEIR_gamma_extr,
            "PS_gamma_inj": self.PS_gamma_inj,
            "PS_gamma_extr": self.PS_gamma_extr,
            "SPS_gamma_inj": self.SPS_gamma_inj,
            "SPS_gamma_extr": self.SPS_gamma_extr,
            "PS_B_field_is_too_low": self.PS_B_field_is_too_low,
            "PS_space_charge_limit_hypothetical": spaceChargeLimitPS,
            "LEIR_space_charge_limit_hit": LEIR_space_charge_limit_hit,
            "SPS_space_charge_limit_hit": SPS_space_charge_limit_hit,
            "LEIR_ratio_SC_limit_maxIntensity": spaceChargeLimitLEIR / totalIntLEIR,
            "SPS_ratio_SC_limit_maxIntensity": spaceChargeLimitSPS / ionsPerBunchSPSinj,
            "LEIR_Lambda": Lambda_LEIR,
            "SPS_Lambda": Lambda_SPS
        }
        print('SPS gamma0: {:.5f}'.format(ref.gamma0_SPS_inj))

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
        seq = Sequences()
        
        # LEIR growth rates
        #leir_line = self.line_LEIR_Pb0.copy()
        #leir_line.particle_ref = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_LEIR, gamma0 = self.LEIR_gamma_inj)
        leir_line = seq.get_LEIR_line(self.mass_GeV, self.Q_LEIR, self.LEIR_gamma_inj)
        
        beamParams_LEIR = BeamParams_LEIR()
        beamParams_LEIR.Nb = Nb_LEIR
        if self.ion_type == 'Ca':
            beamParams_LEIR.sigma_delta = beamParams_LEIR.sigma_delta_Ca
        growth_rates_leir = IBS.get_growth_rates(leir_line, beamParams_LEIR)
        
        # PS growth rates
        #ps_line = self.line_PS_Pb0.copy()
        #ps_line.particle_ref = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_PS, gamma0 = self.PS_gamma_inj)
        
        ps_line = seq.get_LEIR_line(self.mass_GeV, self.Q_PS, self.PS_gamma_inj)
        beamParams_PS = BeamParams_PS()
        beamParams_PS.Nb = Nb_PS
        if self.ion_type == 'Ca':
            beamParams_PS.sigma_delta = beamParams_PS.sigma_delta_Ca
            print('Updating sigma delta for Ca special case!')
        growth_rates_ps = IBS.get_growth_rates(ps_line, beamParams_PS)
        
        # SPS growth rates
        #sps_line = self.line_SPS_Pb0.copy()
        #sps_line.particle_ref = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_SPS, gamma0 = self.SPS_gamma_inj)
        sps_line = seq.get_LEIR_line(self.mass_GeV, self.Q_SPS, self.SPS_gamma_inj)
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


    def calculate_IBS_growth_rates_all_ion_species(self, output_name='IBS_growth_rates.csv', save_file=False):
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
        os.makedirs('output/output_csv', exist_ok=True)

        # Iterate over all ions in data 
        all_ion_IBS_dict = {}
        for ion_type in self.full_ion_data.columns:
            
            # Initiate the correct ion and calculate growth rates
            self.init_ion(ion_type)
            
            # calculate propagated bunch intensity
            result = self.calculate_LHC_bunch_intensity() 
            Nb_LEIR = result['LEIR_space_charge_limit']
            Nb_PS = result['PS_space_charge_limit_hypothetical']
            Nb_SPS = result['SPS_space_charge_limit']
            
            ion_str, growth_rates_dict = self.calculate_IBS_growth_rates(Nb_LEIR, Nb_PS, Nb_SPS)
            all_ion_IBS_dict[ion_str] = growth_rates_dict

        df_IBS = pd.DataFrame(all_ion_IBS_dict)
        if save_file:
            df_IBS.to_csv('output/output_csv/{}'.format(output_name))

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
        os.makedirs('output', exist_ok=True)
        os.makedirs('output/output_csv', exist_ok=True)
        
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
            df_save0.to_csv("output/output_csv/{}.csv".format(output_name))
            
            # Then save copy in exponential form with decimal number for paper - only some columns
            #float_columns = df_all_ions.select_dtypes(include=['float']).columns
            #for col in float_columns:
            #    df_save[col] = df_save[col].apply(lambda x: '{:.1e}'.format(x))
            #df_SC_and_max_intensity = df_save[['LEIR_maxIntensity', 'LEIR_space_charge_limit', 'PS_maxIntensity', 'PS_space_charge_limit', 
            #            'SPS_maxIntensity', 'SPS_space_charge_limit', 'LHC_ionsPerBunch', 'LHC_chargesPerBunch']]
            #df_save.to_csv("output/output_csv/{}_for_paper.csv".format(output_name), index=True)
            
        return df_all_ions