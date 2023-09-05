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
from scipy import constants
import xtrack as xt
import xpart as xp
from collections import defaultdict

# Calculate the absolute path to the data folder relative to the module's location
data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()
output_folder = Path(__file__).resolve().parent.joinpath('../output').absolute()
ibs_folder = Path(__file__).resolve().parent.joinpath('../IBS_for_Xsuite').absolute()

# Import IBS module after "pip install -e IBS_for_Xsuite" has been executed 
from lib.IBSfunctions import NagaitsevIBS

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
        
        self.full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), sep=';', header=0, index_col=0).T
        self.LEIR_PS_strip = LEIR_PS_strip
        self.account_for_SPS_transmission = account_for_SPS_transmission

        ###### Load standard beam parameters ##### - used from John and Bartosik, 2021 (https://cds.cern.ch/record/2749453)
        self.Nb0_LEIR = 1e9
        self.ex_LEIR = 0.4e-6
        self.ey_LEIR = 0.4e-6
        self.sigma_z_LEIR = 4.256
        self.delta_LEIR = 1.18e-3
        
        self.Nb0_PS = 8.1e8
        self.ex_PS = 0.8e-6
        self.ey_PS = 0.5e-6
        self.sigma_z_PS = 4.74
        self.delta_PS = 0.63e-3
        
        self.Nb0_SPS = 3.5e8
        self.ex_SPS = 1.3e-6
        self.ey_SPS = 0.9e-6
        self.sigma_z_SPS = 0.23
        self.delta_SPS = 1e-3
        
        ####################################
        
        # Initiate ion and reference Pb ion type 
        self.init_ion(ion_type)
        self.ion0_referenceValues() 
        
        self.debug_mode = False
        
        # Initiate values for PS B-field
        self.PS_MinB    = 383 * 1e-4 # [T] - minimum magnetic field in PS, (Gauss to Tesla) from Heiko Damerau
        self.PS_MaxB    = 1.26 # [T] - minimum magnetic field in PS, from reyes Alemany Fernandez
        self.PS_rho     = 70.1206 # [m] - PS bending radius 
        
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
        
        ################## Pb energies ##################
        # Use reference data from current Pb ions in the CERN accelerator complex 
        self.m0_GeV = 193.687 # rest mass in GeV for Pb reference case 
        self.Z0 = 82.0
        
        ###### LEIR ##### - reference case for Pb54+ --> BEFORE stripping
        self.Nq0_LEIR_extr = 10e10  # number of observed charges extracted at LEIR
        self.Q0_LEIR = 54.0
        self.Nb0_LEIR_extr = self.Nq0_LEIR_extr/self.Q0_LEIR
        self.gamma0_LEIR_inj = (self.m0_GeV + self.E_kin_per_A_LEIR_inj * 208)/self.m0_GeV
        self.gamma0_LEIR_extr = (self.m0_GeV + self.E_kin_per_A_LEIR_extr * 208)/self.m0_GeV
        
        ##### PS ##### - reference case for Pb54+ --> BEFORE stripping
        self.Nq0_PS_extr =  6e10 # from November 2022 ionlifetime MD, previously 8e10  # number of observed charges extracted at PS for nominal beam
        self.Q0_PS = 54.0
        self.Nb0_PS_extr = self.Nq0_PS_extr/self.Q0_PS
        self.gamma0_PS_inj = (self.m0_GeV + self.E_kin_per_A_PS_inj * 208)/self.m0_GeV
        self.gamma0_PS_extr = (self.m0_GeV + self.E_kin_per_A_PS_extr * 208)/self.m0_GeV
        
        ##### SPS ##### - reference case for Pb82+ --> AFTER stripping
        if not self.account_for_SPS_transmission:
            self.SPS_transmission = 1.0
        self.Nb0_SPS_extr = 2.21e8/self.SPS_transmission # outgoing ions per bunch from SPS (2015 values), adjusted for 62% transmission
        self.Q0_SPS = 82.0
        self.Nq0_SPS_extr = self.Nb0_SPS_extr*self.Q0_SPS
        self.gamma0_SPS_inj = (self.m0_GeV + self.E_kin_per_A_SPS_inj * 208)/self.m0_GeV
        self.gamma0_SPS_extr = (self.m0_GeV + self.E_kin_per_A_SPS_extr * 208)/self.m0_GeV
    
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
    
    
    ##### Methods for full space charge integral calculations #####
    def interpolate_Twiss_table(self, 
                                twissTableXsuite,
                                line,
                                particle_ref,
                                ex,
                                ey,
                                delta, 
                                interpolation_resolution = 100000
                                ):
        """
        Interpolate Xtrack twiss table and beam sizes sigma_x and sigma_y
        """
        gamma = particle_ref.gamma0[0]
        beta = self.beta(gamma)
        ss = np.linspace(0, line.get_length(), interpolation_resolution)
        data2=np.zeros((interpolation_resolution, 8))
        data2[:,1] = np.square(np.interp(ss, twissTableXsuite['s'], np.sqrt(twissTableXsuite['betx'])))
        data2[:,2] = np.square(np.interp(ss, twissTableXsuite['s'], np.sqrt(twissTableXsuite['bety'])))
        data2[:,3] = np.interp(ss, twissTableXsuite['s'], beta * twissTableXsuite['dx'])
        data2[:,4] = np.interp(ss, twissTableXsuite['s'], beta * twissTableXsuite['dy'])
        data2[:,5] = np.interp(ss, twissTableXsuite['s'], twissTableXsuite['mux'])
        data2[:,6] = np.interp(ss, twissTableXsuite['s'], twissTableXsuite['muy'])
        data2[:,7] += line.get_length()/len(ss)
        data2[:,0] = ss    
        data = data2
        columns = ['s', 'betx', 'bety', 'dx', 'dy', 'mux', 'muy', 'l']
        
        # Interpolate Twiss tables and beam sizes 
        twiss_xtrack_interpolated = pd.DataFrame(data, columns = columns)
        sigma_x = np.sqrt(ex * twiss_xtrack_interpolated['betx'] / (beta * gamma) + (delta * twiss_xtrack_interpolated['dx'])**2)
        sigma_y = np.sqrt(ey * twiss_xtrack_interpolated['bety'] / (beta * gamma) + (delta * twiss_xtrack_interpolated['dy'])**2)
        
        return twiss_xtrack_interpolated, sigma_x, sigma_y
    
    

    def calculate_SC_tuneshift(self, Nb, particle_ref, sigma_z, twiss_xtrack_interpolated, sigma_x, sigma_y):
        """
        Finds the SC-induced max detuning dQx and dQy for given Twiss and beam parameters
        """  
        gamma = particle_ref.gamma0[0]
        beta = self.beta(gamma)
        r0 = particle_ref.get_classical_particle_radius0()
        
        # Space charge perveance 
        K_sc = (2 * r0 * Nb) / (beta**2 * gamma**3 * np.sqrt(2*np.pi) * sigma_z)
        
        integrand_x = twiss_xtrack_interpolated['betx'] / (sigma_x * (sigma_x + sigma_y))  
        integrand_y = twiss_xtrack_interpolated['bety'] / (sigma_y * (sigma_x + sigma_y)) 
        
        dQx = - K_sc / (4 * np.pi) * np.trapz(integrand_x, x = twiss_xtrack_interpolated['s'])
        dQy = - K_sc / (4 * np.pi) * np.trapz(integrand_y, x = twiss_xtrack_interpolated['s'])
        
        return dQx, dQy
    
    
    def calculate_SC_tuneshift_for_LEIR(self, Nb, gamma, sigma_z):
        """
        Finds the SC-induced max detuning dQx and dQy for LEIR for given beam parameters 
        assuming for now that emittances and momentum spread delta are identical
        Input: arrays with bunch intensities, gammas and bunch length for LEIR
        """ 
        #### LEIR ####
        particle_LEIR = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_LEIR, gamma0 = gamma)
        line_LEIR = self.line_LEIR_Pb.copy()
        line_LEIR.reference_particle = particle_LEIR
        line_LEIR.build_tracker()
        twiss_LEIR = line_LEIR.twiss()
        twiss_LEIR_interpolated, sigma_x_LEIR, sigma_y_LEIR = self.interpolate_Twiss_table(twiss_LEIR, 
                                                                                            line_LEIR, 
                                                                                            particle_LEIR, 
                                                                                            self.ex_LEIR, 
                                                                                            self.ey_LEIR,
                                                                                            self.delta_LEIR,
                                                                                            )
        dQx_LEIR, dQy_LEIR = self.calculate_SC_tuneshift(Nb, particle_LEIR, sigma_z, 
                         twiss_LEIR_interpolated, sigma_x_LEIR, sigma_y_LEIR)
        
        return dQx_LEIR, dQy_LEIR


    def calculate_SC_tuneshift_for_PS(self, Nb, gamma, sigma_z):
        """
        Finds the SC-induced max detuning dQx and dQy for PS for given beam parameters 
        assuming for now that emittances and momentum spread delta are identical
        Input: arrays with bunch intensities, gammas and bunch length for PS
        """ 
        #### PS ####
        particle_PS = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_PS, gamma0 = gamma)
        line_PS = self.line_PS_Pb.copy()
        line_PS.reference_particle = particle_PS
        line_PS.build_tracker()
        twiss_PS = line_PS.twiss()
        twiss_PS_interpolated, sigma_x_PS, sigma_y_PS = self.interpolate_Twiss_table(twiss_PS, 
                                                                                            line_PS, 
                                                                                            particle_PS, 
                                                                                            self.ex_PS, 
                                                                                            self.ey_PS,
                                                                                            self.delta_PS,
                                                                                            )
        dQx_PS, dQy_PS = self.calculate_SC_tuneshift(Nb, particle_PS, sigma_z, 
                         twiss_PS_interpolated, sigma_x_PS, sigma_y_PS)
        
        return dQx_PS, dQy_PS        
    
    
    def calculate_SC_tuneshift_for_SPS(self, Nb, gamma, sigma_z):
        """
        Finds the SC-induced max detuning dQx and dQy for SPS for given beam parameters 
        assuming for now that emittances and momentum spread delta are identical
        Input: arrays with bunch intensities, gammas and bunch length for SPS 
        """ 
        #### SPS ####
        particle_SPS = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_SPS, gamma0 = gamma)
        line_SPS = self.line_SPS_Pb.copy()
        line_SPS.reference_particle = particle_SPS
        line_SPS.build_tracker()
        twiss_SPS = line_SPS.twiss()
        twiss_SPS_interpolated, sigma_x_SPS, sigma_y_SPS = self.interpolate_Twiss_table(twiss_SPS, 
                                                                                            line_SPS, 
                                                                                            particle_SPS, 
                                                                                            self.ex_SPS, 
                                                                                            self.ey_SPS,
                                                                                            self.delta_SPS,
                                                                                            )
        dQx_SPS, dQy_SPS = self.calculate_SC_tuneshift(Nb, particle_SPS, sigma_z, 
                         twiss_SPS_interpolated, sigma_x_SPS, sigma_y_SPS)
        
        return dQx_SPS, dQy_SPS     
    
    
    def maxIntensity_from_SC_integral(self, 
                                      dQx_max, 
                                      dQy_max, 
                                      particle_ref, 
                                      sigma_z, 
                                      twiss_xtrack_interpolated, 
                                      sigma_x, 
                                      sigma_y
                                      ):
        """
        For a given max tuneshift, calculate the maximum bunch intensity 
        """
        gamma = particle_ref.gamma0[0]
        beta = self.beta(gamma)
        r0 = particle_ref.get_classical_particle_radius0()
        
        # Load interpolated beam sizes and Twiss parameters, then define SC integrands 
        integrand_x = twiss_xtrack_interpolated['betx'] / (sigma_x * (sigma_x + sigma_y))  
        integrand_y = twiss_xtrack_interpolated['bety'] / (sigma_y * (sigma_x + sigma_y)) 

        # Calculate maximum bunch intensity for given max tune shift in x and in y
        Nb_x_max = -dQx_max / ( (2 * r0) / (4 * np.pi * beta**2 * gamma**3 * np.sqrt(2*np.pi) * sigma_z) \
                             * np.trapz(integrand_x, x = twiss_xtrack_interpolated['s']))
        Nb_y_max = -dQy_max / ( (2 * r0) / (4 * np.pi * beta**2 * gamma**3 * np.sqrt(2*np.pi) * sigma_z) \
                             * np.trapz(integrand_y, x = twiss_xtrack_interpolated['s']))
        
        return Nb_x_max, Nb_y_max
    
    ######################################################################
     
    ######################## IBS part ####################################
    def find_analytical_IBS_growth_rates(self,
                                         particle_ref,
                                         twiss,
                                         line,
                                         bunch_intensity, 
                                         sigma_z,
                                         ex,
                                         ey,
                                         sig_delta, 
                                         n_part=5000
                                         ):
        """
        For given beam parameters, calculate analytical (and kinetic?) growth rates for a given bunch intensity
        and other initial conditions at first turn 
        """

        # Create Gaussian bunch of particle object
        #particles = xp.generate_matched_gaussian_bunch(
        #                                                num_particles = n_part, total_intensity_particles = bunch_intensity,
        #                                                nemitt_x = ex, nemitt_y = ey, sigma_z = sigma_z,
        #                                                particle_ref = particle_ref, line = line
        #                                                )
    
        # ----- Initialize IBS object -----
        print('Beta: {}'.format(particle_ref.beta0[0]))
        IBS = NagaitsevIBS()
        IBS.set_beam_parameters(particle_ref)
        #IBS.betar = particle_ref.beta0[0]
        IBS.set_optic_functions(twiss)
        #print("\n\nIBS enTOT: {}\n\n".format(IBS.EnTot))       
        
        # --- Initialize first turn-by-turn data for all modes 
        #sig_x = np.std(particles.x[particles.state > 0])
        #sig_y = np.std(particles.y[particles.state > 0])
        #sig_delta = np.std(particles.delta[particles.state > 0])
        #bl = np.std(particles.zeta[particles.state > 0])
        #eps_x = (sig_x**2 - (twiss['dx'][0] * sig_delta)**2) / twiss['betx'][0]
        #eps_y = sig_y**2 / twiss['bety'][0] 
        
        # Calculate the analytical growth rates 
        IBS.calculate_integrals(
            ex,
            ey,
            sig_delta,
            sigma_z
            )
        
        return IBS.Ixx, IBS.Iyy, IBS.Ipp
    
    
    def calculate_IBS_growth_rate_for_LEIR(self, Nb, gamma, sigma_z):
        """
        Finds the initial IBS growth rates for LEIR 
        assuming for now that emittances and momentum spread delta are identical
        Input: arrays with bunch intensities, gammas and bunch length for LEIR
        """ 
        #### LEIR ####
        particle_LEIR = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_LEIR, gamma0 = gamma)
        line_LEIR = self.line_LEIR_Pb.copy()
        line_LEIR.reference_particle = particle_LEIR
        line_LEIR.build_tracker()
        twiss_LEIR = line_LEIR.twiss()
        Ixx_LEIR, Iyy_LEIR, Ipp_LEIR = self.find_analytical_IBS_growth_rates(particle_LEIR,
                                                                            twiss_LEIR, 
                                                                            line_LEIR,
                                                                            Nb,
                                                                            sigma_z,
                                                                            self.ex_LEIR, 
                                                                            self.ey_LEIR,
                                                                            self.delta_LEIR,
                                                                             )
        return Ixx_LEIR, Iyy_LEIR, Ipp_LEIR


    def calculate_IBS_growth_rate_for_PS(self, Nb, gamma, sigma_z):
        """
        Finds the initial IBS growth rates for PS 
        assuming for now that emittances and momentum spread delta are identical
        Input: arrays with bunch intensities, gammas and bunch length for PS
        """ 
        #### PS ####
        particle_PS = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_PS, gamma0 = gamma)
        line_PS = self.line_PS_Pb.copy()
        line_PS.reference_particle = particle_PS
        line_PS.build_tracker()
        twiss_PS = line_PS.twiss()
        Ixx_PS, Iyy_PS, Ipp_PS = self.find_analytical_IBS_growth_rates(particle_PS,
                                                                        twiss_PS, 
                                                                        line_PS,
                                                                        Nb,
                                                                        sigma_z,
                                                                        self.ex_PS, 
                                                                        self.ey_PS,
                                                                        self.delta_PS,
                                                                         )
        return Ixx_PS, Iyy_PS, Ipp_PS


    def calculate_IBS_growth_rate_for_SPS(self, Nb, gamma, sigma_z):
        """
        Finds the initial IBS growth rates for SPS 
        assuming for now that emittances and momentum spread delta are identical
        Input: arrays with bunch intensities, gammas and bunch length for SPS
        """ 
        #### SPS ####
        particle_SPS = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_SPS, gamma0 = gamma)
        line_SPS = self.line_SPS_Pb.copy()
        line_SPS.reference_particle = particle_SPS
        line_SPS.build_tracker()
        twiss_SPS = line_SPS.twiss()
        Ixx_SPS, Iyy_SPS, Ipp_SPS = self.find_analytical_IBS_growth_rates(particle_SPS,
                                                                        twiss_SPS, 
                                                                        line_SPS,
                                                                        Nb,
                                                                        sigma_z,
                                                                        self.ex_SPS, 
                                                                        self.ey_SPS,
                                                                        self.delta_SPS,
                                                                         )
        return Ixx_SPS, Iyy_SPS, Ipp_SPS

 
    ######################################################################
    
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
    