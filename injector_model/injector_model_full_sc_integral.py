#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation model of the CERN Injector Chain for different ions
solving for full space charge (SC) lattice integral 
- by Elias Waagaard 
"""
import pandas as pd
import numpy as np
from scipy import constants
import xtrack as xt
import xpart as xp
from collections import defaultdict

class InjectorChain_full_SC:
    """
    Representation of the CERN Injector Chain for different ions with full space charge lattice integral. 
    This model accounts for
    - full space charge integrals in LEIR, PS and SPS 
    """
    def __init__(self, ion_type, 
                 ion_data, 
                 nPulsesLEIR = 1,
                 LEIR_bunches = 2,
                 PS_splitting = 2,
                 account_for_SPS_transmission=True,
                 LEIR_PS_strip=False,
                 save_path_csv = '../output/csv_tables'
                 ):
        
        self.full_ion_data = ion_data
        self.LEIR_PS_strip = LEIR_PS_strip
        self.account_for_SPS_transmission = account_for_SPS_transmission

        ###### Load standard beam parameters ##### - used from John and Bartosik, 2021 (https://cds.cern.ch/record/2749453)
        self.Nb0_leir = 1e9
        self.ex_leir = 0.4e-6
        self.ey_leir = 0.4e-6
        self.sigma_z_leir = 4.256
        self.delta_leir = 1.18e-3
        
        self.Nb0_ps = 8.1e8
        self.ex_ps = 0.8e-6
        self.ey_ps = 0.5e-6
        self.sigma_z_ps = 4.74
        self.delta_ps = 0.63e-3
        
        self.Nb0_sps = 3.5e8
        self.ex_sps = 1.3e-6
        self.ey_sps = 0.9e-6
        self.sigma_z_sps = 0.23
        self.delta_sps = 1e-3
        
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
        self.particle0_leir = xp.Particles(mass0 = 1e9 * self.m0_GeV, q0 = self.Q0_LEIR, gamma0 = self.gamma0_LEIR_inj)
        self.line_leir_Pb = xt.Line.from_json('../data/xtrack_sequences/LEIR_2021_Pb_ions_with_RF.json')
        self.line_leir_Pb.reference_particle = self.particle0_leir
        self.line_leir_Pb.build_tracker()
        self.twiss0_leir = self.line_leir_Pb.twiss()
        self.twiss0_leir_interpolated, self.sigma_x0_leir, self.sigma_y0_leir = self.interpolate_Twiss_table(self.twiss0_leir, 
                                                                                                             self.line_leir_Pb, 
                                                                                                             self.particle0_leir, 
                                                                                                             self.ex_leir, 
                                                                                                             self.ey_leir,
                                                                                                             self.delta_leir,
                                                                                                             )
        self.dQx0_leir, self.dQy0_leir = self.calculate_SC_tuneshift(self.Nb0_leir, self.particle0_leir, self.sigma_z_leir, 
                                                 self.twiss0_leir_interpolated, self.sigma_x0_leir, self.sigma_y0_leir)
        #print("LEIR Pb: dQx = {}, dQy = {}".format(self.dQx0_leir, self.dQy0_leir))
        
        # PS 
        self.particle0_ps = xp.Particles(mass0 = 1e9 * self.m0_GeV, q0 = self.Q0_PS, gamma0 = self.gamma0_PS_inj)
        self.line_ps_Pb = xt.Line.from_json('../data/xtrack_sequences/PS_2022_Pb_ions_matched_with_RF.json')
        self.line_ps_Pb.reference_particle = self.particle0_ps
        self.line_ps_Pb.build_tracker()
        self.twiss0_ps = self.line_ps_Pb.twiss()
        self.twiss0_ps_interpolated, self.sigma_x0_ps, self.sigma_y0_ps = self.interpolate_Twiss_table(self.twiss0_ps, 
                                                                                                             self.line_ps_Pb, 
                                                                                                             self.particle0_ps, 
                                                                                                             self.ex_ps, 
                                                                                                             self.ey_ps,
                                                                                                             self.delta_ps,
                                                                                                             )
        self.dQx0_ps, self.dQy0_ps = self.calculate_SC_tuneshift(self.Nb0_ps, self.particle0_ps, self.sigma_z_ps, 
                                                 self.twiss0_ps_interpolated, self.sigma_x0_ps, self.sigma_y0_ps)
        #print("PS Pb: dQx = {}, dQy = {}".format(self.dQx0_ps, self.dQy0_ps))
        
        # SPS 
        self.particle0_sps = xp.Particles(mass0 = 1e9 * self.m0_GeV, q0 = self.Q0_SPS, gamma0 = self.gamma0_SPS_inj)
        self.line_sps_Pb = xt.Line.from_json('../data/xtrack_sequences/SPS_2021_Pb_ions_matched_with_RF.json')
        self.line_sps_Pb.reference_particle = self.particle0_sps
        self.line_sps_Pb.build_tracker()
        self.twiss0_sps = self.line_sps_Pb.twiss()
        self.twiss0_sps_interpolated, self.sigma_x0_sps, self.sigma_y0_sps = self.interpolate_Twiss_table(self.twiss0_sps, 
                                                                                                             self.line_sps_Pb, 
                                                                                                             self.particle0_sps, 
                                                                                                             self.ex_sps, 
                                                                                                             self.ey_sps,
                                                                                                             self.delta_sps,
                                                                                                             )
        self.dQx0_sps, self.dQy0_sps = self.calculate_SC_tuneshift(self.Nb0_sps, self.particle0_sps, self.sigma_z_sps, 
                                                 self.twiss0_sps_interpolated, self.sigma_x0_sps, self.sigma_y0_sps)
        #print("SPS Pb: dQx = {}, dQy = {}".format(self.dQx0_sps, self.dQy0_sps))
        
    

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
    
    
    def maxIntensity_from_SC_integral(self, dQx_max, dQy_max, particle_ref, sigma_z, twiss_xtrack_interpolated, sigma_x, sigma_y):
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
    
    
    def leir(self):
        """
        Calculate gamma at entrance and exit of the LEIR and transmitted bunch intensity 
        using known Pb ion values from LIU report on 
        https://edms.cern.ch/ui/file/1420286/2/LIU-Ions_beam_parameter_table.pdf
        """
        # Estimate gamma at extraction
        self.gamma_LEIR_inj = (self.mass_GeV + self.E_kin_per_A_LEIR_inj * self.A)/self.mass_GeV
        self.gamma_LEIR_extr =  np.sqrt(
                                1 + ((self.Q / 54) / (self.mass_GeV/self.m0_GeV))**2
                                * (self.gamma0_LEIR_extr**2 - 1)
                                )
                 
        # Define particle object for LEIR 
        particle_LEIR = xp.Particles(mass0 = 1e9 * self.mass_GeV, 
                                     q0 = self.Q_leir, 
                                     gamma0 = self.gamma_LEIR_inj
                                     )
        
        # Twiss and beam sizes sigma 
        
        # Maximum intensity for space charge limit - keep same tune shift as today
        self.Nb_x_max_leir, self.Nb_y_max_leir = self.maxIntensity_from_SC_integral(self.dQx0_leir, self.dQy0_leir,
                                                                                    particle_LEIR, 
                                                                                    
                                                                                    
                                                                         """           
                                                                         self.particle0_leir, self.sigma_z_leir,
                                                                         self.twiss0_leir_interpolated, self.sigma_x0_leir, 
                                                                         self.sigma_y0_leir
                                                                         ) 
                                                                            """
        self.Nb_LEIR_extr = min(self.Nb_x_max_leir, self.Nb_y_max_leir)  # pick the limiting intensity 
        self.Nq_LEIR_extr = self.Nb_LEIR_extr*self.Q_LEIR  # number of outgoing charges, before any stripping
        
        # MAYBE ADD FLAG ON WHICH PLANE IS LIMITING? 

    
    def ps(self):
        """
        Calculate gamma at entrance and exit of the PS and transmitted bunch intensity 
        """
        # Estimate gamma at extraction
        self.gamma_PS_inj =  self.gamma_LEIR_extr
        self.gamma_PS_extr =  np.sqrt(
                                1 + (((self.Z if self.LEIR_PS_strip else self.Q) / 54) / (self.mass_GeV/self.m0_GeV))**2
                                * (self.gamma0_PS_extr**2 - 1)
                                )
        
        Nb_x_max_ps, Nb_y_max_ps = inj.maxIntensity_from_SC_integral(inj.dQx0_ps, inj.dQy0_ps, 
                                                                         inj.particle0_ps, inj.sigma_z_ps,
                                                                         inj.twiss0_ps_interpolated, inj.sigma_x0_ps, 
                                                                         inj.sigma_y0_ps
                                                                         ) 
    
    def sps(self):
        """
        Calculate gamma at entrance and exit of the SPS, and transmitted bunch intensity 
        Space charge limit comes from gamma at injection
        """
        # Calculate gamma at injection, simply scaling with the kinetic energy per nucleon as of today
         # consider same magnetic rigidity at PS extraction for all ion species: Brho = P/Q, P = m*gamma*beta*c
        self.gamma_SPS_inj =  np.sqrt(
                                1 + (((self.Z if self.LEIR_PS_strip else self.Q) / 54) / (self.mass_GeV/self.m0_GeV))**2
                                * (self.gamma0_SPS_inj**2 - 1)
                            )
        self.gamma_SPS_extr = (self.mass_GeV + self.E_kin_per_A_SPS_extr * self.A)/self.mass_GeV
        
        
        """
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
        """
    