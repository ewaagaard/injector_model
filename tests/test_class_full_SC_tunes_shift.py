#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General test class for Injector Class with full SC tune shif
"""
from injector_model import InjectorChain_full_SC
import pandas as pd
import numpy as np
import xpart as xp
import xtrack as xt

# Import data 
ion_data = pd.read_csv("../data/Ion_species.csv", sep=';', header=0, index_col=0).T
ion_type = 'Pb'

# load data from Bartosik and John
dQ_data = pd.read_csv('../data/test_and_benchmark_data/John_and_Bartosik_tune_shifts.csv', header=0, index_col=0)


class TestClass_injectorModel_full_SC:
    """
    Test class for comparing SC tune shifts of full lattice integral,
    using Bartosik and John (2021) as reference (https://cds.cern.ch/record/2749453)
    """
    
    def test_detuning_Pb(self):        
        inj = InjectorChain_full_SC('Pb', ion_data)
        dQ_Pb = np.array([inj.dQx0_leir, inj.dQy0_leir, inj.dQx0_ps, inj.dQy0_ps,inj.dQx0_sps, inj.dQy0_sps])
        
        assert np.all(np.isclose(dQ_Pb, dQ_data.loc['Pb'], rtol=5e-2))

    def test_detuning_O(self):
        """
        Compare SC detuning of O: custumized gamma and Nb from Bartosik and John paper 
        """
        inj = InjectorChain_full_SC('O', ion_data)
           	
        ###### LEIR ######
        Nb_LEIR = 110e8
        gamma_LEIR = 1.00451
        sigma_z_LEIR = 8.5
        	
        particle_leir = xp.Particles(mass0 = 1e9 * inj.mass_GeV, q0 = inj.Q_LEIR, gamma0 = gamma_LEIR)
        line_leir_O = xt.Line.from_json('../data/xtrack_sequences/LEIR_2021_Pb_ions_with_RF.json')
        line_leir_O.reference_particle = particle_leir
        line_leir_O.build_tracker()
        twiss_leir = line_leir_O.twiss()
        twiss_leir_interpolated, sigma_x_leir, sigma_y_leir = inj.interpolate_Twiss_table(twiss_leir, 
                                                                                            line_leir_O, 
                                                                                            particle_leir, 
                                                                                            inj.ex_leir, 
                                                                                            inj.ey_leir,
                                                                                            inj.delta_leir,
                                                                                            )
        dQx_leir, dQy_leir = inj.calculate_SC_tuneshift(Nb_LEIR, particle_leir, sigma_z_LEIR, 
                         twiss_leir_interpolated, sigma_x_leir, sigma_y_leir)
        
        ###### PS ######
        Nb_PS = 88e8
        gamma_PS = 1.07214
        sigma_z_PS = 4.74
        	
        particle_PS = xp.Particles(mass0 = 1e9 * inj.mass_GeV, q0 = inj.Q_PS, gamma0 = gamma_PS)
        line_PS_O = xt.Line.from_json('../data/xtrack_sequences/PS_2022_Pb_ions_matched_with_RF.json')
        line_PS_O.reference_particle = particle_PS
        line_PS_O.build_tracker()
        twiss_PS = line_PS_O.twiss()
        twiss_PS_interpolated, sigma_x_PS, sigma_y_PS = inj.interpolate_Twiss_table(twiss_PS, 
                                                                                            line_PS_O, 
                                                                                            particle_PS, 
                                                                                            inj.ex_ps, 
                                                                                            inj.ey_ps,
                                                                                            inj.delta_ps,
                                                                                            )
        dQx_PS, dQy_PS = inj.calculate_SC_tuneshift(Nb_PS, particle_PS, sigma_z_PS, 
                         twiss_PS_interpolated, sigma_x_PS, sigma_y_PS)
        
        ###### SPS ######
        Nb_SPS = 50e8
        gamma_SPS = 7.04405
        sigma_z_SPS = 0.23
        	
        particle_SPS = xp.Particles(mass0 = 1e9 * inj.mass_GeV, q0 = inj.Q_SPS, gamma0 = gamma_SPS)
        line_SPS_O = xt.Line.from_json('../data/xtrack_sequences/SPS_2021_Pb_ions_matched_with_RF.json')
        line_SPS_O.reference_particle = particle_SPS
        line_SPS_O.build_tracker()
        twiss_SPS = line_SPS_O.twiss()
        twiss_SPS_interpolated, sigma_x_SPS, sigma_y_SPS = inj.interpolate_Twiss_table(twiss_SPS, 
                                                                                            line_SPS_O, 
                                                                                            particle_SPS, 
                                                                                            inj.ex_sps, 
                                                                                            inj.ey_sps,
                                                                                            inj.delta_sps,
                                                                                            )
        dQx_SPS, dQy_SPS = inj.calculate_SC_tuneshift(Nb_SPS, particle_SPS, sigma_z_SPS, 
                         twiss_SPS_interpolated, sigma_x_SPS, sigma_y_SPS)
        
        # Compare with calculated tune shifts from their paper
        dQ_O = np.array([dQx_leir, dQy_leir, dQx_PS, dQy_PS, dQx_SPS, dQy_SPS])
        
        assert np.all(np.isclose(dQ_O, dQ_data.loc['O'], rtol=5e-2))
    
    
    def test_Nb_from_max_dQ(self):
        """
        Check if Nb from maximum tune shift agrees with input Nb to calculate tune shift
        --> test the formula backwards
        """    
        # First calculate Pb tune shifts from standard intensities
        inj = InjectorChain_full_SC('Pb', ion_data)
        
        # Calculate max intensities for LEIR, PS and SPS 
        Nb_x_max_leir, Nb_y_max_leir = inj.maxIntensity_from_SC_integral(inj.dQx0_leir, inj.dQy0_leir, 
                                                                         inj.particle0_leir, inj.sigma_z_leir,
                                                                         inj.twiss0_leir_interpolated, inj.sigma_x0_leir, 
                                                                         inj.sigma_y0_leir
                                                                         ) 
        Nb_x_max_ps, Nb_y_max_ps = inj.maxIntensity_from_SC_integral(inj.dQx0_ps, inj.dQy0_ps, 
                                                                         inj.particle0_ps, inj.sigma_z_ps,
                                                                         inj.twiss0_ps_interpolated, inj.sigma_x0_ps, 
                                                                         inj.sigma_y0_ps
                                                                         ) 
        Nb_x_max_sps, Nb_y_max_sps = inj.maxIntensity_from_SC_integral(inj.dQx0_sps, inj.dQy0_sps, 
                                                                         inj.particle0_sps, inj.sigma_z_sps,
                                                                         inj.twiss0_sps_interpolated, inj.sigma_x0_sps, 
                                                                         inj.sigma_y0_sps
                                                                         ) 
        ##### Compare the respective intensities - intensities should agree between x and y, and with input Nb for tune shift
        # LEIR
        assert np.isclose(Nb_x_max_leir, Nb_y_max_leir, rtol=1e-2)
        assert np.isclose(Nb_x_max_leir, inj.Nb0_leir, rtol=1e-2)
    
        # PS 
        assert np.isclose(Nb_x_max_ps, Nb_y_max_ps, rtol=1e-2)
        assert np.isclose(Nb_x_max_ps, inj.Nb0_ps, rtol=1e-2)
        
        # SPS 
        assert np.isclose(Nb_x_max_sps, Nb_y_max_sps, rtol=1e-2)
        assert np.isclose(Nb_x_max_sps, inj.Nb0_sps, rtol=1e-2)
    
    	
     
        

