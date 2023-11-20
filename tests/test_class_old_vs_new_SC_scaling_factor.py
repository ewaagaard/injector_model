"""
Test class to compare simplified space charge scaling factor in old injector model
vs new full SC lattice integral

"""
from injector_model import InjectorChain_full_SC
import pandas as pd
import numpy as np


class TestClass_old_vs_new_SC:
    """
    Compare for given SC tune shift (assuming that we are close to the limit):
    - exact expression from space charge tune shift Eq (1) in Hannes' and Isabelle's space charge report
    at https://cds.cern.ch/record/2749453
    - simplified scaling factor assuming lattice integral is constant and that this scaling 
    stays constant
    """
    #@staticmethod
    def test_SC_scaling_factor_O(self):
    
        # Instantiate new model with full lattice integrals 
        inj = InjectorChain_full_SC('Xe')
        inj.simulate_injection()

        ###################### LEIR ######################
        # Calculate for Pb
        IntX_Pb_LEIR, IntY_Pb_LEIR = self.calculate_SC_integral(inj.twiss0_LEIR_interpolated, 
                                                                inj.sigma_x0_LEIR, 
                                                                inj.sigma_y0_LEIR)
        gamma_Pb_LEIR = inj.gamma0_LEIR_inj
        beta_Pb_LEIR = inj.beta(gamma_Pb_LEIR)
        
        # Calculate for O
        IntX_O_LEIR, IntY_O_LEIR = self.calculate_SC_integral(inj.twiss_LEIR_interpolated, 
                                                                inj.sigma_x_LEIR, 
                                                                inj.sigma_y_LEIR)
        gamma_O_LEIR = inj.gamma_LEIR_inj
        beta_O_LEIR = inj.beta(gamma_O_LEIR)
        
        # Calculate difference between new and old scaling 
        factor_LEIR_X = (gamma_O_LEIR * beta_O_LEIR * IntX_O_LEIR) / (gamma_Pb_LEIR * beta_Pb_LEIR * IntX_Pb_LEIR)
        factor_LEIR_Y = (gamma_O_LEIR * beta_O_LEIR * IntY_O_LEIR) / (gamma_Pb_LEIR * beta_Pb_LEIR * IntY_Pb_LEIR)
        
        print('LEIR')
        print('Gamma O LEIR: {}'.format(gamma_O_LEIR))
        print('O charge LEIR: {}'.format(inj.Q_LEIR))
        print('IntX  O LEIR: {}'.format(IntX_O_LEIR))
        print('LEIR new/old: X = {}, Y = {}\n'.format(factor_LEIR_X, factor_LEIR_Y))
        
        ###################### PS ######################
        # Calculate for Pb
        IntX_Pb_PS, IntY_Pb_PS = self.calculate_SC_integral(inj.twiss0_PS_interpolated, 
                                                                inj.sigma_x0_PS, 
                                                                inj.sigma_y0_PS)
        gamma_Pb_PS = inj.gamma0_PS_inj
        beta_Pb_PS = inj.beta(gamma_Pb_PS)
        
        # Calculate for O
        IntX_O_PS, IntY_O_PS = self.calculate_SC_integral(inj.twiss_PS_interpolated, 
                                                                inj.sigma_x_PS, 
                                                                inj.sigma_y_PS)
        gamma_O_PS = inj.gamma_PS_inj
        beta_O_PS = inj.beta(gamma_O_PS)
        
        # Calculate difference between new and old scaling 
        factor_PS_X = (gamma_O_PS * beta_O_PS * IntX_O_PS) / (gamma_Pb_PS * beta_Pb_PS * IntX_Pb_PS)
        factor_PS_Y = (gamma_O_PS * beta_O_PS * IntY_O_PS) / (gamma_Pb_PS * beta_Pb_PS * IntY_Pb_PS)
        
        print('PS')
        print('Gamma O PS: {}'.format(gamma_O_PS))
        print('O charge PS: {}'.format(inj.Q_PS))
        print('IntX  O PS: {}'.format(IntX_O_PS))
        print('PS new/old: X = {}, Y = {}\n'.format(factor_PS_X, factor_PS_Y))
        
        ###################### SPS ######################
        # Calculate for Pb
        IntX_Pb_SPS, IntY_Pb_SPS = self.calculate_SC_integral(inj.twiss0_SPS_interpolated, 
                                                                inj.sigma_x0_SPS, 
                                                                inj.sigma_y0_SPS)
        gamma_Pb_SPS = inj.gamma0_SPS_inj
        beta_Pb_SPS = inj.beta(gamma_Pb_SPS)
        
        # Calculate for O
        IntX_O_SPS, IntY_O_SPS = self.calculate_SC_integral(inj.twiss_SPS_interpolated, 
                                                                inj.sigma_x_SPS, 
                                                                inj.sigma_y_SPS)
        gamma_O_SPS = inj.gamma_SPS_inj
        beta_O_SPS = inj.beta(gamma_O_SPS)
        
        # Calculate difference between new and old scaling 
        factor_SPS_X = (gamma_O_SPS * beta_O_SPS * IntX_O_SPS) / (gamma_Pb_SPS * beta_Pb_SPS * IntX_Pb_SPS)
        factor_SPS_Y = (gamma_O_SPS * beta_O_SPS * IntY_O_SPS) / (gamma_Pb_SPS * beta_Pb_SPS * IntY_Pb_SPS)
        
        print('SPS')
        print('SPS new/old: X = {}, Y = {}'.format(factor_SPS_X, factor_SPS_Y))
        print('Gamma O SPS: {}'.format(gamma_O_SPS))
        print('O charge SPS: {}'.format(inj.Q_SPS))
        print('IntX  O SPS: {}'.format(IntX_O_SPS))
        
        # Assert differences 
        assert np.all(np.isclose(1.0, factor_LEIR_X, rtol=1e-3))
        assert np.all(np.isclose(1.0, factor_LEIR_Y, rtol=1e-3))
        
        assert np.all(np.isclose(1.0, factor_PS_X, rtol=1e-3))
        assert np.all(np.isclose(1.0, factor_PS_Y, rtol=1e-3))

        assert np.all(np.isclose(1.0, factor_SPS_X, rtol=1e-3))
        assert np.all(np.isclose(1.0, factor_SPS_Y, rtol=1e-3))


    def calculate_SC_integral(self,
                              twiss_xtrack_interpolated, 
                              sigma_x, 
                              sigma_y
                              ):
        """Lattice integral with numpy trapezois integration"""
        
        # Load interpolated beam sizes and Twiss parameters, then define SC integrands 
        integrand_x = twiss_xtrack_interpolated['betx'] / (sigma_x * (sigma_x + sigma_y))  
        integrand_y = twiss_xtrack_interpolated['bety'] / (sigma_y * (sigma_x + sigma_y)) 
        
        IntX = np.trapz(integrand_x, x = twiss_xtrack_interpolated['s'])
        IntY = np.trapz(integrand_y, x = twiss_xtrack_interpolated['s'])
        
        return IntX, IntY

a = TestClass_old_vs_new_SC()    
a.test_SC_scaling_factor_O()
    