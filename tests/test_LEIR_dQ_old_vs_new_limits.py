"""
Calculate space charge tune shifts from "space charge limit" bunch intensity, old and new
"""
from injector_model import InjectorChain_v2, SC_Tune_Shifts, BeamParams_LEIR
import xpart as xp

# Instantiate injector chain version 2 and calculate LHC bunch intensity for all ions
inj = InjectorChain_v2(nPulsesLEIR=7)  

# LEIR particle - energy is the same
LEIR_particle = xp.Particles(mass0 = 1e9 * inj.mass_GeV, q0 = inj.Q_LEIR, gamma0 = inj.LEIR_gamma_inj)

# Instantiate space charge calculator
leir_sc = SC_Tune_Shifts()
ways = ['old', 'new']
Nb_limits = [1.85e9, 1e9]

for i, Nb in enumerate(Nb_limits):

    # Find tune shifts of space charge limits
    dQx, dQy = leir_sc.calculate_SC_tuneshift(twissTableXsuite=inj.twiss_LEIR_Pb0,
                                        particle_ref=LEIR_particle,
                                        line_length=inj.line_LEIR_length, 
                                        Nb=Nb,
                                        beamParams=BeamParams_LEIR)
    print('\n{} with Nb={:.2e}: dQx = {:.4f}, dQy = {:.4f}'.format(ways[i], Nb, dQx, dQy))