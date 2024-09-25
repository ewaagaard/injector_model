"""
Main script to calculate final nucleon intensity into the LHC with new injector model version
Includes:
- Full space charge limit for LEIR and SPS, but NOT for PS
- Electron cooling (set to True)
"""
import matplotlib.pyplot as plt
import pandas as pd 
from injector_model import InjectorChain
import numpy as np
from pathlib import Path
import os

# Load ion data and initialize for test for bunch intensities 
data_folder = Path(__file__).resolve().parent.joinpath('../../../data').absolute()
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/output_for_paper', exist_ok=True)


# Compare to reference intensities - WG5 and Roderik
ref_Table_SPS = pd.read_csv('{}/test_and_benchmark_data/SPS_final_intensities_WG5_and_Hannes.csv'.format(data_folder), index_col=0)
Roderik_LHC_charges_per_bunch = pd.read_csv('{}/test_and_benchmark_data/Roderik_2021_LHC_charges_per_bunch_output.csv'.format(data_folder), index_col=0)
ref_val = Roderik_LHC_charges_per_bunch.sort_values('Z')

# Define all relevant scenarios (baseline, stripping, PS splitting, etc) in a function
def calculate_LHC_intensities_all_scenarios(output_extra_str = '', # to identify case
                                            savefig=True,
                                            generate_tables_for_paper = False,
                                            return_dataframes = False 
                                            ):
    
    ## CASE 1: BASELINE (default Pb production)
    output_1 = '1_baseline{}'.format(output_extra_str)
    injector_chain1 = InjectorChain(LEIR_bunches = 2,
                                    PS_splitting = 2,
                                    account_for_LEIR_ecooling=True,
                                    PS_factor_SC=np.inf
                                    )
    result = injector_chain1.calculate_LHC_bunch_intensity()
    
    # Calculate LHC bunch intensity for all ions
    df = injector_chain1.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = output_1)
    
    if generate_tables_for_paper:
        # output in single-decimal exponential_notation
        df_SC_and_max_intensity = df[['LEIR_maxIntensity', 'LEIR_space_charge_limit', 'PS_maxIntensity', 'PS_space_charge_limit', 
        			'SPS_maxIntensity', 'SPS_spaceChargeLimit', 'LHC_ionsPerBunch', 'LHC_chargesPerBunch']]
        for col in df_SC_and_max_intensity.columns:
            df_SC_and_max_intensity[col] = df_SC_and_max_intensity[col].apply(lambda x: '{:.1e}'.format(x))
        df_SC_and_max_intensity.to_csv('output/output_for_paper/{}_for_paper.csv'.format(output_1), index=True)
    
    
    
    ## 2: TRY WITHOUT PS SPLITTING
    output_2 ='2_no_PS_splitting{}'.format(output_extra_str)
    injector_chain2 = InjectorChain(LEIR_bunches = 2,
                                    PS_splitting = 1,
                                    account_for_LEIR_ecooling=True,
                                    PS_factor_SC=np.inf
                                    )
    df2 = injector_chain2.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name=output_2)
    
    if generate_tables_for_paper:
        df2_SC_and_max_intensity = df2[['LEIR_maxIntensity', 'LEIR_space_charge_limit', 'PS_maxIntensity', 'PS_space_charge_limit', 
        			'SPS_maxIntensity', 'SPS_spaceChargeLimit', 'LHC_ionsPerBunch', 'LHC_chargesPerBunch']]
        for col in df2_SC_and_max_intensity.columns:
            df2_SC_and_max_intensity[col] = df2_SC_and_max_intensity[col].apply(lambda x: '{:.1e}'.format(x))
        df2_SC_and_max_intensity.to_csv('output/output_for_paper/{}_for_paper.csv'.format(output_2), index=True)
    
    
    ## 3: WITH PS SPLITTING AND LEIR-PS STRIPPING
    output_3 = '3_LEIR_PS_stripping{}'.format(output_extra_str)
    
    injector_chain3 = InjectorChain(LEIR_bunches = 2,
                                    PS_splitting = 2,
                                    account_for_LEIR_ecooling=True,
                                    LEIR_PS_strip=True,
                                    PS_factor_SC=np.inf
                                    )
    
    df3 = injector_chain3.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name=output_3)
    
    if generate_tables_for_paper:
        df3_SC_and_max_intensity = df3[['LEIR_maxIntensity', 'LEIR_space_charge_limit', 'PS_maxIntensity', 'PS_space_charge_limit', 
        			'SPS_maxIntensity', 'SPS_spaceChargeLimit', 'LHC_ionsPerBunch', 'LHC_chargesPerBunch']]
        for col in df3_SC_and_max_intensity.columns:
            df3_SC_and_max_intensity[col] = df3_SC_and_max_intensity[col].apply(lambda x: '{:.1e}'.format(x))
        df3_SC_and_max_intensity.to_csv('output/output_for_paper/{}_for_paper.csv'.format(output_3), index=True)
    
    
    ## 4: WITH NO SPLITTING AND LEIR-PS STRIPPING
    output_4 = '4_no_PS_splitting_and_LEIR_PS_stripping{}'.format(output_extra_str)
    injector_chain4 = InjectorChain(LEIR_bunches = 2,
                                    PS_splitting = 1,
                                    account_for_LEIR_ecooling=True,
                                    LEIR_PS_strip=True,
                                    PS_factor_SC=np.inf
                                    )
    df4 = injector_chain4.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name=output_4)
    
    if generate_tables_for_paper:
        df4_SC_and_max_intensity = df4[['LEIR_maxIntensity', 'LEIR_space_charge_limit', 'PS_maxIntensity', 'PS_space_charge_limit', 
        			'SPS_maxIntensity', 'SPS_spaceChargeLimit', 'LHC_ionsPerBunch', 'LHC_chargesPerBunch']]
        for col in df4_SC_and_max_intensity.columns:
            df4_SC_and_max_intensity[col] = df4_SC_and_max_intensity[col].apply(lambda x: '{:.1e}'.format(x))
        df4_SC_and_max_intensity.to_csv('output/output_for_paper/{}_for_paper.csv'.format(output_4), index=True)
    
    #### PLOT THE DATA #######
    SMALL_SIZE = 13.5
    MEDIUM_SIZE = 17
    BIGGER_SIZE = 23
    plt.rcParams["font.family"] = "serif"
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # Define bar width for bar plot
    bar_width = 0.35
    x = np.arange(len(df.index))
    
    fig, ax = plt.subplots(1, 1, figsize = (6,5))
    fig.suptitle("")
    bar2 = ax.bar(x - bar_width/2, ref_Table_SPS['WG5 Intensity']*df['atomicNumber'], bar_width, color='red', label='WG5') #
    bar1 = ax.bar(x + bar_width/2, df['LHC_chargesPerBunch'], bar_width, color='blue', label='Baseline scenario') #
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("LHC charges per bunch")
    ax.legend()
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if savefig:
        fig.savefig('output/figures/{}_ChargesPerBunch.png'.format(output_1), dpi=250)
    plt.close()
    
    # Baseline scenario
    fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
    bar22 = ax2.bar(x - bar_width/2, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width, color='red', label='WG5') #
    bar12 = ax2.bar(x + bar_width/2, df['LHC_ionsPerBunch']*df['massNumber'], bar_width, color='blue', label='Baseline scenario') #
    ax2.set_xticks(x)
    ax2.set_xticklabels(df.index)
    ax2.set_ylabel("Nucleons per bunch")
    ax2.legend()
    fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if savefig:
        fig2.savefig('output/figures/{}_NucleonsPerBunch.png'.format(output_1), dpi=250)
    plt.close()
    
    # No PS splitting 
    bar_width2 = 0.25
    fig3, ax3 = plt.subplots(1, 1, figsize = (6,5))
    bar31 = ax3.bar(x - bar_width2, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width2, color='red', label='WG5') #
    bar32 = ax3.bar(x, df['LHC_ionsPerBunch']*df['massNumber'], bar_width2, color='blue', label='Baseline scenario') #
    bar33 = ax3.bar(x + bar_width2, df2['LHC_ionsPerBunch']*df2['massNumber'], bar_width2, color='gold', label='No PS splitting') #
    ax3.set_xticks(x)
    ax3.set_xticklabels(df.index)
    ax3.set_ylabel("Nucleons per bunch")
    ax3.legend()
    fig3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if savefig:
        fig3.savefig('output/figures/{}.png'.format(output_2), dpi=250)
    plt.close()
    
    # Interpretation - Ca and Xe higher intensity due to higher LEIR charge state 
    # for In, LEIR is the limitation, i.e. can only inject intensities in the SPS that are below space charge limit
    
    # LEIR-PS stripping
    bar_width4 = 0.2
    fig4, ax4 = plt.subplots(1, 1, figsize = (6,5))
    bar41 = ax4.bar(x - 1.5*bar_width4, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width4, color='red', label='WG5') #
    bar42 = ax4.bar(x - 0.5*bar_width4, df['LHC_ionsPerBunch']*df['massNumber'], bar_width4, color='blue', label='Baseline scenario') #
    bar43 = ax4.bar(x + 0.5*bar_width4, df2['LHC_ionsPerBunch']*df2['massNumber'], bar_width4, color='gold', label='No PS splitting') #
    bar44 = ax4.bar(x + 1.5*bar_width4, df3['LHC_ionsPerBunch']*df3['massNumber'], bar_width4, color='limegreen', label='LEIR-PS stripping') #
    ax4.set_xticks(x)
    ax4.set_xticklabels(df.index)
    ax4.set_ylabel("Nucleons per bunch")
    ax4.legend()
    fig4.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if savefig:
        fig4.savefig('output/figures/{}.png'.format(output_3), dpi=250)
    plt.close()
    
    # LEIR-PS stripping and NO PS splitting
    bar_width5 = 0.15
    fig5, ax5 = plt.subplots(1, 1, figsize = (6,5))
    bar51 = ax5.bar(x, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width5, color='red', label='WG5') #
    bar52 = ax5.bar(x + bar_width5, df['LHC_ionsPerBunch']*df['massNumber'], bar_width5, color='blue', label='1: Baseline scenario') #
    bar53 = ax5.bar(x + 2*bar_width5, df2['LHC_ionsPerBunch']*df2['massNumber'], bar_width5, color='gold', label='2: No PS splitting') #
    bar54 = ax5.bar(x + 3*bar_width5, df3['LHC_ionsPerBunch']*df3['massNumber'], bar_width5, color='limegreen', label='3: LEIR-PS stripping') #
    bar55 = ax5.bar(x + 4*bar_width5, df4['LHC_ionsPerBunch']*df4['massNumber'], bar_width5, color='gray', label='4: LEIR-PS stripping, \nno PS splitting') #
    ax5.set_xticks(x + 2*bar_width5)
    ax5.set_xticklabels(df.index)
    ax5.set_ylabel("Nucleons per bunch")
    ax5.legend()
    fig5.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if savefig:
        fig5.savefig('output/figures/{}.png'.format(output_4), dpi=250)
    plt.close()
    print("Succesfully made all plots!")
    
    if return_dataframes:
        return df, df2, df3, df4


if __name__ == '__main__': 
    
    # First cases, following Roderik's methods: not considering space charge limits, 
    df, df2, df3, df4 = calculate_LHC_intensities_all_scenarios(
                                            output_extra_str = '',
                                            return_dataframes=True
                                            )
