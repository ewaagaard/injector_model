# InjectorModel

### Background

As of today, the LHC ion physics programme is mostly based on Pb ion collisions. Lighter ions species have the potential to yield higher nucleon-nucleon luminosities, as requested from the ALICE3 detector proposal. This proposal was recently discussed in the [2022 Letter of Intent](https://arxiv.org/abs/2211.02491). To achieve the requested luminosity increase, the CERN Ion Injector chain (consisting of Linac3, LEIR, PS and SPS) will need to provide significantly higher beam intensities with light ion beams as compared to the Pb ion beams. So far the operational experience with light ion beams inthe injectors is very limited and the beam dynamics limitations for these beams are not well known. 

In this repository, we develop a simulation tool for different ions based on the Mathematica notebook presented by [Bruce (2021)](https://indico.cern.ch/event/1085343/contributions/4563386/attachments/2326159/3964426/2021.10.12--LIU_ions--Run4_light_ion_scenarios.pdf). The tool is contained in a class to represent the Injector Chain, taking an ion species as input and returning the calculated bunch intensity into the LHC. We compare the output to estimates from the [Working Group 5 (WG5) report](https://cds.cern.ch/record/2650176). 

### Usage 

- The Python class `CERN_Injector_Chain()` contained in `Injector_Chain.py` aims at modelling different ion species throughout the CERN accelerators. 
- The example script `Calculate_LHC_intensities.py` imports this class and generates both plots and table for different cases also discussed by Bruce (2021). 
- The directory `Data` contains input data for different ion in `Ion_species.csv` and stable isotopes for all considered ion species, generated from `stable_isotopes.py`
- **Input**: pandas dataframe with ion data, can be loaded from `Data/Ion_species.csv`. The intensity limit for a single ion can be calculated (specifying the ion type). The class is instantiated as in this example:
  ```python
  from Injector_Chain import CERN_Injector_Chain
  
  injector_chain = CERN_Injector_Chain(ion_type, 
                        ion_data, 
                        nPulsesLEIR = 0, # 0 = maximum number of pulses is calculated automatically
                        LEIR_bunches = 2,
                        PS_splitting = 1,
                        account_for_SPS_transmission=True,
                        LEIR_PS_strip=True
                        )
  ```
- **Output**: after instantiation, the method `CERN_Injector_Chain.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name='output1')` calculates the extracted bunch intensity from LEIR, PS and the SPS going into the LHC. This limit comes from either the total maximum intensity from the previous injector or from the space charge limit, for now calculated from the linear space charge tune shift corresponding to the maximum tune shift for present Pb beams. Also information about bunches, splitting, stripping and the relativistic $\gamma$ for each accelerator is calculated. If desired, this full csv table is saved in `Output`.   

### Example output plot for different scenarios

Various plots are generated in the example script `Calculate_LHC_intensities.py`. The last one, comparing different cases, is presented here:

![3_LEIR_PS_stripping|200](https://github.com/ewaagaard/InjectorModel/assets/68541324/cacad841-63fd-4aff-8b58-e5ed89b971b5)
