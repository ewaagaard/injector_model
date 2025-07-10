"""
Script to plot results from isotope scan in "scan_isotopes_for_lhc_intensities"
"""
import matplotlib.pyplot as plt
import pandas as pd 
import json
import numpy as np
from pathlib import Path
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import injector_model

# Which ion scenarios we consider - also whether to include electron cooling or not
account_for_LEIR_ecooling = True
ecool_str = '' if account_for_LEIR_ecooling else '_without_ecooling_limits'

# Load ion data and initialize for test for bunch intensities 
data_folder = Path(__file__).resolve().parent.joinpath('../../../data').absolute()
full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), header=0, index_col=0).T

# Stable isotope data
He_isotopes = np.array([3., 4.])
O_isotopes = np.array([16., 17., 18.])
Mg_isotopes = np.array([24., 25., 26.])
Ar_isotopes = np.array([36., 38., 40.])
Ca_isotopes = np.array([40., 42., 43., 44., 46., 48.])
Kr_isotopes = np.array([78., 80., 82., 83., 84., 86.])
In_isotopes = np.array([113., 115.])
Xe_isotopes = np.array([124., 126., 128., 129., 130., 131., 132, 134., 136])
Pb_isotopes = np.array([204., 206., 207., 208.])

all_isotopes = [He_isotopes, O_isotopes, Mg_isotopes, Ar_isotopes, Ca_isotopes, Kr_isotopes, In_isotopes, Xe_isotopes, Pb_isotopes]

# Create the combined figure with subplots
num_cols = 2  # Two columns
num_rows = (len(full_ion_data.T.index) + 1) // num_cols  # Integer division to determine the number of rows, Kr appears twice
fig0, axs = plt.subplots(num_rows, num_cols, figsize=(8.27, 10.2), constrained_layout=True)

def read_isotope_scan_results(ion_type, output_extra_str, count):
    
    # Define output strings to read correct file
    output_1 = '1_baseline{}'.format(output_extra_str)
    output_2 ='2_no_PS_splitting{}'.format(output_extra_str)
    output_3 = '3_LEIR_PS_stripping{}'.format(output_extra_str)
    output_4 = '4_no_PS_splitting_and_LEIR_PS_stripping{}'.format(output_extra_str)
    
    df1 = pd.read_csv("output/isotope_scan_results/{}_{}{}.csv".format(ion_type, output_1, ecool_str), header=1, index_col=0)
    df2 = pd.read_csv("output/isotope_scan_results/{}_{}{}.csv".format(ion_type, output_2, ecool_str), header=1, index_col=0)
    df3 = pd.read_csv("output/isotope_scan_results/{}_{}{}.csv".format(ion_type, output_3, ecool_str), header=1, index_col=0)
    df4 = pd.read_csv("output/isotope_scan_results/{}_{}{}.csv".format(ion_type, output_4, ecool_str), header=1, index_col=0)
    
    # Convert strings of charge states to numpy array
    #A_states = np.array(list(map(int, df1.columns.values)))
    A_states = np.array(list(map(int, df1.loc['massNumber'])))
    
    # Find default isotope and its performance in baseline scenario
    A_default = full_ion_data[ion_type]['A']
    ind_A_default = np.where(A_states == A_default)[0][0]
    Nb0 = float(df1.loc['LHC_ionsPerBunch'].iloc[ind_A_default])

    #### PLOTTING - performance of all charge states ####
    fig, ax = plt.subplots(1, 1, figsize = (6,5), constrained_layout=True)
    ax.plot(A_default, Nb0, 'ro', markersize=13, alpha=0.8, label='1: Baseline with\ndefault charge state')
    ax.plot(A_states, np.array(list(map(float, df1.loc['LHC_ionsPerBunch'].values))), marker='o', color='blue', linewidth=4.2, linestyle='-', label='1: Baseline')
    ax.plot(A_states, np.array(list(map(float, df2.loc['LHC_ionsPerBunch'].values))), marker='o', linestyle='--', color='gold', linewidth=3.8, label='2: No PS splitting') #
    ax.plot(A_states, np.array(list(map(float, df3.loc['LHC_ionsPerBunch'].values))), marker='o', linestyle='-.', color='limegreen', linewidth=3.5, label='3: LEIR-PS stripping') #
    ax.plot(A_states, np.array(list(map(float, df4.loc['LHC_ionsPerBunch'].values))), marker='o', linestyle='--', color='gray', linewidth=3, label='4: LEIR-PS stripping, \nno PS splitting') #
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.set_ylabel('LHC bunch intensity')
    if ion_type == 'Xe':
        ax.tick_params(axis='x', which='major', labelsize=12)
    ax.set_xlabel('Mass number A')
    ax.legend(fontsize=12)
    #fig.tight_layout()
    fig.savefig('output/figures/{}{}_isotope_state_scan{}.png'.format(ion_type, output_extra_str, ecool_str), dpi=250)
    plt.close()

    #### PLOTTING - performance of all charge states ####
    fig2, ax2 = plt.subplots(1, 1, figsize = (6,5), constrained_layout=True)
    ax2.plot(A_default, Nb0*A_default, 'ro', markersize=13, alpha=0.8, label='1: Baseline with\ndefault charge state')
    ax2.plot(A_states, np.array(list(map(float, df1.loc['LHC_ionsPerBunch'].values))) * np.array(list(map(float, df1.loc['massNumber'].values))), marker='o', color='blue', linewidth=4.2, linestyle='-', label='1: Baseline')
    ax2.plot(A_states, np.array(list(map(float, df2.loc['LHC_ionsPerBunch'].values))) * np.array(list(map(float, df2.loc['massNumber'].values))), marker='o', linestyle='--', color='gold', linewidth=3.8, label='2: No PS splitting') #
    ax2.plot(A_states, np.array(list(map(float, df3.loc['LHC_ionsPerBunch'].values))) * np.array(list(map(float, df3.loc['massNumber'].values))), marker='o', linestyle='-.', color='limegreen', linewidth=3.5, label='3: LEIR-PS stripping') #
    ax2.plot(A_states, np.array(list(map(float, df4.loc['LHC_ionsPerBunch'].values))) * np.array(list(map(float, df1.loc['massNumber'].values))), marker='o', linestyle='--', color='gray', linewidth=3, label='4: LEIR-PS stripping, \nno PS splitting') #
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax2.set_ylabel('LHC nucleons per bunch')
    if ion_type == 'Xe':
        ax2.tick_params(axis='x', which='major', labelsize=12)
    ax2.set_xlabel('Mass number A')
    ax2.legend(fontsize=12)
    #fig.tight_layout()
    fig2.savefig('output/figures/{}{}_isotope_state_scan_nucleons_{}.png'.format(ion_type, output_extra_str, ecool_str), dpi=250)
    plt.close()
    
    # Also fill in the combined superplot
    row3 = (count-1) // num_cols  # Row index
    col3 = (count-1) % num_cols  # Column index
    ax3 = axs[row3, col3]  # Select the current subplot
    ax3.plot(A_default, Nb0 * A_default, 'ro', markersize=13, alpha=0.8, label='1: Baseline with\ndefault charge state')
    ax3.plot(A_states, np.array(list(map(float, df1.loc['LHC_ionsPerBunch'].values))) * np.array(list(map(float, df1.loc['massNumber'].values))), marker='o', color='blue', linewidth=4.2, linestyle='-', label='1: Baseline')
    ax3.plot(A_states, np.array(list(map(float, df2.loc['LHC_ionsPerBunch'].values))) * np.array(list(map(float, df2.loc['massNumber'].values))), marker='o', linestyle='--', color='gold', linewidth=3.8, label='2: No PS splitting') #
    ax3.plot(A_states, np.array(list(map(float, df3.loc['LHC_ionsPerBunch'].values))) * np.array(list(map(float, df3.loc['massNumber'].values))), marker='o', linestyle='-.', color='limegreen', linewidth=3.5, label='3: LEIR-PS stripping') #
    ax3.plot(A_states, np.array(list(map(float, df4.loc['LHC_ionsPerBunch'].values))) * np.array(list(map(float, df4.loc['massNumber'].values))), marker='o', linestyle='--', color='gray', linewidth=3, label='4: LEIR-PS stripping, \nno PS splitting') #
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.grid(alpha=0.45)
    
    # Determine where to place ion label
    if count == 1:
        vtext_loc = 0.81
    elif count>1 and count<4:
        vtext_loc = 0.5
    elif count == 4:
        vtext_loc = 0.64
    elif count>4 and count<7:
        vtext_loc = 0.48
    elif count == 7:
        vtext_loc = 0.42
    elif count == 8:
        vtext_loc = 0.6
    else:
        vtext_loc = 0.41
    
    ax3.text(0.023, vtext_loc, '{}'.format(ion_type), fontsize=18, weight='bold', transform=ax3.transAxes)
    #ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    
    # Add legend in oxygen plot
    if count == 3:
        ax3.legend(fontsize=8, loc='upper right')
    
    # Find best isotope, compare to baseline case
    print('\nIon type: {}'.format(ion_type))
    print('Default A: {}'.format(A_default))
    print('Scenario 2: no bunch splitting')
    Nb0_case2 = float(df2.loc['LHC_ionsPerBunch'].iloc[ind_A_default])
    A0_case2 = float(df2.loc['massNumber'].iloc[ind_A_default])
    Nb_case2_relative = np.array(list(map(float, df2.loc['LHC_ionsPerBunch'].values))) / Nb0_case2
    nucleons_case2_relative = (np.array(list(map(float, df2.loc['LHC_ionsPerBunch'].values))) * np.array(list(map(float, df2.loc['massNumber'].values))) ) / (Nb0_case2 * A0_case2)
    ind_A_max = np.argmax(Nb_case2_relative)
    ind_nucleon_max = np.argmax(nucleons_case2_relative)
    print('Best A = {}, with {:.2f} percent of outgoing Nb\n'.format(A_states[ind_A_max], 100 * Nb_case2_relative[ind_A_max]))
    print('A max index: {}, nucleon max index: {}'.format(ind_A_max, ind_nucleon_max))
    
    return A_states[ind_A_max], float(Nb_case2_relative[ind_A_max]), float(nucleons_case2_relative[ind_A_max])
    

# Make empty dictionary 
isotope_dict = {'Ion' : [],
                'Best A': [],
                'Scenario2_Nb0_improvement_factor': [],
                'Scenario2_nucleon_improvement_factor': []
                }

# Keep track of subplot position
count = 1  

# Scan over ions and load results
for ion_type in full_ion_data.columns:

    # Plot results
    best_A, improvement_Nb_case2_rel, improvement_nucleon_case2_rel = read_isotope_scan_results(ion_type, '', count)
    
    isotope_dict['Ion'].append(ion_type)
    isotope_dict['Best A'].append(best_A)
    isotope_dict['Scenario2_Nb0_improvement_factor'].append(improvement_Nb_case2_rel)
    isotope_dict['Scenario2_nucleon_improvement_factor'].append(improvement_nucleon_case2_rel)
    
    count += 1
    
    
# Fix superplot
# Share y-axes for the same row and share x-label for the same column
#for col in range(num_cols):
axs[-1, 0].set_xlabel('Mass number A', fontsize=13)
axs[-2, 1].set_xlabel('Mass number A', fontsize=13)
axs[-1, -1].axis('off')

# Share y-label for the same row
for row in axs:
    row[0].set_ylabel('$\mathcal{N}_{b}$ into LHC', fontsize=13)

#### PLOT SETTINGS #######
fig0.savefig('output/figures/Combined_isotope_scan.png', dpi=250)
plt.show()
        

# Flatten array
isotope_dict['Scenario2_Nb0_improvement_factor'] = np.array(isotope_dict['Scenario2_Nb0_improvement_factor']).tolist()

# save dictionary
with open("output/isotope_scan.json", "w") as fp:
    json.dump(isotope_dict , fp, default=str)     
print('Created dictionary:\n{}'.format(isotope_dict))



