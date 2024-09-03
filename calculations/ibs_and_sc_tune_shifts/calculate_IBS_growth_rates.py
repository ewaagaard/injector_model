"""
Calculating growth rates with Nagaitsev formalism
- assume bunch intensity at the space charge limit
"""
import injector_model

inj = injector_model.InjectorChain()
df_IBS = inj.calculate_IBS_growth_rates_all_ion_species()

# Calculate relative growth rates - divide by Pb column
df_IBS_rel = df_IBS.div(df_IBS['Pb'], axis=0)
df_IBS_rel.to_csv('output_csv/IBS_growth_rates_relative.csv')