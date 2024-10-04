"""
Script to plot 1-month integrated luminosity (AA) and nucleon-nucleon (NN) integrated luminosity 
from "conservative" (baseline) and "optimistic" (optimized charge state + no PS split) scenario
from CTE output from Roderik Bruce 
--> 2024 update with LEIR electron cooling limit
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import injector_model


index = ['O', 'Ar', 'Ca', 'Kr', 'In', 'Xe', 'Pb']
x = np.arange(len(index))
bar_width = 0.25

df_AA = pd.read_csv('ion_lumi_1month_AA_2024.csv', index_col=0)
df_NN = pd.read_csv('ion_lumi_1month_NN_2024.csv', index_col=0)

# Plot integrated luminosity
fig, ax = plt.subplots(1, 1, figsize = (6,5))
ax.bar(x - bar_width, df_AA['WG5'], bar_width, color='red', label='WG5') #
ax.bar(x, df_AA['2024 conservative'], bar_width, color='blue', label='Conservative') #
ax.bar(x + bar_width, df_AA['2024 optimistic'], bar_width, color='lime', label='Optimistic') #
ax.set_xticks(x)
ax.set_xticklabels(index)
ax.set_ylabel(r'$\int \mathcal{L}_{AA} dt$  [nb$^{-1}$]')
ax.legend()
ax.set_yscale('log')
fig.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('AA_integrated_luminosity.png', dpi=250)

# Plot integrated nucleon-nucleon luminosity
fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
ax2.bar(x - bar_width, df_NN['WG5'], bar_width, color='red', label='WG5') #
ax2.bar(x, df_NN['2024 conservative'], bar_width, color='blue', label='Conservative') #
ax2.bar(x + bar_width, df_NN['2024 optimistic'], bar_width, color='lime', label='Optimistic') #
ax2.set_xticks(x)
ax2.set_xticklabels(index)
ax2.set_ylabel(r'$\int \mathcal{L}_{NN} dt$  [pb$^{-1}$]')
ax2.legend()
ax2.set_yscale('log')
fig2.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
fig2.savefig('NN_integrated_luminosity.png', dpi=250)
plt.show()