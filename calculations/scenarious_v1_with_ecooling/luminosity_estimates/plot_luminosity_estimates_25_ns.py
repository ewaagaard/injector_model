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
import os

os.makedirs('Figures', exist_ok=True)

index = ['O', 'Ar', 'Ca', 'Kr', 'In', 'Xe', 'Pb']
x = np.arange(len(index))
bar_width = 0.25

df_NN = pd.read_csv('ion_lumi_1month_NN_2024_25_ns_conservative.csv', index_col=0)
df_NN['Relative increase'] = df_NN['2024 conservative 25 ns'] / df_NN['2024 conservative old Nb without ecooling limit']

print(df_NN)

# Plot integrated nucleon-nucleon luminosity for conservative - 50 ns vs 25 ns
fig2, ax2 = plt.subplots(1, 1, figsize = (6,5), constrained_layout=True)
ax2.bar(x - bar_width, df_NN['WG5'], bar_width, color='red', label='WG5') #
ax2.bar(x, df_NN['2024 conservative old Nb without ecooling limit'], bar_width, color='blue', label='Conservative 50 ns') #
ax2.bar(x + bar_width, df_NN['2024 conservative 25 ns'], bar_width, color='orange', label='Conservative 25 ns') #
ax2.set_xticks(x)
ax2.set_xticklabels(index)
ax2.set_ylabel(r'$\int \mathcal{L}_{NN} dt$  [pb$^{-1}$]')
ax2.legend(fontsize=13.5)
ax2.set_yscale('log')
fig2.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
fig2.savefig('Figures/NN_integrated_luminosity_25_ns.png', dpi=250)
plt.show()