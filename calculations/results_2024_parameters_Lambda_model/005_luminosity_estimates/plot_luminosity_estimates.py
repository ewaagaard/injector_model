"""
Script to plot 1-month integrated luminosity (AA) and nucleon-nucleon (NN) integrated luminosity 
from "conservative" (baseline) and "optimistic" (optimized charge state + no PS split) scenario
from CTE output from Roderik Bruce 
--> 2025 update with LEIR electron cooling limit and values from the 2024 Pb ion run
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import injector_model
import os

os.makedirs('output', exist_ok=True)

index = ['O', 'Ar', 'Ca', 'Kr', 'In', 'Xe', 'Pb']
x = np.arange(len(index))
bar_width = 0.19

df_AA = pd.read_csv('ion_lumi_1month_AA_2025.csv', index_col=0)
df_NN = pd.read_csv('ion_lumi_1month_NN_2025.csv', index_col=0)

# Plot integrated luminosity
fig, ax = plt.subplots(1, 1, figsize = (6,5), constrained_layout=True)
#ax.grid(alpha=0.55)
ax.bar(x - 1.5 * bar_width, df_AA['WG5'], bar_width, color='red', label='WG5') #
ax.bar(x - 0.5 * bar_width, df_AA['2025 conservative'], bar_width, color='blue', label='Baseline') #
ax.bar(x + 0.5 * bar_width, df_AA['2025 optimistic'], bar_width, color='lime', label='Optimistic') #
ax.bar(x + 1.5 * bar_width, df_AA['2025 25 ns'], bar_width, color='darkorange', label='25 ns') #
ax.set_xticks(x)
ax.set_xticklabels(index)
ax.set_ylabel(r'$\int \mathcal{L}_{AA} dt$  [nb$^{-1}$]')
ax.legend()
ax.set_yscale('log')
fig.savefig('output/AA_integrated_luminosity.png', dpi=250)

# Plot integrated nucleon-nucleon luminosity
fig2, ax2 = plt.subplots(1, 1, figsize = (6,5), constrained_layout=True)
#ax2.grid(alpha=0.55)
ax2.bar(x - 1.5 * bar_width, df_NN['WG5'], bar_width, color='red', label='WG5') #
ax2.bar(x - 0.5 * bar_width, df_NN['2025 conservative'], bar_width, color='blue', label='Baseline') #
ax2.bar(x + 0.5 * bar_width, df_NN['2025 optimistic'], bar_width, color='lime', label='Optimistic') #
ax2.bar(x + 1.5 * bar_width, df_NN['2025 25 ns'], bar_width, color='darkorange', label='25 ns') #
ax2.set_xticks(x)
ax2.set_xticklabels(index)
ax2.set_ylabel(r'$\int \mathcal{L}_{NN} dt$  [pb$^{-1}$]')
ax2.legend()
ax2.set_yscale('log')
fig2.savefig('output/NN_integrated_luminosity.png', dpi=250)
plt.show()
