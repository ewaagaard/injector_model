"""
Script to plot peak luminosity and event rates for conservative, optimistic and 25 ns scenario
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

# Read dataframes, then stack them into one large df
df_conservative = pd.read_csv('peak_rates_conservative.csv', index_col=0)
df_optimistic = pd.read_csv('peak_rates_optimistic.csv', index_col=0)
df_25_ns = pd.read_csv('peak_rates_25ns.csv', index_col=0)
df_combo = pd.concat([df_conservative, df_optimistic, df_25_ns], axis=1)
df_combo.to_csv('output/peak_rates_all_2025.csv', index=True)
