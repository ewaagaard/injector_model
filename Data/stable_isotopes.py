#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data from Mathematica on stable isotope among considered ions 
"""
import pandas as pd

# The isotope data 
data = [
    ["Ion", "Z", "Standard isotope", "Isotope", "Natural abundance"],
    ["O", 8, 16, [
        [16, "", 0.99757],
        [17, "", 0.00038],
        [18, "", 0.00205]
    ]],
    ["Ar", 18, 40, [
        [36, "", 0.003365],
        [38, "", 0.000632],
        [40, "", 0.996003]
    ]],
    ["In", 49, 115, [
        [113, "", 0.0429],
        [115, "", 0.9571]
    ]],
    ["He", 2, 4, [
        [3, "", 1.37e-6],
        [4, "", 0.99999863]
    ]],
    ["Ca", 20, 40, [
        [40, "", 0.96941],
        [42, "", 0.00647],
        [43, "", 0.00135],
        [44, "", 0.02086],
        [46, "", 0.00004],
        [48, "", 0.00187]
    ]],
    ["Kr", 36, 84, [
        [78, "", 0.0035],
        [80, "", 0.0228],
        [82, "", 0.1158],
        [83, "", 0.1149],
        [84, "", 0.5700],
        [86, "", 0.1730]
    ]],
    ["Xe", 54, 129, [
        [124, "", 0.0009],
        [126, "", 0.0009],
        [128, "", 0.0192],
        [129, "", 0.2644],
        [130, "", 0.0408],
        [131, "", 0.2118],
        [132, "", 0.2689],
        [134, "", 0.1044],
        [136, "", 0.0887]
    ]],
    ["Pb", 82, 208, [
        [204, "", 0.014],
        [206, "", 0.241],
        [207, "", 0.221],
        [208, "", 0.524]
    ]]
]


# Pandas dataframe 
df = pd.DataFrame(columns=data[0])

for row in data[1:]:
    ion, Z, A, isotopes = row
    for isotope in isotopes:
        df = df.append({
            "Ion": ion,
            "Z": Z,
            "Standard isotope": A,
            "Isotope": isotope[0],
            "Natural abundance": isotope[2]
        }, ignore_index=True)
        
df.to_csv("Stable_isotopes.csv", index=False)