#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input parameter table for the Injector Chain ions
"""
import pandas as pd
ion_data = pd.read_csv("Data/Ion_species.csv", sep=';', header=0, index_col=0).T
