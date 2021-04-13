#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:28:07 2021

@author: hanbo
"""
import pandas as pd

my_path = "data/sample201320.csv"
df = pd.read_csv(my_path)
df.info()
my_cols = df.columns
for col in my_cols:
    print(col)