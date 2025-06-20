# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:16:27 2025

@author: praktikant
"""

import pickle

# Load the dictionaries from the .pkl file
with open("analytics.pkl", "rb") as f:
    data = pickle.load(f)

# Access the individual dictionaries
circuit_dict = data["circuit_dict"]
DRT_dict = data["DRT_dict"]
invalid_model = data["invalid_model"]

# Optional: print to verify
print("Circuit Dictionary:\n", circuit_dict)
print("\nDRT Dictionary:\n", DRT_dict)
print("\nInvalid Models:\n", invalid_model)