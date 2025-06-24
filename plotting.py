# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:21:16 2025

@author: praktikant
"""

import re
import pandas as pd

# Step 1: Read the file
with open("RH40,1000\\model_summary.txt", "r") as file:
    lines = file.readlines()

# Step 2: Extract data using regex
data = []
pattern = re.compile(
    r"Model (?P<model>.*?) appears .*? = (?P<percent>\d+\.\d+)% and a mean MAPE of (?P<mape>\d+\.\d+)"
)

for line in lines:
    match = pattern.search(line)
    if match:
        model = match.group("model").strip()
        percent = float(match.group("percent"))
        mape = float(match.group("mape"))
        data.append({"Circuit Model": model, "Appearance %": percent, "Mean MAPE": mape})

# Step 3: Convert to DataFrame
df = pd.DataFrame(data)

# Step 4: Save to Excel
df.to_excel("model_summary.xlsx", index=False)
print("Saved to model_summary.xlsx")


