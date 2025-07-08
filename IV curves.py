# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 10:31:23 2025

@author: Biswadeep
"""
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import os 
import re



def generateLists(folder):

    AST_labels = []
    for f in os.listdir(folder):
        AST_labels.append(str(f))            
    return AST_labels


start = 1
end  = 42

required_currents = [0.1, 0.5, 1] #we need to extract the voltages at these currents

results = []


for current in required_currents:

    for i in range(start,end+1):
        f = f"Cell_{i:02}"  # Pads i to two digits
        folder = f"D:\\Dataset\\01-Data\\{f}\\03-IV-curve_RH100"
        AST_labels = generateLists(folder)
        
        for label in AST_labels:
            path = f"D:\\Dataset\\01-Data\\{f}\\03-IV-curve_RH100\\{label}\\IV_data.csv"
            #extract the IV curve values at RH = 100%
    
            df = pd.read_csv(path,sep='\t')  
    
            current_density = df['I.C[A/cm2]'].values
            voltage = df['U.C[V]'].values
            #extract current density and voltage values        
    
            sorted_indices = np.argsort(current_density)
            current_density = current_density[sorted_indices]
            voltage = voltage[sorted_indices]
    
            # Fit a univariate spline (you can adjust 's' for smoothness)
            spline = UnivariateSpline(current_density, voltage, s=0)
            
            smooth_current = np.linspace(min(current_density), max(current_density), 500)
            smooth_voltage = spline(smooth_current)
            
            required_voltage = spline(current)
            cycle_number = re.findall(r'\d+',label)[0]
            results.append({
                "Cell": f,
                "Cycle": cycle_number,
                "Current": current ,
                "Voltage":required_voltage})


df = pd.DataFrame(results)
path = 'IV.xlsx'
df.to_excel(path)
print("Generated Excel file")