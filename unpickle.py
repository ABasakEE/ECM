# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:16:27 2025

@author: praktikant
"""
import os

candidate_models = ['L0-R0-p(R1,C1)','L0-R0-p(R1,CPE1)',
                    'L0-R0-p(R1,C1-W1)','L0-R0-p(R1,C1-G1)']

def generateLists(folder):

    AST_labels = []

    for f in os.listdir(folder):
        AST_labels.append(str(f))            
    return AST_labels

import pickle

# Load the dictionaries from the .pkl file
data = None
with open("RH100,1000\\analytics.pkl", "rb") as f:
    data = pickle.load(f)

# Access the individual dictionaries
circuit_dict = data["circuit_dict"] 
DRT_dict = data["DRT_dict"]
invalid_model = data["invalid_model"]


start = 1
end = 42

threshold = 5 #5% allowed error for single Randles circuit
single_ECM = []
unique_cells = []

for i in range(start,end+1):
    f = f"Cell_{i:02}"  # Pads i to two digits, e.g., "01", "12", etc.
    #folder = f"C:\\Users\\praktikant\\Desktop\\Dataset\\01-Data\\{f}\\04-EIS_H2Air_RH100\\1000mAcm2"
    
    folder = f"D:\\Dataset\\01-Data\\{f}\\04-EIS_H2Air_RH100\\1000mAcm2"
    if not os.path.exists(folder):
        continue
    
    AST_labels = generateLists(folder)
    # print('AST: ',AST_labels)
    for j in AST_labels:
        for k in candidate_models:
            key1 = f
            key2 = f'AST {j}'
            key3 = k
            key = (key1,key2,key3)
            if key in circuit_dict and circuit_dict[key][-1] < threshold:
                single_ECM.append(key)
                if key1 not in unique_cells:
                    unique_cells.append(key1)
                
            
            
            
	

# Optional: print to verify
# print("Circuit Dictionary:\n", circuit_dict)
# print("\nDRT Dictionary:\n", DRT_dict)
# print("\nInvalid Models:\n", invalid_model)

print('Single peak ECMs: ',single_ECM)
print('Cells involved: ',unique_cells)