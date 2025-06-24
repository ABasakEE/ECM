# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:35:32 2025

@author: Biswadeep
"""
import pickle
import matplotlib.pyplot as plt

file = f"RH40,1000\\analytics.pkl"

data = None
with open(file, "rb") as f:
    data = pickle.load(f)

# Access the individual dictionaries
circuit_dict = data["circuit_dict"] 
DRT_dict = data["DRT_dict"]
invalid_model = data["invalid_model"]

cell = 9
cycle_number = 0
key1 = f"Cell_{cell:02}"
key2 = f"AST {cycle_number}cycles"

key = (key1,key2)

R_inf_DRT, L_0_DRT, gamma_DRT, tau_vec = DRT_dict[key]

fig = plt.gcf()
plt.semilogx(tau_vec, gamma_DRT, linewidth=4, color='black', label='DRT') 
plt.axis([1E-6, 1E2, 0, 130])
plt.legend(frameon=False, fontsize = 15, loc='upper left')
plt.xlabel(r'$\tau/\rm s$', fontsize = 20)
plt.ylabel(r'$\gamma/\Omega$', fontsize = 20)
fig.set_size_inches(6.472, 4)

plt.show()