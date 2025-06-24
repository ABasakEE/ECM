# -*- coding: utf-8 -*-

import os
import pandas as pd


import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


#libraries required to generate the DRT plot for a given fuel cell
import pyDRTtools
import pyDRTtools.basics as basics # pyDRTtools functions
import pyDRTtools.GUI as UI

import importlib
from numpy import loadtxt
from matplotlib import gridspec # for the contour plots
from cvxopt import matrix, solvers

#library used to perform circuit fitting
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist


ckt_init = 'L0-R0-' #these two elements must exist

candidate_models = ['p(R1,C1)-p(R2,C2)',
                    'p(R1,C1)-p(R2,CPE2)',
                    'p(R1,CPE1)-p(R2,C2)',
                    'p(R1,CPE1)-p(R2,CPE2)',
                    'p(R1,C1)-p(R2,C2-W2)',                   
                    'p(R1,C1-W1)-p(R2,C2-W2)',
                    'p(R1,C1)-p(R2,C2-G2)',                   
                    'p(R1,C1-G1)-p(R2,C2-G2)',
                    'p(R1,C1)','p(R1,CPE1)',
                    'p(R1,C1-W1)','p(R1,C1-G1)']

single_circuit = []

#define the possible circuit models that we must iterate through


#generate the nyquist plot given the dataframe containing EIS data
def plot_Nyquist(dfs, labels=None):
    plt.figure(figsize=(6, 6))
    for i, df in enumerate(dfs):
        Z = df['Z.Re.C[mOhm*cm2]'].values + 1j * df['Z.Im.C[mOhm*cm2]'].values
        label = labels[i] if labels else f'Cycle {i+1}'
        plt.plot(df['Z.Re.C[mOhm*cm2]'], -df['Z.Im.C[mOhm*cm2]'], label=label)
    plt.xlabel("Z' (mΩ·cm²)")
    plt.ylabel("-Z'' (mΩ·cm²)")
    plt.title('Nyquist Plot')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#generate a Bode plot, given the input EIS data
def plot_Bode(dfs, labels=None):
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for i, df in enumerate(dfs):
        freq = df['f.C[Hz]'].values
        Z = df['Z.Re.C[mOhm*cm2]'].values + 1j * df['Z.Im.C[mOhm*cm2]'].values
        mag = np.abs(Z)
        phase = np.angle(Z, deg=True)
        label = labels[i] if labels else f'Cycle {i+1}'
        ax_mag.loglog(freq, mag, label=label)
        ax_phase.semilogx(freq, phase, label=label)
    ax_mag.set_ylabel('|Z| (mΩ·cm²)')
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.set_ylabel('Phase (°)')
    ax_mag.set_title('Bode Plot - Magnitude')
    ax_phase.set_title('Bode Plot - Phase')
    ax_mag.grid(True, which='both')
    ax_phase.grid(True, which='both')
    ax_mag.legend()
    ax_phase.legend()
    plt.tight_layout()
    plt.show()


def generateLists(folder):

    df_list = []
    AST_labels = []

    for f in os.listdir(folder):
        AST_labels.append(str(f))
        for file in os.listdir(os.path.join(folder,f)):
            file_raw = os.path.join(folder,f,file)
            temp_df = pd.read_csv(file_raw, sep='\t')
            temp_df['Z'] = temp_df['Z.Re.C[mOhm*cm2]'] + 1j * temp_df['Z.Im.C[mOhm*cm2]']
            df_list.append(temp_df)
            
    return AST_labels, df_list


def generateDRT(N_freqs, frequencies, Z, Z_real, Z_imag, Z_mag): #subroutine to generate the DRT plot for a given fuel cell

    freqs = np.flip(frequencies)
    Z_complex = np.flip(Z)
    
    
    #define the max number of relaxation times that you want to solve
    N_taus = 81
    log_tau_min, log_tau_max = -6, 2
    tau_vec = np.logspace(log_tau_min, log_tau_max, N_taus) 
    
    #generate the DRT kernel using Gaussian distribution
    from pyDRTtools.basics import compute_epsilon, assemble_A_re, assemble_A_im, assemble_M_2,  quad_format_combined
    
    coeff = 0.5
    rbf_type = 'Gaussian'
    shape_control = 'FWHM Coefficient'
    
    epsilon = compute_epsilon(freqs, coeff, rbf_type, shape_control)
    
    # Real‐part kernel
    A_re = assemble_A_re(freqs, tau_vec, epsilon , rbf_type)
    # prepend columns for R_inf and L0
    A_re_R_inf = np.ones((N_freqs, 1))
    A_re_L_0 = np.zeros((N_freqs, 1))
    A_re = np.hstack(( A_re_R_inf, A_re_L_0, A_re))
    
    # differentiation matrix for the imaginary part of the impedance
    A_im = assemble_A_im(freqs, tau_vec, epsilon, rbf_type)
    
    #assemble_A_im(freq_vec, tau_vec, epsilon, 'Piecewise Linear', flag1='simple', flag2='impedance')
    
    A_im_R_inf = np.zeros((N_freqs, 1))
    A_im_L_0 = 2*np.pi*freqs.reshape((N_freqs, 1))
    A_im = np.hstack(( A_im_R_inf, A_im_L_0, A_im))
    
    # complete discretization matrix
    A = np.vstack((A_re, A_im))
    
    # second-order differentiation matrix for ridge regression (RR)
    M2 = np.zeros((N_taus+2, N_taus+2))
    M2[2:,2:] = basics.assemble_M_2(tau_vec, epsilon, rbf_type)
    from pyDRTtools.basics import optimal_lambda
    
    Z_re = Z_complex.real
    Z_im = Z_complex.imag
    
    data_used   = np.arange(N_freqs)   # use every point in your spectrum
    induct_used = 1        
    
    lambda_opt = optimal_lambda(A_re, A_im, Z_re, Z_im, M2, data_used, induct_used, -3, 'GCV')
    
    from cvxopt import matrix, solvers
    
    lb = np.zeros([N_taus+2])
    bound_mat = np.eye(lb.shape[0])
    
    H_combined, c_combined = quad_format_combined(A_re, A_im, Z_re, Z_im, M2, lambda_opt)
    
    # set bound constraint
    # G = matrix(-np.identity(Z_complex.imag.shape[0]+2))
    # h = matrix(np.zeros(Z_complex.imag.shape[0]+2))
    
    n_vars = N_taus + 2
    G = matrix(-np.identity(n_vars))
    h = matrix(np.zeros(n_vars))
    
    sol = solvers.qp(matrix(H_combined), matrix(c_combined),G,h)
    ## deconvolved DRT
    x = np.array(sol['x']).flatten()
    
    R_inf_DRT, L_0_DRT = x[0:2]
    gamma_DRT = x[2:]
    
    
    #display the obtained DRT graph
    # fig = plt.gcf()
    # plt.semilogx(tau_vec, gamma_DRT, linewidth=4, color='black', label='DRT') 
    # plt.axis([1E-6, 1E2, 0, 130])
    # plt.legend(frameon=False, fontsize = 15, loc='upper left')
    # plt.xlabel(r'$\tau/\rm s$', fontsize = 20)
    # plt.ylabel(r'$\gamma/\Omega$', fontsize = 20)
    # fig.set_size_inches(6.472, 4)
    
    # plt.show()
    
    return R_inf_DRT, L_0_DRT, gamma_DRT, tau_vec
    




# #based on the Nyquist plots obtained till now, we must have a L0 and R0 circuit element
#for each model we perform splitting as per the DRT peaks obtained till now

def initial_guess (model, R_inf_DRT, L_0_DRT, gamma_DRT, tau_vec, n_CPE = 0.85):
    
    peaks, properties = find_peaks(gamma_DRT, height=0.1)  # Adjust `height` threshold if needed
    #(peak will be found if it is 10% of the maximum height in the graph)
    peak_heights = properties["peak_heights"]
    
    R0_guess = R_inf_DRT
    L0_guess = L_0_DRT
    
    init_guess = [L0_guess,R0_guess]
    
    selected_heights = None
    selected_taus = None
    R1, R2 = None, None
    C1, C2 = None, None
    CPE1, CPE2 = 0.85, 0.9
    tau1, tau2 =  None, None
    
    #select the 2 highest peaks
    
    top_indices = np.argsort(peak_heights)[-2:][::-1]  # sort by descending height
    selected_peaks = peaks[top_indices]
    selected_heights = peak_heights[top_indices]
    selected_taus = tau_vec[selected_peaks]
    
    ##### some circuits have only 1 DRT peak
    diameter = Z_real.max() - R0_guess
    
    weights = selected_heights / np.sum(selected_heights)
    R_vals = diameter * weights
    
    check = False
    
    if len(R_vals) < 2: #two peaks not detected
        R1 = R_vals[-1]
        tau1 = selected_taus[-1]
        C1 = tau1 / R1
        Q1 = 1 / (R1 * tau1 ** n_CPE)
        omega1 = 1 / tau1
        W1 = R1 * np.sqrt(omega1)
        check = True
    else:
        R1, R2 = R_vals[:2]
        tau1, tau2 = selected_taus[:2]
        
        C1 = tau1 / R1
        Q2 = 1 / (R2 * tau2 ** n_CPE)
        
        Q1 = 1 / (R1 * tau1 ** n_CPE)
        C2 = tau2 / R2
        
        #generate the diffusion models for this split
        omega1 = 1 / tau1
        omega2 = 1/ tau2
        W1 = R1 * np.sqrt(omega1)
        W2 = R2 * np.sqrt(omega2)
        
        
    # print('Printing stuff')
    # print(R1, R2, C1, C2, CPE1, CPE2, tau1, tau2)
    
    diffusion_guess = []  
  
    if check:
        if model[6:] == 'p(R1,C1-W1)':
            diffusion_guess.extend([R1,C1,W1])
        elif model[6:] == 'p(R1,C1-G1)':
            diffusion_guess.extend([R1,C1,R1,tau1])
    else:
        
        if model[6:] == 'p(R1,C1)-p(R2,C2-W2)':
            diffusion_guess.extend([R1,C1,R2,C2,W2])
        elif model[6:] == 'p(R1,C1-W1)-p(R2,C2-W2)':
            diffusion_guess.extend([R1,C1,W1,R2,C2,W2])
        elif model[6:] == 'p(R1,C1)-p(R2,C2-G2)':
            diffusion_guess.extend([R1,C1,R2,C2,R2,tau2])
        elif model[6:] == 'p(R1,C1-G1)-p(R2,C2-G2)':
            diffusion_guess.extend([R1,C1,R1,tau1,R2,C2,R2,tau2])
    
    
    init_guess = [L0_guess,R0_guess]
    
    if check:
        if model[6:] == 'p(R1,C1)':
            init_guess.extend([R1,C1])  
        elif model[6:] == 'p(R1,CPE1)':
            init_guess.extend([R1,Q1,CPE1])
    else:
            
        if model[6:] == 'p(R1,C1)-p(R2,C2)':
            init_guess.extend([R1,C1,R2,C2])    
        elif model[6:] == 'p(R1,CPE1)-p(R2,C2)':
            init_guess.extend([R1,Q1,CPE1,R2,C2])
        elif model[6:] == 'p(R1,C1)-p(R2,CPE2)':
            init_guess.extend([R1,C1,R2,Q2,CPE2])
        elif model[6:] == 'p(R1,CPE1)-p(R2,CPE2)':
            init_guess.extend([R1,Q1,CPE1,R2,Q2,CPE2])
    
    if 'R' in model or 'G' in model:
        init_guess.extend(diffusion_guess)
   
    return init_guess, R_vals


def fitted_model(circuit):
    # print("\nFitted circuit parameters:")
    # print(circuit)

    Z_fit = circuit.predict(frequencies)
    residuals = np.array(Z) - Z_fit

    residual_mag = np.abs(residuals)

    

    # fig, ax = plt.subplots()
    # plot_nyquist(Z, fmt='o', scale=10, ax=ax)
    # plot_nyquist(Z_fit, fmt='-', scale=10, ax=ax)
    
    # plt.title(circuit.circuit)
    # plt.legend(['Data', 'Fit'])
    # plt.show()


    # plt.figure()
    # plt.plot(residuals.real, residuals.imag, 'o')
    # plt.xlabel('Residual Re(Z)')
    # plt.ylabel('Residual Im(Z)')
    # plt.title('Complex Residual Plot')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()

    # plt.figure()
    # plt.semilogx(frequencies, residual_mag, 'r.')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('|Residual| (mΩ·cm²)')
    # plt.title('Residual Magnitude vs Frequency')
    # plt.grid(True, which='both')
    # plt.tight_layout()
    # plt.show()
    
    return residuals, Z_fit

def generate_guesses(index, R_inf_DRT, L_0_DRT, gamma_DRT, tau_vec):
    model = ckt_init+candidate_models[index]
    init_vector, R_vals = initial_guess(model, R_inf_DRT, L_0_DRT, gamma_DRT, tau_vec)

    print('Circuit:',model)
    
    if len(R_vals) < 2 and index <= 7 or len(R_vals) >1 and index >7: #do not attempt to fit the circuit
        return 'single', None, None

    circuit = CustomCircuit(model, initial_guess=init_vector)
    try:
        circuit.fit(frequencies, Z, maxfev=25000) 
    except:
        return None, None, None
    
    residuals, Z_fit = fitted_model(circuit)
    
    return circuit, residuals, Z_fit


#generate the folder paths to extract the EIS data from multiple cells
DRT_dict = dict() #store the computed DRT values which will be used again and again
circuit_dict = dict()
single_ckt=[]

AST_fit = dict()
start = 1
end = 42

threshold = 3 #3% threshold as a preliminary elimination method 

skipped_cells = []
for k in range(start,end+1):
    # if k == 6:
    #     continue #missing data for cell 6
    
    f = f"Cell_{k:02}"  # Pads i to two digits, e.g., "01", "12", etc.
    folder = f"C:\\Users\\praktikant\\Desktop\\Dataset\\01-Data\\{f}\\09-EIS_H2Air_RH40\\100mAcm2"
    # folder = f"D:\\Dataset\\01-Data\\{f}\\07-EIS_H2Air_RH70\\500mAcm2"
    #obtain the best ECM model at this RH and current density
    if not os.path.exists(folder):
        skipped_cells.append(k)
        print(f'Skipping cell {k} due to lack of data')
        continue
        
    #step 1: extract the data as Pandas dataframe
    AST_labels, df_list = generateLists(folder)
    
    #generate the Nyquist and Bode plots for the different cycles
    #plot_Nyquist(df_list, AST_labels)
    #plot_Bode(df_list, AST_labels)
    
    
   
    #for each of these cycles generate a best fit circuit
    for j in range(len(df_list)):
        N_freqs = df_list[j].shape[0]
        frequencies = np.array(df_list[j]['f.C[Hz]'])
        Z = np.array(df_list[j]['Z'])
        Z_real = np.array(df_list[j]['Z.Re.C[mOhm*cm2]']) 
        Z_imag = np.array(df_list[j]['Z.Im.C[mOhm*cm2]']) #normalize to Ohm/cm^2
        Z_mag = np.abs(Z)
        
        R_inf_DRT, L_0_DRT, gamma_DRT, tau_vec = generateDRT(N_freqs, frequencies, Z, Z_real, Z_imag, Z_mag)
        
        key1 = f'Cell_{k:02}'
        key2 = f'AST {AST_labels[j]}'
        key = (key1, key2)
        DRT_dict[key] = [R_inf_DRT, L_0_DRT, gamma_DRT, tau_vec] #store for future uses
        
        results = []
        #iterate over all the possible circuit models we have
        for i in range(len(candidate_models)):
            model = ckt_init+candidate_models[i]
            print(f'Evaluating circuit {model} for Cell {k} and AST {AST_labels[j]}')
            
            circuit,residuals, Z_fit = generate_guesses(i, R_inf_DRT, L_0_DRT, gamma_DRT, tau_vec)
            
            if residuals is None and circuit is not None and circuit == 'single': #single DRT peak detected
                if k not in single_ckt:
                    single_ckt.append(k) #store problematic circuits and the model that you tried
                continue
            
            if residuals is None:
                continue #do not perform calculations on None data
                
            magnitude = np.sqrt(np.mean(np.abs(residuals)**2))
            
            rmse = np.sqrt(np.mean(np.abs(residuals)**2))
            mape = np.mean(np.abs((Z - Z_fit) / Z)) * 100

            print(f'RMSE: {rmse:.4f}')
            print(f'MAPE: {mape:.2f}%')
            
            keyVal = (key1, key2, model)
            circuit_dict[keyVal] = [residuals, circuit, rmse, mape] 
            #store the model, residuals, circuit object and the error analytics
            
            if mape < threshold:
                if model in AST_fit:
                    AST_fit[model][0] += 1
                else:
                    AST_fit[model] = [1,0]
                
        
        
#now check how many acceptable ASTs appear (we have already kept only the physically explainable models) 
invalid_models = dict()
#among the overall acceptable ECM values, let us see how the average MAPE changes across each entry
for model in AST_fit:
    mape = count = 0
    for k in range(start,end+1):
        if k == 6:
            continue
        for j in range(len(df_list)):
            key1 = f'Cell_{k:02}'
            key2 = f'AST {AST_labels[j]}'
            key = (key1, key2, model)
            if key in circuit_dict:
                mape += circuit_dict[key][-1]
                count += 1
            else: #mention that the given circuit cannot be used for the following cell and AST
                value = (key1,key2)
                if model in invalid_models:
                    invalid_models[model].append(value)
                else:
                    invalid_models[model] = [value,]
                
    AST_fit[model][-1] = round(mape/count,4) #store the average MAPE whenever the circuit is used for a fit
    #the MAPE should be bound below the threshold ... so we are choosing fairly good fits and checking which would 
    #be the best among them
            

#print(f'Count of acceptable models: {AST_fit}')    
total = sum(AST_fit[model][0] for model in AST_fit)

with open ('model_summary.txt','w') as f:
    for model in AST_fit:
        count = AST_fit[model][0]
        mape = AST_fit[model][1]
        p = count / total * 100
        line = f'Model {model} appears {count} times  = {p:.2f}% and a mean MAPE of {mape:.4f}\n'
        f.write(line)
    s  = 'Skipped the following cells due to lack of EIS data: '
    for cell in skipped_cells:
        s = s + str(cell)+', '
    f.write(s)
        
#store the evaluated models as files 
import pickle

# Save dictionaries to a file
with open("analytics.pkl", "wb") as f:
    pickle.dump({
        "circuit_dict": circuit_dict,
        "DRT_dict": DRT_dict,
        "invalid_model": invalid_models
    }, f)
        
    
    
    
    



    