# -*- coding: utf-8 -*-
"""
Created on Thur Jul  3 09:18:15 2025

@author: Biswadeep
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== Step 1: Process Each CV File ==========
# Main folder that contains all cell folders
main_dir = r"D:\Dataset\01-Data"

# Loop through all folders under the main folder
for subdir, dirs, files in os.walk(main_dir):
    if os.path.basename(subdir).lower() == '10-cv':
        # Look into subfolders of 10-CV for CV_cycle_corr.csv
        for cycle_folder in dirs:
            data_file = os.path.join(subdir, cycle_folder, 'CV_cycle_corr.csv')
            if os.path.isfile(data_file):
                try:
                    
                    # ------------Load and prepare the data-----------------
                    df = pd.read_csv(data_file, sep='\t')
                    
                    #-------------Save the results-------------------
                    output_plot_path = os.path.dirname(data_file)
                    
                    # Close the CV loop by adding the first row at the end
                    df_closed_loop = pd.concat([df, df.iloc[[0]]], ignore_index=True)
                    
                    #voltage and current values
                    voltage_col = 'U.C[V]'
                    current_col = 'I.C[mA/cm2]'

                    #---------------Auto detect baseline threshold for adsorption and desorption-------------- 
                    adsorption_region = df_closed_loop[
                        (df_closed_loop[voltage_col] >= 0.4) &
                        (df_closed_loop[voltage_col] <= 0.5) &
                        (df_closed_loop[current_col] < 0)
                    ]

                    desorption_region = df_closed_loop[
                        (df_closed_loop[voltage_col] >= 0.4) &
                        (df_closed_loop[voltage_col] <= 0.5) &
                        (df_closed_loop[current_col] > 0)
                    ]

                    auto_threshold_1 = adsorption_region[current_col].mean()
                    auto_threshold_2 = desorption_region[current_col].mean()

                    #----------Select Non Faradaic current density at 0.45 V to calculate the double layer capacity-----------
                    # Target voltage
                    target_voltage = 0.45
                    
                    # Find the closest current density values at 0.45 V in the data
                    # Adsorption
                    ads_idx = (adsorption_region[voltage_col] - target_voltage).abs().idxmin()
                    ads_current_045 = adsorption_region.loc[ads_idx, current_col]
                    
                    # Desorption
                    des_idx = (desorption_region[voltage_col] - target_voltage).abs().idxmin()
                    des_current_045 = desorption_region.loc[des_idx, current_col]

                    #--------------Calculate Adsorption and Desorption areas (used to calculate ECSA)------------------
                    v_min = 0.05
                    v_max = 0.4
                    x_iv = df_closed_loop[voltage_col].to_numpy()
                    y_iv = df_closed_loop[current_col].to_numpy()

                    # Adsorption
                    y_auto_selected_1 = np.where(
                        (x_iv >= v_min) & (x_iv <= v_max) & (y_iv < auto_threshold_1),
                        y_iv,
                        auto_threshold_1
                    )
                    auto_selected_area_1 = np.trapz(y_auto_selected_1, x_iv)

                    # Desorption
                    y_auto_selected_2 = np.where(
                        (x_iv >= v_min) & (x_iv <= v_max) & (y_iv > auto_threshold_2),
                        y_iv,
                        auto_threshold_2
                    )
                    auto_selected_area_2 = np.trapz(y_auto_selected_2, x_iv)

                    #--------------Plot----------------
                    # plt.figure(figsize=(10, 6))
                    # plt.plot(x_iv, y_iv, color='black', linewidth=3)
                    # plt.fill_between(x_iv, y_auto_selected_1, color='skyblue', alpha=0.4)
                    # plt.fill_between(x_iv, y_auto_selected_2, color='skyblue', alpha=0.4)
                    # plt.axhline(y=auto_threshold_1, color='purple', linestyle='--', linewidth=2)
                    # plt.axhline(y=auto_threshold_2, color='purple', linestyle='--', linewidth=2)
                    # plt.axvline(x=v_min, color='red', linestyle='--', linewidth=2)
                    # plt.axvline(x=v_max, color='red', linestyle='--', linewidth=2)
                    
                    # x_center_1 = (v_min + v_max) / 3
                    # y_center_1 = np.mean(y_auto_selected_1[(x_iv >= v_min) & (x_iv <= v_max)])
                    # plt.text(
                    #     x_center_1,
                    #     y_center_1,
                    #     "Adsorption",
                    #     fontsize=16,
                    #     color="purple",
                    #     ha='center',
                    #     va='center',
                    #     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                    # )
                    
                    # x_center_2 = (v_min + v_max) / 3
                    # y_center_2 = np.mean(y_auto_selected_2[(x_iv >= v_min) & (x_iv <= v_max)])
                    # plt.text(
                    #     x_center_2,
                    #     y_center_2,
                    #     "Desorption",
                    #     fontsize=16,
                    #     color="purple",
                    #     ha='center',
                    #     va='center',
                    #     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                    # )
                    
                    # plt.xlabel("Voltage (V)", fontsize=16)
                    # plt.ylabel("Current Density (mA/cm²)", fontsize=16)
                    # plt.tick_params(axis='both', labelsize=16)
                    
                    # #Add the value 0.05 V in the x-Axis
                    # xticks = plt.xticks()[0]
                    # xticks = [tick for tick in xticks if abs(tick - 0.00) > 1e-6]
                    # xticks.append(0.05)
                    # plt.xticks(sorted(xticks))
                    
                    
                    # plt.grid(True)
                    # plt.tight_layout()
                    # plot_filename = os.path.join(output_plot_path, "CV_Ads_Des_plot.png")
                    # plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    # plt.close()
                    
                    #------------Save the Results--------------
                    results = {
                        "Parameter": [
                            "Adsorption Threshold",
                            "Adsorption Area",
                            "Adsorption current density at 0.45 V",
                            "Desorption Threshold",
                            "Desorption Area",
                            "Desorption current density at 0.45 V"
                        ],
                        "Value": [
                            auto_threshold_1,
                            auto_selected_area_1,
                            ads_current_045,
                            auto_threshold_2,
                            auto_selected_area_2,
                            des_current_045
                        ],
                        "Unit": [
                            "mA/cm²", "mA·V/cm²", "mA/cm²",
                            "mA/cm²", "mA·V/cm²", "mA/cm²"
                        ]
                    }
                    results_df = pd.DataFrame(results)
                    excel_path = os.path.join(output_plot_path, "CV_results_Ads_Des.xlsx")
                    results_df.to_excel(excel_path, index=False)
                    print(f"Processed: {data_file}")

                except Exception as e:
                    print(f"Error processing {data_file}: {e}")

# ========== Step 2: Aggregate All Cell Results ==========
results = []
for subdir, dirs, files in os.walk(main_dir):
    if 'CV_results_Ads_Des.xlsx' in files:
        result_path = os.path.join(subdir, 'CV_results_Ads_Des.xlsx')
        try:
            #---------------read the excel file containing the saved results and access the Adsorption Area, Adsorption current density at 0.45 V and Desorption current density at 0.45 V
            df_result = pd.read_excel(result_path)
            area_value = df_result.loc[df_result['Parameter'] == 'Adsorption Area', 'Value'].values[0]
            ads_current_density = df_result.loc[df_result['Parameter'] == 'Adsorption current density at 0.45 V', 'Value'].values[0]
            des_current_density = df_result.loc[df_result['Parameter'] == 'Desorption current density at 0.45 V', 'Value'].values[0]
            cycle = os.path.basename(os.path.dirname(result_path))
            cell = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(result_path))))
            
            # Extract numeric cycle value for proper sorting (example: 0, 100, 1000)
            cycle_number = int(re.search(r'\d+', cycle).group()) if re.search(r'\d+', cycle) else 0

            results.append({
                "Cell": cell,
                "Cycle": cycle,
                "Cycle_Number": cycle_number,
                "Adsorption Area (mA·V/cm²)": area_value,
                "Adsorption current density (mA/cm²)": ads_current_density,
                "Desorption current density (mA/cm²)": des_current_density
            })
            print(f"Aggregated: {result_path}")

        except Exception as e:
            print(f"Failed to read {result_path}: {e}")

df_master = pd.DataFrame(results)
adsorption_path = 'Adsorption.xlsx'

# ========== Step 3: Remove Specific Cycles and add the Pt load values ==========

# Load Pt load values
pt_load_df = pd.read_excel("PtLoad.xlsx")

# Merge Pt load values
df_filtered = df_master.merge(pt_load_df, on="Cell", how="left")
df_filtered.to_excel('Adsorption.xlsx')
print(f"Adsorption data saved to:\n{adsorption_path}")
# ========== Step 4: Calculate ECSA and C_dl ==========
scan_rate = 0.1       # V/s (100 mV/s) (in the paper)
capacitance = 0.21    #mC/cm² (is a constant value: charge required to reduce a monolayer of protons on Pt)
#file_path = output_excel
file_path = 'Adsorption.xlsx'
df = pd.read_excel(file_path)


df['A_Pt (cm²_Pt/cm²_electrode)'] = df['Adsorption Area (mA·V/cm²)'] / (capacitance * scan_rate)
df['ECSA (cm²_Pt/mg_Pt)'] = df['A_Pt (cm²_Pt/cm²_electrode)'] / df['Pt load (mg/cm²_electrode)']
df['C_dl (mF/cm²)'] = (df['Desorption current density (mA/cm²)'] + abs(df['Adsorption current density (mA/cm²)'])) / (2 * scan_rate)

output_path = 'ECSA.xlsx'
df.to_excel(output_path, index=False)
print(f"ECSA saved to:\n{output_path}")
