import math
import csv
import matplotlib.pyplot as plt
import pandas as pd

# Define constants and efficiency parameters
EtaC = 0.85  # Compressor efficiency
EtaF = 0.85  # Fan efficiency
PrB = 1  # Burner pressure ratio (assumed no pressure loss in the burner)
EtaT = 0.90  # Turbine efficiency
EtaN = 0.98  # Core nozzle efficiency
EtaNF = 0.97  # Fan nozzle efficiency
EtaD = 0.97  # Diffuser efficiency
gammaD = 1.4  # Specific heat ratio for diffuser
gammaC = 1.37  # Specific heat ratio for compressor
gammaF = 1.4  # Specific heat ratio for fan
gammaB = 1.35  # Specific heat ratio for burner
gammaT = 1.33  # Specific heat ratio for turbine
gammaN = 1.36  # Specific heat ratio for core nozzle
gammaNF = 1.4  # Specific heat ratio for fan nozzle
R = 287  # Gas constant in J/(kg*K)
QR = 4.5E+07  # Heat release in J/kg

# Calculate specific heats
CpC = gammaC * R / (gammaC - 1)  # Specific heat at constant pressure for compressor
CpF = gammaF * R / (gammaF - 1)  # Specific heat at constant pressure for fan
CpB = gammaB * R / (gammaB - 1)  # Specific heat at constant pressure for burner
CpT = gammaT * R / (gammaT - 1)  # Specific heat at constant pressure for turbine

# Initialize data storage for each variable in the format of lists within a dictionary
data = {
    'CpC': [],       # Specific heat at constant pressure for compressor
    'CpF': [],       # Specific heat at constant pressure for fan
    'CpB': [],       # Specific heat at constant pressure for burner
    'CpT': [],       # Specific heat at constant pressure for turbine
    'Prc': [],       # Compressor pressure ratio
    'B': [],         # Bypass ratio
    'PrF': [],       # Fan pressure ratio
    'Pa': [],        # Ambient pressure in Pa
    'Ta': [],        # Ambient temperature in K
    'Uinf': [],      # Inlet velocity (assumed 0 for Ma=0)
    'T02': [],       # Stagnation temperature at diffuser outlet
    'P02': [],       # Stagnation pressure at diffuser outlet
    'Rho02': [],     # Density at diffuser outlet
    'U02': [],       # Velocity at compressor inlet
    'P03': [],       # Stagnation pressure at compressor outlet
    'T03': [],       # Stagnation temperature at compressor outlet
    'P04': [],       # Stagnation pressure at burner outlet
    'T04': [],       # Stagnation temperature at burner outlet
    'f': [],         # Fuel-air ratio
    'P08': [],       # Stagnation pressure at fan outlet
    'T08': [],       # Stagnation temperature at fan outlet
    'T05': [],       # Stagnation temperature at turbine outlet
    'P05': [],       # Stagnation pressure at turbine outlet
    'UeC': [],       # Core nozzle exit velocity
    'UeF': [],       # Fan nozzle exit velocity
    'TMa': [],       # Thrust per unit mass flow
    'CoreA02': [],   # Core area at compressor inlet
    'Tprod': [],     # Total thrust production
    'TSFC': [],      # Thrust-specific fuel consumption
    'Icore': [],     # Core noise intensity
    'Ifan': [],      # Fan noise intensity
    'I': []          # Total noise intensity
}

# Set parameters
B = 2.8  # Bypass ratio
PrC = 14.6  # Compressor pressure ratio
PrF_values = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]  # Different fan pressure ratios
Ma = 0  # Mach number (assumed 0 for this calculation)
Pa = 96647.533  # Ambient pressure in Pa
Ta = 288.706  # Ambient temperature in K
a0 = (1.4 * R * Ta)**0.5  # Speed of sound in m/s
rho0 = Pa / (R * Ta)
thrust_required = 3700 * 4.44822  # Convert 3700 lbs thrust to Newtons

for PrF in PrF_values:
    # Calculate diffuser properties
    Uin = Ma * (gammaD * R * Ta)**0.5  # Inlet velocity (assumed 0 for Ma=0)
    T02 = Ta * (1 + 0.5 * (gammaD - 1) * Ma**2)  # Stagnation temperature at diffuser outlet
    P02 = Pa * (1 + EtaD * ((T02 / Ta) - 1))**(gammaD / (gammaD - 1))  # Stagnation pressure at diffuser outlet
    Rho02 = P02 / (R * T02)  # Density at diffuser outlet
    U02 = 0.3 * (gammaD * R * T02)**0.5  # Velocity at compressor inlet

    # Calculate compressor outlet properties
    P03 = PrC * P02  # Stagnation pressure at compressor outlet
    T03 = T02 * (1 + (1 / EtaC) * (PrC**((gammaC - 1) / gammaC) - 1))  # Stagnation temperature at compressor outlet

    # Fan diameter and total inlet area calculation
    fan_diameter = 28.2 / 39.37  # Convert fan diameter from inches to meters
    A_total = math.pi / 4 * fan_diameter**2  # Total inlet area
    A_core_target = A_total / (1 + B)  # Target core area

    T04 = 1100  # Initial guess for T04
    tolerance = 0.000001  # Tolerance for thrust matching

    while True:
        f = (T04 - T03) / ((QR / CpB) - T04)  # Fuel-air ratio

        # Calculate burner outlet properties
        P04 = PrB * P03  # Stagnation pressure at burner outlet
        
        # Calculate fan outlet properties
        P08 = PrF * P02  # Stagnation pressure at fan outlet
        T08 = T02 * (1 + (1 / EtaF) * (PrF**((gammaF - 1) / gammaF) - 1))  # Stagnation temperature at fan outlet

        # Calculate turbine outlet properties
        h_comp = CpC * (T03 - T02)  # Compressor work
        h_fan = CpF * (T08 - T02)  # Fan work
        T05 = T04 - ((1 / (1 + f)) * (h_comp / CpT)) - ((B / (1 + f)) * (h_fan / CpT))  # Stagnation temperature at turbine outlet
        P05 = P04 * (1 - (1 / EtaT) * (1 - T05 / T04))**(gammaT / (gammaT - 1))  # Stagnation pressure at turbine outlet

        # Calculate nozzle exit velocities
        UeC = (((2 * EtaN * gammaN * R * T05) / (gammaN - 1)) * (1 - (Pa / P05)**((gammaN - 1) / gammaN)))**0.5  # Core nozzle exit velocity
        UeF = (((2 * EtaNF * gammaNF * R * T08) / (gammaNF - 1)) * (1 - (Pa / P08)**((gammaNF - 1) / gammaNF)))**0.5  # Fan nozzle exit velocity

        # Calculate thrust and specific thrust
        TMa = (1 + f) * UeC + B * UeF - (B + 1) * Uin  # Thrust per unit mass flow
        CoreMa = thrust_required / TMa  # Core mass flow rate
        A02 = CoreMa / (Rho02 * U02)  # Area at compressor inlet

        if abs(A02 - A_core_target) < tolerance:
            break

        T04 += 0.01  # Adjust T04 to match the required thrust

    TP = TMa * CoreMa  # Total thrust production
    TSFC = f / TMa  # Thrust-specific fuel consumption

    # Calculate core noise
    T7 = T05 * (1 - EtaN * (1 - (Pa / P05)**((gammaN - 1) / gammaN)))  # Temperature at core nozzle exit
    Core = UeC / a0  # Core Mach number
    if Core.real < 2:
        P_core = (0.0001 * rho0 * UeC**8 * A02) / a0**5  # Core noise power
        I_core = P_core / (4 * math.pi * 450**2)  # Core noise intensity
        SICore = 10 * math.log10(I_core.real / 10**-12)  # Core sound intensity level
    else:
        P_core = 0.003 * rho0 * Core.real**3 * A02 * a0**3  # Core noise power (alternative method)
        I_core = P_core / (4 * math.pi * 450**2)  # Core noise intensity (alternative method)
        SICore = 10 * math.log10(I_core.real / 10**-12)  # Core sound intensity level (alternative method)

    # Calculate fan noise
    T9 = T08 * (1 - EtaNF * (1 - (Pa / P08)**((gammaNF - 1) / gammaNF)))  # Temperature at Fan nozzle exit
    P_fan = (0.0001 * rho0 * UeF**8 * (B * A02)) / a0**5  # Fan noise power
    I_fan = P_fan / (4 * math.pi * 450**2)  # Fan noise intensity
    SIfan = 10 * math.log10(I_fan.real / 10**-12)  #

    # Calculate total noise
    I = 10 * math.log10(10 ** (SICore.real / 10) + 10 ** (SIfan.real / 10))
    # Append calculated values to the corresponding lists in the dictionary
    data['CpC'].append(CpC)
    data['CpF'].append(CpF)
    data['CpB'].append(CpB)
    data['CpT'].append(CpT)
    data['Prc'].append(PrC)
    data['B'].append(B)
    data['PrF'].append(PrF)
    data['Pa'].append(Pa)
    data['Ta'].append(Ta)
    data['Uinf'].append(Uin)
    data['T02'].append(T02)
    data['P02'].append(P02)
    data['Rho02'].append(Rho02)
    data['U02'].append(U02)
    data['P03'].append(P03)
    data['T03'].append(T03)
    data['P04'].append(P04)
    data['T04'].append(T04)
    data['f'].append(f)
    data['P08'].append(P08)
    data['T08'].append(T08)
    data['T05'].append(T05)
    data['P05'].append(P05)
    data['UeC'].append(UeC)
    data['UeF'].append(UeF)
    data['TMa'].append(TMa)
    data['CoreA02'].append(A02)
    data['Tprod'].append(TP)
    data['TSFC'].append(TSFC)
    data['Icore'].append(SICore)
    data['Ifan'].append(SIfan)
    data['I'].append(I)

# Convert the data dictionary to a DataFrame for easy CSV writing and plotting
df = pd.DataFrame(data)

# Write the DataFrame to a CSV file
df.to_csv('output_final.csv')