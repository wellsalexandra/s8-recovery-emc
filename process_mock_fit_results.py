import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5 

folderpath = './fit_results/alex_mock_wp_ds_noSmall_a_b_with_aDS/'

max_likelihoods = []
evidences = []
chi_squareds = []

for i in range(40):
    f = open(folderpath+"output"+str(i)+".txt", "r")
    line1 = f.readline()
    line2 = f.readline()
    line3 = f.readline()
    
    max_likelihood = float(line1[15:])
    evidence = float(line2[9:])
    chi_squared = float(line3[12:])
    
    max_likelihoods.append(max_likelihood)
    evidences.append(evidence)
    chi_squareds.append(chi_squared)
    

# Plot over cosmological parameters

cosmo = pd.read_csv('./aa_cosmos.csv')

# Get individual arrays
ombh2 = cosmo['ombh2']
omch2 = cosmo['omch2']
w0 = cosmo['w0'] 
ns = cosmo['ns'] 
H0 = cosmo['H0'] 
h = H0 / 100
Neff = cosmo['Neff'] 
sigma8 = cosmo['sigma8']

# Derived parameters:
om = (ombh2 + omch2) / (h**2)
s8 = sigma8 * np.sqrt(om / 0.3)


import csv
def save_as_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['log_l', 's8', 'w0', 'Om'])
        writer.writerows(data.T)

data = np.asarray([max_likelihoods, s8, w0, om])
save_as_csv(data, 'cosmo_logl_rsd_a_b_with_aDS.csv')


# Remove box 23 
s8 = np.delete(s8.values, 23)
max_likelihoods = np.delete(max_likelihoods, 23)

# Plot
plt.scatter(s8, max_likelihoods, label='Data', color='blue')
plt.xlabel('$S_8$')
plt.ylabel('Maximum Log Likelihood')
plt.title('Max Log Likelihoods vs $S_8$')
plt.show()

