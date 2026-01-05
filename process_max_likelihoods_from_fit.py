#%%
import sys
pip install numpy pandas matplotlib astroML halotools pytest-astropy nautilus-sampler corner
pip install --upgrade --force-reinstall git+https://github.com/johannesulf/TabCorr.git

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5 
import tabcorr
import csv

#%%
# 
fit = 'rsd' # options: 'wpds' or 'rsd'
mock_type = 'mass' # options: 'mass+conc' or 'mass' 

#%%
max_likelihoods = []
evidences = []

folderpath = './fit_results/'+str(fit)+'_'+str(mock_type)+'/'

box = [0, 1, 4, 13, 100, 101, 102, 103, 104, 105, 112, 113, 116, 117, 
       118, 119, 120, 125, 126, 130, 131, 132, 133, 134, 135, 136, 
       137, 138, 139, 140, 141, 142, 143, 144, 145, 146]

for i in box:
    f = open(folderpath+str(fit)+"_output"+str(i)+".txt", "r")

    line1 = f.readline()
    line2 = f.readline()
    
    max_likelihood = float(line1[15:])
    evidence = float(line2[9:])
    
    max_likelihoods.append(max_likelihood)
    evidences.append(evidence)

print(box[np.argmax(max_likelihoods)])
print(np.max(max_likelihoods))



#%%
# Read in cosmological parameters from TabCorr database
box = [0, 1, 4, 13, 100, 101, 102, 103, 104, 105, 112, 113, 116, 117, 
       118, 119, 120, 125, 126, 130, 131, 132, 133, 134, 135, 136, 
       137, 138, 139, 140, 141, 142, 143, 144, 145, 146]

s8 = []
om = []
n_s = []
sigma8 = []
for b in box:
    cosmo = tabcorr.database.cosmology('AbacusSummit', b)
    sigma_8 = cosmo.sigma8
    sigma8.append(sigma_8)

    Om0 = cosmo.Om0
    om.append((Om0))

    s_8 = sigma_8 * np.sqrt((Om0) / 0.3) 
    s8.append(s_8)

    n_s.append(cosmo.ns)

#%%
s8 = np.asarray(s8)
om = np.asarray(om)
n_s = np.asarray(n_s)

#%%

import csv

def save_as_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['log L_max', 'S_8', 'n_s', 'O_m'])
        writer.writerows(data.T)

data = np.asarray([max_likelihoods, s8, n_s, om])
save_as_csv(data, 'cosmo_logl_'+str(fit)+'_'+str(mock_type)+'.csv')
