!pip install git+https://github.com/johannesulf/TabCorr.git --ignore-installed --no-deps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5 
from astropy.table import Table
from nautilus.bounds.basic import Ellipsoid
from scipy.spatial import ConvexHull
from scipy.special import logsumexp
from scipy.spatial import Rectangle

import numpy as np
import scipy as sci
!pip install astroML
!pip install halotools
!pip install pytest-astropy
import halotools
import argparse

# !pip install tabcorr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import wp # here
from halotools.sim_manager import CachedHaloCatalog
from halotools.sim_manager import sim_defaults
from matplotlib import cm
from matplotlib import colors
from tabcorr import TabCorr
import h5py

!pip install nautilus-sampler
from scipy.stats import norm
from nautilus import Prior
from nautilus import Sampler

from tabcorr import database
from halotools.empirical_models import PrebuiltHodModelFactory
!pip install corner
import corner

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import multivariate_normal
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import wp
from halotools.sim_manager import CachedHaloCatalog
from halotools.sim_manager import sim_defaults
from tabcorr import TabCorr
from tabcorr import database
from nautilus import Prior, Sampler
import h5py
import corner
import os


folderpath_wp_ds = './fit_results/new_mock_wp_ds_Abacus/'
folderpath_rsd = './fit_results/new_mock_rsd_Abacus/'

import tabcorr
max_likelihoods_wp_ds = []
evidences_wp_ds = []
max_likelihoods_rsd = []
evidences_rsd = []

box = [0, 1, 4, 13, 100, 101, 102, 103, 104, 105, 112, 113, 116, 117, 
       118, 119, 120, 125, 126, 130, 131, 132, 133, 134, 135, 136, 
       137, 138, 139, 140, 141, 142, 143, 144, 145, 146]

for i in box:
    f = open(folderpath_wp_ds+"output"+str(i)+".txt", "r")
    line1 = f.readline()
    line2 = f.readline()
    
    max_likelihood = float(line1[15:])
    evidence = float(line2[9:])
    
    max_likelihoods_wp_ds.append(max_likelihood)
    evidences_wp_ds.append(evidence)

    f = open(folderpath_rsd+"output"+str(i)+".txt", "r")
    line1 = f.readline()
    line2 = f.readline()
    
    max_likelihood = float(line1[15:])
    evidence = float(line2[9:])
    
    max_likelihoods_rsd.append(max_likelihood)
    evidences_rsd.append(evidence)
    
# wp+ds
best_wp_ds_index = np.argmax(max_likelihoods_wp_ds)
print(box[best_wp_ds_index])
print(np.max(max_likelihoods_wp_ds))

# rsd
best_rsd_index = np.argmax(max_likelihoods_rsd)
print(box[best_rsd_index])
print(np.max(max_likelihoods_rsd))

# Load in points array from sampler from the fits of these two boxes
table_wp_ds = Table.read('sampler_post_wp_ds.csv')
log_l = table_wp_ds['log_l']
log_w = table_wp_ds['log_w']
w = np.exp(log_w)
points = [table_wp_ds['logM1'], table_wp_ds['logM0'], table_wp_ds['alpha'], table_wp_ds['logMmin'], 
                   table_wp_ds['sigma_logM'], table_wp_ds['alpha_s'], table_wp_ds['log_eta'], 
                   table_wp_ds['f_compl'], table_wp_ds['mean_occupation_centrals_assembias_param1'], 
                   table_wp_ds['mean_occupation_satellites_assembias_param1']]


points = np.asarray(points).T
best_fit_index = np.argmax(log_l)
best_fit_params = points[best_fit_index]

# create final model with best fit params specified
# model

box_index = 112 # wp+ds best fit cosmology
# box_index = 103 # RSD best fit cosmology
rp_lim_ds = 2.5
rp_lim_xi = 2.5
rp_lim_wp = 1.0

os.environ['TABCORR_DATABASE'] = './tabcorr'

from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import Zheng07Cens
from halotools.empirical_models import Zheng07Sats
from halotools.empirical_models import HeavisideAssembias

class IncompleteZheng07Cens(Zheng07Cens):

    def __init__(self, **kwargs):
        Zheng07Cens.__init__(self, **kwargs)
        self.param_dict['f_compl'] = 1.0

    def mean_occupation(self, **kwargs):
        return (Zheng07Cens.mean_occupation(self, **kwargs) *
                self.param_dict['f_compl'])

class AssembiasIncompleteZheng07Cens(
        IncompleteZheng07Cens, HeavisideAssembias):

    def __init__(self, **kwargs):

        IncompleteZheng07Cens.__init__(self, **kwargs)
        HeavisideAssembias.__init__(
            self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate="mean_occupation",
            **kwargs)

cens_occ_model = AssembiasIncompleteZheng07Cens(
prim_haloprop_key='halo_m200m', sec_haloprop_key='halo_vmax')
sats_occ_model = Zheng07Sats(
    prim_haloprop_key='halo_m200m', sec_haloprop_key='halo_vmax')

model = HodModelFactory(centrals_occupation=cens_occ_model,
                        satellites_occupation=sats_occ_model, redshift=0.5)

# halotab
halotab_wp = database.read('AbacusSummit', 0.5, 'wp', i_cosmo=box_index, tab_config='efficient')
halotab_ds = database.read('AbacusSummit', 0.5, 'ds', i_cosmo=box_index, tab_config='efficient')
halotab_xi0 = database.read('AbacusSummit', 0.5, 'xi0', i_cosmo=box_index, tab_config='efficient')
halotab_xi2 = database.read('AbacusSummit', 0.5, 'xi2', i_cosmo=box_index, tab_config='efficient')
halotab_xi4 = database.read('AbacusSummit', 0.5, 'xi4', i_cosmo=box_index, tab_config='efficient')

# obs first
folderpath = './data_2301.08692/'
df = pd.read_csv(folderpath + 'obs_1n.csv')
rp_min = df['rp_min']
rp_max = df['rp_max']
rp_ave = 0.5 * (rp_min + rp_max)

# mock data
mock_cov_all = pd.read_csv('./mocks/cov_model1.csv', header=None)

cov_mock_wp = mock_cov_all.iloc[:14, :14]
cov_wp_arr = cov_mock_wp.to_numpy()
# cov_mock_ds = (mock_cov_all.iloc[14:27, 14:27] / 1e24)
cov_mock_ds = (mock_cov_all.iloc[14:27, 14:27])
cov_ds_arr = cov_mock_ds.to_numpy()
cov_mock_xi = mock_cov_all.iloc[27:69, 27:69]
cov_xi_arr = cov_mock_xi.to_numpy() 

mock_measurements = pd.read_csv('./mocks/mock_model1.csv', sep = ',')
wp_mock = mock_measurements['wp'].values
ds_mock = mock_measurements['ds'].values
# ds_mock = ds_mock / 1e12

all_xi_mock = []
all_xi_mock = all_xi_mock + list(mock_measurements['xi0'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi2'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi4'])
all_xi_mock = pd.Series(all_xi_mock).values

# masks
rp_ave_cut = rp_ave[:13]
ds_mock_cut = ds_mock[:13]

model.param_dict['logM1'] = best_fit_params[0]
model.param_dict['logM0'] = best_fit_params[1]
model.param_dict['alpha'] = best_fit_params[2]
model.param_dict['logMmin'] = best_fit_params[3]
model.param_dict['sigma_logM'] = best_fit_params[4]
model.param_dict['alpha_s'] = best_fit_params[5]
model.param_dict['log_eta'] = best_fit_params[6]
# model.param_dict['alpha_c'] = best_fit_params[7] # for RSDs
model.param_dict['f_compl'] = best_fit_params[7]
model.param_dict['mean_occupation_centrals_assembias_param1'] = best_fit_params[8]
model.param_dict['mean_occupation_satellites_assembias_param1'] = best_fit_params[9]

wp_mod_best_fit = halotab_wp.predict(model, check_consistency=False)[1]
ds_mod_best_fit = halotab_ds.predict(model, check_consistency=False)[1] / 1e12

ax = plt.gca()  
ax.axvspan(0, 1, facecolor='gray', alpha=0.5)

plt.errorbar(rp_ave, wp_mock, yerr= np.sqrt(np.diag(cov_wp_arr)), fmt='.', color='blue', capsize=3, capthick=0.5)
plt.plot(rp_ave, wp_mod_best_fit, c='blue', label='$w_p$')
plt.xlabel(r'$r_p \ [h^{-1} \ \mathrm{Mpc}]$')
plt.ylabel(r'$w_p \ [h^{-1} \ \mathrm{Mpc}]$')
plt.xscale('log')
plt.yscale('log')
# plt.legend(fontsize=12, frameon=False)
# plt.title('Best Fit of $w_p$', fontsize=15)
plt.show()

ax = plt.gca()  
ax.axvspan(0, 2.5, facecolor='gray', alpha=0.5)
plt.errorbar(rp_ave_cut, rp_ave_cut * ds_mock[:13], yerr=rp_ave_cut * np.sqrt(np.diag(cov_ds_arr)), fmt='.', color='blue', capsize=5, capthick=0.5)
plt.plot(rp_ave_cut, rp_ave_cut * ds_mod_best_fit[:13], c='blue', label='$\Delta\Sigma$')
plt.xlabel(r'$r_{\rm p} \ [h^{-1} \ \mathrm{Mpc}]$')
plt.ylabel(r'$r_{\rm p} \times \Delta\Sigma \ [10^6 \, M_\odot / \mathrm{pc}]$')
plt.xscale('log')
plt.yscale('log')
# plt.legend(fontsize=12, frameon=False)
# plt.title('Best Fit of $\Delta\Sigma$', fontsize=15)
plt.show()
