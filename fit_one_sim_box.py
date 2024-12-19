import numpy as np
import scipy as sci
!pip install astroML
!pip install halotools
!pip install pytest-astropy
import halotools
import argparse

!pip install tabcorr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import wp 
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

%env TABCORR_DATABASE=./tabcorr

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

os.environ['TABCORR_DATABASE'] = './tabcorr'

box_index = 0
rp_ave_lim = 2.0

# Read in observational data for rp values
df = pd.read_csv('./data_2301.08692/obs_1n.csv')

# Read in mock covariance matrices
mock_cov_all = pd.read_csv('./mocks/alex_mock_a_b/cov_a_b_with_aDS.csv', header=None)

cov_mock_wp = mock_cov_all.iloc[:14, :14]
cov_wp_arr = cov_mock_wp.to_numpy()
cov_mock_ds = (mock_cov_all.iloc[14:28, 14:28] / 1e24)
cov_ds_arr = cov_mock_ds.to_numpy()
cov_mock_xi = mock_cov_all.iloc[28:70, 28:70]
cov_xi_arr = cov_mock_xi.to_numpy() 


# Read in mock data
mock_measurements = pd.read_csv('./mocks/alex_mock_a_b/mock_a_b_with_aDS.csv', sep = ',')
wp_mock = mock_measurements['wp'].values
ds_mock = mock_measurements['ds'].values
ds_mock = ds_mock / 1e12
all_xi_mock = []
all_xi_mock = all_xi_mock + list(mock_measurements['xi0'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi2'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi4'])
all_xi_mock = pd.Series(all_xi_mock).values

halotab_wp = database.read('AemulusAlpha', 0.4, 'wp', i_cosmo=box_index, tab_config='default')
halotab_ds = database.read('AemulusAlpha', 0.4, 'ds', i_cosmo=box_index, tab_config='default')

# for modelling RSD:
halotab_xi0 = database.read('AemulusAlpha', 0.4, 'xi0', i_cosmo=box_index, tab_config='default')

halotab_xi2 = database.read('AemulusAlpha', 0.4, 'xi2', i_cosmo=box_index, tab_config='default')

halotab_xi4 = database.read('AemulusAlpha', 0.4, 'xi4', i_cosmo=box_index, tab_config='default')


### Model For Additional Parameters:

from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import AssembiasZheng07Cens
from halotools.empirical_models import AssembiasZheng07Sats


class IncompleteAssembiasZheng07Cens(AssembiasZheng07Cens):

    def __init__(self, **kwargs):
        AssembiasZheng07Cens.__init__(self, **kwargs)
        self.param_dict['f_compl'] = 1.0

    def mean_occupation(self, **kwargs):
        return (AssembiasZheng07Cens.mean_occupation(self, **kwargs) *
                self.param_dict['f_compl'])


cens_occ_model = IncompleteAssembiasZheng07Cens(
prim_haloprop_key='halo_m200m', sec_haloprop_key='halo_vmax')
sats_occ_model = AssembiasZheng07Sats(
    prim_haloprop_key='halo_m200m', sec_haloprop_key='halo_vmax')

model = HodModelFactory(centrals_occupation=cens_occ_model,
                        satellites_occupation=sats_occ_model, redshift=0.4)


rp_min = df['rp_min']
rp_max = df['rp_max']
rp_ave = 0.5 * (rp_min + rp_max)


# MASK TO GET SUBSET OF RP VALUES
tripled_rp_ave = pd.Series(list(rp_ave) + list(rp_ave) + list(rp_ave))
tripled_rp_mask = tripled_rp_ave > rp_ave_lim

rp_mask = rp_ave > rp_ave_lim
rp_ave = rp_ave[rp_mask]

wp_mock = wp_mock[rp_mask]
ds_mock = ds_mock[rp_mask]
all_xi_mock = all_xi_mock[tripled_rp_mask]

# Apply mask to covariance matrices
cov_wp_arr = cov_wp_arr[np.outer(rp_mask, rp_mask)].reshape(np.sum(rp_mask), np.sum(rp_mask))
cov_ds_arr = cov_ds_arr[np.outer(rp_mask, rp_mask)].reshape(np.sum(rp_mask), np.sum(rp_mask))
cov_xi_arr = cov_xi_arr[np.outer(tripled_rp_mask, tripled_rp_mask)].reshape(np.sum(tripled_rp_mask), np.sum(tripled_rp_mask))


prior = Prior()  # initialize prior

# HOD parameters
prior.add_parameter('logM1', dist=(13.5, 15))
prior.add_parameter('logM0', dist=(12, 15))
prior.add_parameter('alpha', dist=(0.5, 2))
prior.add_parameter('logMmin', dist=(12.5, 14))
prior.add_parameter('sigma_logM', dist=(0.1, 1))

# Phase Space parameters
prior.add_parameter('alpha_s', dist=(0.8, 1.2))
prior.add_parameter('log_eta', dist=(-0.477, 0.477))

### ADDED THIS FOR RSDs
#prior.add_parameter('alpha_c', dist=(0.0, 0.4))

# Additional Parameters
prior.add_parameter('f_compl', dist=(0.5, 1.0))
prior.add_parameter('mean_occupation_centrals_assembias_param1', dist=(-1, +1))
prior.add_parameter('mean_occupation_satellites_assembias_param1', dist=(-1, +1))

# Number density of galaxies
ngal_info = pd.read_csv('ngal.csv')
ngal = ngal_info['ngal']
ngal_err = ngal_info['ngal_err']


def likelihood_combined(param_dict):
    model.param_dict.update(param_dict)
    ngal_wp, wp_mod = halotab_wp.predict(model)
    wp_mod = wp_mod[rp_mask]
    log_l_wp = multivariate_normal.logpdf(wp_mod, mean=wp_mock, cov=cov_wp_arr)

    ngal_ds, ds_mod = halotab_ds.predict(model)
    ds_mod = ds_mod / (1e12)
    ds_mod = ds_mod[rp_mask]
    log_l_ds = multivariate_normal.logpdf(ds_mod, mean=ds_mock, cov=cov_ds_arr)
    
    log_l_ngal_wp = multivariate_normal.logpdf(ngal_wp, mean=ngal, cov=ngal_err**2)
    log_l_ngal_ds = multivariate_normal.logpdf(ngal_ds, mean=ngal, cov=ngal_err**2)
    
    return log_l_wp + log_l_ds + log_l_ngal_wp

#    model.param_dict.update(param_dict)
#    xi0_mod = halotab_xi0.predict(model)[1]
#    xi2_mod = halotab_xi2.predict(model)[1]
#    xi4_mod = halotab_xi4.predict(model)[1]
#
#    all_xi_mod = []
#    all_xi_mod = all_xi_mod + list(xi0_mod)
#    all_xi_mod = all_xi_mod + list(xi2_mod)
#    all_xi_mod = all_xi_mod + list(xi4_mod)
#
#    all_xi_mod = pd.Series(all_xi_mod).values
#    all_xi_mod = all_xi_mod[tripled_rp_mask]
#    all_xi_mod = np.asarray(all_xi_mod)
#
#
#    log_l_xi = multivariate_normal.logpdf(all_xi_mod, mean=all_xi_mock, cov=cov_xi_arr, allow_singular=True)
#    return log_l_xi

sampler = Sampler(prior, likelihood_combined, filepath='box_'+str(box_index)+'.hdf5', n_live=1000, resume=False)  # the sampler uses an emulator for the likelihood to determine which parts of parameter space to sample
sampler.run(verbose=True)

points, log_w, log_l = sampler.posterior()

best_fit_index = np.argmax(log_l)
best_fit_params = points[best_fit_index]

# create final model with best fit params specified
model.param_dict['logM1'] = best_fit_params[0]
model.param_dict['logM0'] = best_fit_params[1]
model.param_dict['alpha'] = best_fit_params[2]
model.param_dict['logMmin'] = best_fit_params[3]
model.param_dict['sigma_logM'] = best_fit_params[4]
model.param_dict['alpha_s'] = best_fit_params[5]
model.param_dict['log_eta'] = best_fit_params[6]
#model.param_dict['alpha_c'] = best_fit_params[7] # Uncomment for RSDs
model.param_dict['f_compl'] = best_fit_params[7]
model.param_dict['mean_occupation_centrals_assembias_param1'] = best_fit_params[8]
model.param_dict['mean_occupation_satellites_assembias_param1'] = best_fit_params[9]


wp_mod_best_fit = halotab_wp.predict(model)[1]
ds_mod_best_fit = halotab_ds.predict(model)[1] / 1e12

log_l_max = np.max(log_l)
neg_chi_squared = 2 * log_l_max + (len(cov_wp_arr) * np.log(2 * np.pi) + np.linalg.slogdet(cov_wp_arr)[1]) + (len(cov_ds_arr) * np.log(2 * np.pi) + np.linalg.slogdet(cov_ds_arr)[1])
# neg_chi_squared = 2 * log_l_max + (len(cov_xi_arr) * np.log(2 * np.pi) + np.linalg.slogdet(cov_xi_arr)[1])
chi_squared = -1 * neg_chi_squared

# apply mask to best fit wp and ds:
wp_mod_best_fit = wp_mod_best_fit[rp_mask]
ds_mod_best_fit = ds_mod_best_fit[rp_mask]

# Plot best fit for this simulation box
plt.errorbar(rp_ave, wp_mock, yerr= np.sqrt(np.diag(cov_wp_arr)), fmt='o', color='blue', label='Mock $w_p$', capsize=5)
plt.plot(rp_ave, wp_mod_best_fit, c='red', label='Best fit $w_p$')
plt.xlabel(r'$r_p \ [h^{-1} \ \mathrm{Mpc}]$', fontsize=15)
plt.ylabel(r'$w_p \ [h^{-1} \ \mathrm{Mpc}]$', fontsize=15)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=12, frameon=False)
plt.title('Best Fit of $w_p$', fontsize=15)
plt.show()

plt.errorbar(rp_ave, rp_ave * ds_mock, yerr=rp_ave * np.sqrt(np.diag(cov_ds_arr)), fmt='o', color='blue', label='Mock $\Delta\Sigma$', capsize=5)
plt.plot(rp_ave, rp_ave * ds_mod_best_fit, c='red', label='Best fit $\Delta\Sigma$')
plt.xlabel(r'$r_{\rm p} \ [h^{-1} \ \mathrm{Mpc}]$', fontsize=15)
plt.ylabel(r'$r_{\rm p} \times \Delta\Sigma \ [10^6 \, M_\odot / \mathrm{pc}]$', fontsize=15)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=12, frameon=False)
plt.title('Best Fit of $\Delta\Sigma$', fontsize=15)
plt.show()
