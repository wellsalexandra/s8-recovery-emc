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

# Command Line Argument from submit_job.sh file
parser = argparse.ArgumentParser(description="Fit observational data for a specified Box index")
parser.add_argument('box_index', type=int, help='The Box index to fit')
parser.add_argument('max_rp_ave', type=float, help='Upper threshold value of rp_ave')
args = parser.parse_args()
box_index = args.box_index
rp_ave_lim = args.max_rp_ave

# Read in Observational Data for rp values
df = pd.read_csv('./data_2301.08692/obs_1n.csv')

# Read in mock measurements - covariance matrices
mock_cov_all = pd.read_csv('./mocks/alex_mock_a_b/cov_a_b_with_aDS.csv', header=None)
cov_mock_wp = mock_cov_all.iloc[:14, :14]
cov_wp_arr = cov_mock_wp.to_numpy()
cov_mock_ds = (mock_cov_all.iloc[14:28, 14:28] / 1e24)
cov_ds_arr = cov_mock_ds.to_numpy()
cov_mock_xi = mock_cov_all.iloc[28:70, 28:70]
cov_xi_arr = cov_mock_xi.to_numpy()

# Read in mock measurements
mock_measurements = pd.read_csv('./mocks/alex_mock_a_b/mock_a_b_with_aDS.csv', sep=',')
wp_mock = mock_measurements['wp'].values
ds_mock = mock_measurements['ds'].values
ds_mock = ds_mock / 1e12
all_xi_mock = []
all_xi_mock = all_xi_mock + list(mock_measurements['xi0'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi2'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi4'])
all_xi_mock = pd.Series(all_xi_mock).values


max_likelihoods = []
evidences = []
chi_squareds = []

# For wp+ds
halotab_wp = database.read('AemulusAlpha', 0.4, 'wp', i_cosmo=box_index, tab_config='default')
halotab_ds = database.read('AemulusAlpha', 0.4, 'ds', i_cosmo=box_index, tab_config='default')

# For RSD:
halotab_xi0 = database.read('AemulusAlpha', 0.4, 'xi0', i_cosmo=box_index, tab_config='default')

halotab_xi2 = database.read('AemulusAlpha', 0.4, 'xi2', i_cosmo=box_index, tab_config='default')

halotab_xi4 = database.read('AemulusAlpha', 0.4, 'xi4', i_cosmo=box_index, tab_config='default')

### Updated Model for Additional Parameters:

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

## Apply mask to covariance matrices
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

### ADDED PARAMETER FOR RSDs ONLY
prior.add_parameter('alpha_c', dist=(0.0, 0.4))

# Additional parameters
prior.add_parameter('f_compl', dist=(0.5, 1.0))
prior.add_parameter('mean_occupation_centrals_assembias_param1', dist=(-1, +1))
prior.add_parameter('mean_occupation_satellites_assembias_param1', dist=(-1, +1))

# Number density of galaxies
ngal_info = pd.read_csv('ngal.csv')
ngal = ngal_info['ngal']
ngal_err = ngal_info['ngal_err'] # This is a singular value for the error
cov_ngal = ngal_err**2

def likelihood_combined(param_dict):
    model.param_dict.update(param_dict)
    ngal_wp, wp_mod = halotab_wp.predict(model)
    wp_mod = wp_mod[rp_mask] # Exclude smallest scales
    log_l_wp = multivariate_normal.logpdf(wp_mod, mean=wp_mock, cov=cov_wp_arr)

    ngal_ds, ds_mod = halotab_ds.predict(model)
    ds_mod = ds_mod / (1e12)
    ds_mod = ds_mod[rp_mask] # Exclude smallest scales
    log_l_ds = multivariate_normal.logpdf(ds_mod, mean=ds_mock, cov=cov_ds_arr)

    # Compute likelihoods of ngal for wp and for ds
    log_l_ngal_wp = multivariate_normal.logpdf(ngal_wp, mean=ngal, cov=cov_ngal)
    log_l_ngal_ds = multivariate_normal.logpdf(ngal_ds, mean=ngal, cov=cov_ngal)

    return log_l_wp + log_l_ds + log_l_ngal_wp + log_l_ngal_ds


#    model.param_dict.update(param_dict)
#    ngal_0, xi0_mod = halotab_xi0.predict(model)
#    ngal_2, xi2_mod = halotab_xi2.predict(model)
#    ngal_4, xi4_mod = halotab_xi4.predict(model)
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
#    log_l_ngal_0 = multivariate_normal.logpdf(ngal_0, mean=ngal, cov=cov_ngal)
#    log_l_ngal_2 = multivariate_normal.logpdf(ngal_2, mean=ngal, cov=cov_ngal)
#    log_l_ngal_4 = multivariate_normal.logpdf(ngal_4, mean=ngal, cov=cov_ngal)
#
#    log_l_xi = multivariate_normal.logpdf(all_xi_mod, mean=all_xi_mock, cov=cov_xi_arr, allow_singular=True)
#    return log_l_xi + log_l_ngal_0 + log_l_ngal_2 + log_l_ngal_4

sampler = Sampler(prior, likelihood_combined, filepath='box_'+str(box_index)+'.hdf5', n_live=1000, resume=False)
sampler.run(verbose=False)

points, log_w, log_l = sampler.posterior()
log_l_max = np.max(log_l)

# Calculate chi-squareds:

# For wp+ds:
neg_chi_squared = 2 * log_l_max + (len(cov_wp_arr) * np.log(2 * np.pi) + np.linalg.slogdet(cov_wp_arr)[1]) + (len(cov_ds_arr) * np.log(2 * np.pi) + np.linalg.slogdet(cov_ds_arr)[1]) + (len(cov_ngal) * np.log(2 * np.pi) + np.linalg.slogdet(cov_ngal)[1])
chi_squared = -1 * neg_chi_squared
# For RSDs:
#neg_chi_squared = 2 * log_l_max + (len(cov_xi_arr) * np.log(2 * np.pi) + np.linalg.slogdet(cov_xi_arr)[1])
#chi_squared = -1 * neg_chi_squared

max_likelihoods.append(log_l_max)
evidences.append(sampler.log_z)
chi_squareds.append(chi_squared)

file = open('output'+str(box_index)+'.txt', 'w')
file.write(f"Max Likelihood: {max_likelihoods[0]}\n")
file.write(f"Evidence: {evidences[0]}\n")
file.write(f"Chi-Squared: {chi_squareds[0]}\n")
file.close()
