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
parser.add_argument('max_ds_xi_rp', type=float, help='Upper threshold value of rp_ave for lensing and RSDs')
parser.add_argument('max_wp_rp', type=float, help='Upper threshold value of rp_ave for wp')
args = parser.parse_args()
box_index = args.box_index
rp_lim_ds = args.max_ds_xi_rp
rp_lim_xi = args.max_ds_xi_rp
rp_lim_wp = args.max_wp_rp

# Read in Observational Data for rp values
df = pd.read_csv('./data_2301.08692/obs_1n.csv')

# Read in mock measurements - covariance matrices
mock_cov_all = pd.read_csv('./mocks/cov_model1.csv', header=None)
cov_mock_wp = mock_cov_all.iloc[:14, :14]
cov_wp_arr = cov_mock_wp.to_numpy()
cov_mock_ds = (mock_cov_all.iloc[14:27, 14:27]) # mocks already account for /1e24 difference
cov_ds_arr = cov_mock_ds.to_numpy()

cov_mock_xi = mock_cov_all.iloc[27:69, 27:69]
cov_xi_arr = cov_mock_xi.to_numpy()

# Read in mock measurements
mock_measurements = pd.read_csv('./mocks/mock_model1.csv', sep = ',')
wp_mock = mock_measurements['wp'].values
ds_mock = mock_measurements['ds'].values # mocks already account for /1e12 difference
#ds_mock = ds_mock / 1e12
all_xi_mock = []
all_xi_mock = all_xi_mock + list(mock_measurements['xi0'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi2'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi4'])
all_xi_mock = pd.Series(all_xi_mock).values


max_likelihoods = []
evidences = []
chi_squareds = []

# For wp+ds
halotab_wp = database.read('AbacusSummit', 0.5, 'wp', i_cosmo=box_index, tab_config='efficient')
halotab_ds = database.read('AbacusSummit', 0.5, 'ds', i_cosmo=box_index, tab_config='efficient')

# For RSD:
halotab_xi0 = database.read('AbacusSummit', 0.5, 'xi0', i_cosmo=box_index, tab_config='efficient')
halotab_xi2 = database.read('AbacusSummit', 0.5, 'xi2', i_cosmo=box_index, tab_config='efficient')
halotab_xi4 = database.read('AbacusSummit', 0.5, 'xi4', i_cosmo=box_index, tab_config='efficient')

### Updated Model for Additional Parameters:

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


rp_min = df['rp_min']
rp_max = df['rp_max']
rp_ave = 0.5 * (rp_min + rp_max)

# MASK TO GET SUBSET OF RP VALUES
tripled_rp_ave = pd.Series(list(rp_ave) + list(rp_ave) + list(rp_ave))
tripled_rp_mask = tripled_rp_ave > rp_lim_xi

# First cut 14th element from lensing (if included; if not, makes no change), and for rp_ave_ds
rp_ave_cut = rp_ave[:13]
ds_mock_cut = ds_mock[:13]

rp_mask_ds = rp_ave_cut > rp_lim_ds
rp_mask_wp = rp_ave > rp_lim_wp

rp_ave_ds = rp_ave_cut[rp_mask_ds]
rp_ave_wp = rp_ave[rp_mask_wp]

wp_mock = wp_mock[rp_mask_wp]
ds_mock = ds_mock_cut[rp_mask_ds]
all_xi_mock = all_xi_mock[tripled_rp_mask]

## Apply mask to covariance matrices
cov_wp_arr = cov_wp_arr[np.outer(rp_mask_wp, rp_mask_wp)].reshape(np.sum(rp_mask_wp), np.sum(rp_mask_wp))
# First delete 14th measurement from ds cov matrix (if included; if not, makes no change)
cov_ds_arr = cov_ds_arr[:13, :13]
cov_ds_arr = cov_ds_arr[np.outer(rp_mask_ds, rp_mask_ds)].reshape(np.sum(rp_mask_ds), np.sum(rp_mask_ds))
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
ngal = ngal_info['ngal'].values
ngal = np.reshape(ngal, [1, 1])
ngal_err = ngal_info['ngal_err'].values
ngal_err = np.reshape(ngal_err, [1, 1])

def likelihood_combined(param_dict):
    
#    model.param_dict.update(param_dict)
#    ngal_wp, wp_mod = halotab_wp.predict(model, check_consistency=False)
#    wp_mod = wp_mod[rp_mask_wp]
#
#    ngal_ds, ds_mod = halotab_ds.predict(model, check_consistency=False)
#    ds_mod = ds_mod / (1e12)
#    ds_mod = ds_mod[rp_mask_ds]
#
#    # precision matrices to apply hartlap correction::
#    pre_wp = (125 - len(cov_wp_arr) - 2) / (125 - 1) * np.linalg.inv(cov_wp_arr)
#    pre_ds = (125 - len(cov_ds_arr) - 2) / (125 - 1) * np.linalg.inv(cov_ds_arr)
#
#    chi_sq_wp = np.inner(np.inner(wp_mod - wp_mock, pre_wp), wp_mod - wp_mock)
#    chi_sq_ds = np.inner(np.inner(ds_mod - ds_mock, pre_ds), ds_mod - ds_mock)
#
#    ngal_mod = np.mean([ngal_wp, ngal_ds])
#    chi_sq_ngal = np.inner(np.inner(ngal_mod - ngal, np.linalg.inv(ngal_err**2)), ngal_mod - ngal)
#    chi_sq_ngal = chi_sq_ngal[0, 0]

#    total_chi_sq = chi_sq_wp + chi_sq_ds + chi_sq_ngal
#    return total_chi_sq * -0.5


    model.param_dict.update(param_dict)
    ngal_0, xi0_mod = halotab_xi0.predict(model, check_consistency=False)
    ngal_2, xi2_mod = halotab_xi2.predict(model, check_consistency=False)
    ngal_4, xi4_mod = halotab_xi4.predict(model, check_consistency=False)

    all_xi_mod = []
    all_xi_mod = all_xi_mod + list(xi0_mod)
    all_xi_mod = all_xi_mod + list(xi2_mod)
    all_xi_mod = all_xi_mod + list(xi4_mod)

    all_xi_mod = pd.Series(all_xi_mod).values
    all_xi_mod = all_xi_mod[tripled_rp_mask]
    all_xi_mod = np.asarray(all_xi_mod)

    ngal_mod = np.mean([ngal_0, ngal_2, ngal_4])
    
    # precision matrix to apply hartlap correction:
    pre_xi = (125 - len(cov_xi_arr) - 2) / (125 - 1) * np.linalg.inv(cov_xi_arr)

    chi_sq_xi = np.inner(np.inner(all_xi_mod - all_xi_mock, pre_xi), all_xi_mod - all_xi_mock)
    chi_sq_ngal = np.inner(np.inner(ngal_mod - ngal, np.linalg.inv(ngal_err**2)), ngal_mod - ngal)
    chi_sq_ngal = chi_sq_ngal[0, 0]

    total_chi_sq = chi_sq_xi + chi_sq_ngal
    return total_chi_sq * -0.5
    

sampler = Sampler(prior, likelihood_combined, filepath='box_'+str(box_index)+'.hdf5', n_live=2000, resume=False)
sampler.run(f_live = 1e-10, discard_exploration=True, verbose=False)

points, log_w, log_l = sampler.posterior()
log_l_max = np.max(log_l)

max_likelihoods.append(log_l_max)
evidences.append(sampler.log_z)

file = open('output'+str(box_index)+'.txt', 'w')
file.write(f"Max Likelihood: {max_likelihoods[0]}\n")
file.write(f"Evidence: {evidences[0]}\n")
file.close()
