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

# for command line
parser = argparse.ArgumentParser(description="Fit observational data for a specified Box index")
parser.add_argument('box_index', type=int, help='The Box index to fit')
parser.add_argument('max_rp_ave', type=float, help='Upper threshold value of rp_ave')
args = parser.parse_args()
box_index = args.box_index
rp_ave_lim = args.max_rp_ave

# obs data
folderpath = './data_2301.08692/'
df = pd.read_csv(folderpath + 'obs_1n.csv')
#
#
#cov_wp_1n = pd.read_csv(folderpath + 'cov_wp_1n.csv', sep=' ', header=None)
#cov_wp_arr = cov_wp_1n.to_numpy()
#cov_ds_1n = pd.read_csv(folderpath + 'cov_ds_1n.csv', sep=' ', header=None)
#cov_ds_arr = cov_ds_1n.to_numpy()
#cov_xi_1n = pd.read_csv(folderpath+'cov_xi_1n_updated.csv', sep=',', header=None)
#cov_xi_arr = cov_xi_1n.to_numpy()

# mock data
#mock_cov_all = pd.read_csv('./mocks/alex_mocks_with_mock_cov/cov_alex_newest.csv', header=None)
mock_cov_all = pd.read_csv('./mocks/alex_mock_a_b/cov_a_b_with_aDS.csv', header=None)

cov_mock_wp = mock_cov_all.iloc[:14, :14]
cov_wp_arr = cov_mock_wp.to_numpy()
cov_mock_ds = (mock_cov_all.iloc[14:28, 14:28] / 1e24)
cov_ds_arr = cov_mock_ds.to_numpy()
cov_mock_xi = mock_cov_all.iloc[28:70, 28:70]
cov_xi_arr = cov_mock_xi.to_numpy() #redefined cov_xi_arr here for when I wanna use mock

# old mock data
#mock_measurements = pd.read_csv('mock.csv', sep=',')
#wp_mock = mock_measurements['wp'].values
#ds_mock = mock_measurements['ds'].values
#ds_mock = ds_mock / 1e12

# new mock data
#mock_measurements = pd.read_csv('./mocks/alex_mocks_with_mock_cov/mock_alex_newest.csv', sep=',')
mock_measurements = pd.read_csv('./mocks/alex_mock_a_b/mock_a_b_with_aDS.csv', sep=',')
wp_mock = mock_measurements['wp'].values
ds_mock = mock_measurements['ds'].values
ds_mock = ds_mock / 1e12

all_xi_mock = []
all_xi_mock = all_xi_mock + list(mock_measurements['xi0'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi2'])
all_xi_mock = all_xi_mock + list(mock_measurements['xi4'])
all_xi_mock = pd.Series(all_xi_mock).values


#### LOOP BEGINS HERE

max_likelihoods = []
evidences = []
chi_squareds = []

halotab_wp = database.read('AemulusAlpha', 0.4, 'wp', i_cosmo=box_index, tab_config='default')
halotab_ds = database.read('AemulusAlpha', 0.4, 'ds', i_cosmo=box_index, tab_config='default')

# for modelling RSD:
halotab_xi0 = database.read('AemulusAlpha', 0.4, 'xi0', i_cosmo=box_index, tab_config='default')

halotab_xi2 = database.read('AemulusAlpha', 0.4, 'xi2', i_cosmo=box_index, tab_config='default')

halotab_xi4 = database.read('AemulusAlpha', 0.4, 'xi4', i_cosmo=box_index, tab_config='default')

#model = PrebuiltHodModelFactory('zheng07', prim_haloprop_key='halo_m200m', redshift=0.25)

### ADDED/CHANGED MODEL FOR ADDITIONAL PARAMETERS - from Johannes

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

#wp_obs = df['wp'].values
#ds_obs = df['ds'].values

# RSD observations
#all_xi = []
#all_xi = all_xi + list(df['xi0'])
#all_xi = all_xi + list(df['xi2'])
#all_xi = all_xi + list(df['xi4'])
#all_xi_obs = all_xi

# MASK TO GET SUBSET OF RP VALUES


tripled_rp_ave = pd.Series(list(rp_ave) + list(rp_ave) + list(rp_ave))
tripled_rp_mask = tripled_rp_ave > rp_ave_lim

rp_mask = rp_ave > rp_ave_lim
rp_ave = rp_ave[rp_mask]

#
#wp_obs = wp_obs[rp_mask]
#ds_obs = ds_obs[rp_mask]
#
wp_mock = wp_mock[rp_mask]
ds_mock = ds_mock[rp_mask]
all_xi_mock = all_xi_mock[tripled_rp_mask]
#
## UPDATE COV MATRICES
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

# phase space parameters
prior.add_parameter('alpha_s', dist=(0.8, 1.2))
prior.add_parameter('log_eta', dist=(-0.477, 0.477))
### ADDED THIS FOR RSDs
prior.add_parameter('alpha_c', dist=(0.0, 0.4))

# additional parameters
prior.add_parameter('f_compl', dist=(0.5, 1.0))
prior.add_parameter('mean_occupation_centrals_assembias_param1', dist=(-1, +1))
prior.add_parameter('mean_occupation_satellites_assembias_param1', dist=(-1, +1))

# number density of galaxies
ngal_info = pd.read_csv('ngal.csv')
ngal = ngal_info['ngal']
ngal_err = ngal_info['ngal_err']
cov_ngal = ngal_err**2

def likelihood_combined(param_dict):
#    model.param_dict.update(param_dict)
#    ngal_wp, wp_mod = halotab_wp.predict(model)
#    wp_mod = wp_mod[rp_mask]
#    log_l_wp = multivariate_normal.logpdf(wp_mod, mean=wp_mock, cov=cov_wp_arr)
#
#    ngal_ds, ds_mod = halotab_ds.predict(model)
#    ds_mod = ds_mod / (1e12)
#    ds_mod = ds_mod[rp_mask]
#    log_l_ds = multivariate_normal.logpdf(ds_mod, mean=ds_mock, cov=cov_ds_arr)
#
#    log_l_ngal_wp = multivariate_normal.logpdf(ngal_wp, mean=ngal, cov=cov_ngal)
#    log_l_ngal_ds = multivariate_normal.logpdf(ngal_ds, mean=ngal, cov=cov_ngal)
#
##    return log_l_wp + log_l_ds + log_l_ngal_wp
#    return log_l_wp + log_l_ds + log_l_ngal_wp + log_l_ngal_ds

#    return log_l_wp + log_l_ds

    model.param_dict.update(param_dict)
    ngal_0, xi0_mod = halotab_xi0.predict(model)
    ngal_2, xi2_mod = halotab_xi2.predict(model)
    ngal_4, xi4_mod = halotab_xi4.predict(model)

    all_xi_mod = []
    all_xi_mod = all_xi_mod + list(xi0_mod)
    all_xi_mod = all_xi_mod + list(xi2_mod)
    all_xi_mod = all_xi_mod + list(xi4_mod)

    all_xi_mod = pd.Series(all_xi_mod).values
    all_xi_mod = all_xi_mod[tripled_rp_mask]
    all_xi_mod = np.asarray(all_xi_mod)

    log_l_ngal_0 = multivariate_normal.logpdf(ngal_0, mean=ngal, cov=cov_ngal)
    log_l_ngal_2 = multivariate_normal.logpdf(ngal_2, mean=ngal, cov=cov_ngal)
    log_l_ngal_4 = multivariate_normal.logpdf(ngal_4, mean=ngal, cov=cov_ngal)
    
    log_l_xi = multivariate_normal.logpdf(all_xi_mod, mean=all_xi_mock, cov=cov_xi_arr, allow_singular=True)
    return log_l_xi + log_l_ngal_0 + log_l_ngal_2 + log_l_ngal_4

sampler = Sampler(prior, likelihood_combined, filepath='box_'+str(box_index)+'.hdf5', n_live=1000, resume=False)  # the sampler uses an emulator for the likelihood to determine which parts of parameter space to sample
sampler.run(verbose=False)

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
model.param_dict['alpha_c'] = best_fit_params[7] # for RSDs
model.param_dict['f_compl'] = best_fit_params[8]
model.param_dict['mean_occupation_centrals_assembias_param1'] = best_fit_params[9]
model.param_dict['mean_occupation_satellites_assembias_param1'] = best_fit_params[10]


#wp_mod_best_fit = halotab_wp.predict(model)[1]
#ds_mod_best_fit = halotab_ds.predict(model)[1] / 1e12

log_l_max = np.max(log_l)
#neg_chi_squared = 2 * log_l_max + (len(cov_wp_arr) * np.log(2 * np.pi) + np.linalg.slogdet(cov_wp_arr)[1]) + (len(cov_ds_arr) * np.log(2 * np.pi) + np.linalg.slogdet(cov_ds_arr)[1]) + (len(cov_ngal) * np.log(2 * np.pi) + np.linalg.slogdet(cov_ngal)[1])
#neg_chi_squared = 2 * log_l_max + (len(cov_xi_arr) * np.log(2 * np.pi) + np.linalg.slogdet(cov_xi_arr)[1])
#chi_squared = -1 * neg_chi_squared

max_likelihoods.append(log_l_max)
evidences.append(sampler.log_z)
#chi_squareds.append(chi_squared)

file = open('output'+str(box_index)+'.txt', 'w')
file.write(f"Max Likelihood: {max_likelihoods[0]}\n")
file.write(f"Evidence: {evidences[0]}\n")
file.write(f"Chi-Squared: {chi_squareds[0]}\n")
file.close()

#print("Finished with Box index " + str(box_index))
#
## plot over cosmological parameters
#cosmo = pd.read_csv('./aa_cosmos.csv')
#
## Get individual arrays
#ombh2 = cosmo['ombh2']
#omch2 = cosmo['omch2']
#w0 = cosmo['w0']  # 1
#ns = cosmo['ns']  # 2
#H0 = cosmo['H0']  # 3
#h = H0 / 100
#Neff = cosmo['Neff']  # 4
#sigma8 = cosmo['sigma8']  # 5
#
#om = (ombh2 + omch2) / (h**2)  # 6
#s8 = sigma8 * np.sqrt(om / 0.3)  # 7
#
#fig1, ax1 = plt.subplots(2, 4, figsize=(20, 10))
#
#ax1[0, 0].scatter(om, max_likelihoods) # 1
#ax1[0, 0].set_xlabel("$\Omega_m$")
#ax1[0, 1].scatter(w0, max_likelihoods) # 2
#ax1[0, 1].set_xlabel("$w_0$")
#ax1[0, 2].scatter(ns, max_likelihoods) # 3
#ax1[0, 2].set_xlabel("$n_s$")
#ax1[0, 3].scatter(H0, max_likelihoods) # 4
#ax1[0, 3].set_xlabel("$H_0$")
#ax1[1, 0].scatter(Neff, max_likelihoods) # 5
#ax1[1, 0].set_xlabel("$N_{eff}$")
#ax1[1, 1].scatter(sigma8, max_likelihoods) # 6
#ax1[1, 1].set_xlabel("$\sigma_8$")
#ax1[1, 2].scatter(s8, max_likelihoods) # 7
#ax1[1, 2].set_xlabel("$S_8 = \sigma_8 \sqrt{\Omega_m / 0.3}$")
#fig1.delaxes(ax1[1,3])
#
#ax1[0,0].set_ylabel("Max Log Likelihood")
#ax1[0,1].set_ylabel("Max Log Likelihood")
#ax1[0,2].set_ylabel("Max Log Likelihood")
#ax1[0,3].set_ylabel("Max Log Likelihood")
#ax1[1,0].set_ylabel("Max Log Likelihood")
#ax1[1,1].set_ylabel("Max Log Likelihood")
#ax1[1,2].set_ylabel("Max Log Likelihood")
#plt.show()
#
#fig2, ax2 = plt.subplots(2, 4, figsize=(20, 10))
#
#ax2[0, 0].scatter(om, evidences) # 1
#ax2[0, 0].set_xlabel("$\Omega_m$")
#ax2[0, 1].scatter(w0, evidences) # 2
#ax2[0, 1].set_xlabel("$w_0$")
#ax2[0, 2].scatter(ns, evidences) # 3
#ax2[0, 2].set_xlabel("$n_s$")
#ax2[0, 3].scatter(H0, evidences) # 4
#ax2[0, 3].set_xlabel("$H_0$")
#ax2[1, 0].scatter(Neff, evidences) # 5
#ax2[1, 0].set_xlabel("$N_{eff}$")
#ax2[1, 1].scatter(sigma8, evidences) # 6
#ax2[1, 1].set_xlabel("$\sigma_8$")
#ax2[1, 2].scatter(s8, evidences) # 7
#ax2[1, 2].set_xlabel("$S_8 = \sigma_8 \sqrt{\Omega_m / 0.3}$")
#fig2.delaxes(ax2[1,3])
#
#ax2[0,0].set_ylabel("Evidence")
#ax2[0,1].set_ylabel("Evidence")
#ax2[0,2].set_ylabel("Evidence")
#ax2[0,3].set_ylabel("Evidence")
#ax2[1,0].set_ylabel("Evidence")
#ax2[1,1].set_ylabel("Evidence")
#ax2[1,2].set_ylabel("Evidence")
#plt.show()
#
#fig2, ax2 = plt.subplots(2, 4, figsize=(20, 10))
#
#ax2[0, 0].scatter(om, chi_squareds) # 1
#ax2[0, 0].set_xlabel("$\Omega_m$")
#ax2[0, 1].scatter(w0, chi_squareds) # 2
#ax2[0, 1].set_xlabel("$w_0$")
#ax2[0, 2].scatter(ns, chi_squareds) # 3
#ax2[0, 2].set_xlabel("$n_s$")
#ax2[0, 3].scatter(H0, chi_squareds) # 4
#ax2[0, 3].set_xlabel("$H_0$")
#ax2[1, 0].scatter(Neff, chi_squareds) # 5
#ax2[1, 0].set_xlabel("$N_{eff}$")
#ax2[1, 1].scatter(sigma8, chi_squareds) # 6
#ax2[1, 1].set_xlabel("$\sigma_8$")
#ax2[1, 2].scatter(s8, chi_squareds) # 7
#ax2[1, 2].set_xlabel("$S_8 = \sigma_8 \sqrt{\Omega_m / 0.3}$")
#fig2.delaxes(ax2[1,3])
#
#ax2[0,0].set_ylabel("Chi_squared")
#ax2[0,1].set_ylabel("Chi_squared")
#ax2[0,2].set_ylabel("Chi_squared")
#ax2[0,3].set_ylabel("Chi_squared")
#ax2[1,0].set_ylabel("Chi_squared")
#ax2[1,1].set_ylabel("Chi_squared")
#ax2[1,2].set_ylabel("Chi_squared")
#plt.show()
