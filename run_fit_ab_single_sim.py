#%%
import sys

pip install numpy pandas matplotlib astroML halotools pytest-astropy nautilus-sampler corner
pip install --upgrade --force-reinstall git+https://github.com/johannesulf/TabCorr.git

#%%
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import halotools
from scipy.stats import multivariate_normal
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import wp
from halotools.sim_manager import CachedHaloCatalog
from halotools.sim_manager import sim_defaults
from tabcorr import TabCorr
from tabcorr import database
from nautilus import Prior, Sampler
from astropy.table import Table
import h5py
import corner
import os
import csv

%env TABCORR_DATABASE=./tabcorr
os.environ['TABCORR_DATABASE'] = './tabcorr'

#%%
# From previous fitting over cosmologies, obtained box 120 as 
# best fit for wp+ds and box 103 as best fit for RSD. 
box_index = 120 
fit = 'wpds' # options: wpds or rsd

#%%
rp_lim_ds = 2.5
rp_lim_xi = 2.5
rp_lim_wp = 1.0

config = database.configuration('efficient')
s_bins = np.sqrt(config['s_bins'][1:] * config['s_bins'][:-1])
rp_wp_bins = np.sqrt(config['rp_wp_bins'][1:] * config['rp_wp_bins'][:-1])
rp_ds_bins = np.sqrt(config['rp_ds_bins'][1:] * config['rp_ds_bins'][:-1])


# mock data
mock_cov_all = pd.read_csv('./mocks/cov_model1.csv', header=None)
table = Table.read('mock_model1.fits')
wp_mock = table['wp'].data
ds_mock = table['ds'].data
all_xi_mock = np.concatenate([table['xi0'].data,
                              table['xi2'].data,
                              table['xi4'].data])


xi0_mock = table['xi0'].data
xi2_mock = table['xi2'].data
xi4_mock = table['xi4'].data

# number density of galaxies
ngal = table.meta['N']
ngal = np.reshape(ngal, [1, 1])
ngal_err = table.meta['N_ERR']
ngal_err = np.reshape(ngal_err, [1, 1])

cov_xi_arr = np.array(mock_cov_all.iloc[27:69, 27:69])
cov_wp_ds_arr = np.array(mock_cov_all.iloc[:27, :27])

# apply scale cuts to data:

# First cut 14th element from lensing
ds_mock_cut = ds_mock[:13]

# create masks
rp_mask_ds = rp_ds_bins > rp_lim_ds
rp_mask_wp = rp_wp_bins > rp_lim_wp
rp_mask_wp_ds = np.concatenate([rp_mask_wp, rp_mask_ds]) 
tripled_rp_ave = pd.Series(list(s_bins) + list(s_bins) + list(s_bins))
tripled_rp_mask = tripled_rp_ave > rp_lim_xi # tripled rp_ave mask to be applied to xi

# apply masks
wp_mock = wp_mock[rp_mask_wp]
ds_mock = ds_mock_cut[rp_mask_ds]
all_xi_mock = all_xi_mock[tripled_rp_mask]
cov_xi_arr = cov_xi_arr[np.outer(tripled_rp_mask, tripled_rp_mask)].reshape(np.sum(tripled_rp_mask), np.sum(tripled_rp_mask))
cov_wp_ds_arr = cov_wp_ds_arr[np.outer(rp_mask_wp_ds, rp_mask_wp_ds)].reshape(np.sum(rp_mask_wp_ds), np.sum(rp_mask_wp_ds))

#%%
max_likelihoods = []
evidences = []
chi_squareds = []

halotab_wp = database.read('AbacusSummit', 0.5, 'wp', i_cosmo=box_index, tab_config='efficient')
halotab_ds = database.read('AbacusSummit', 0.5, 'ds', i_cosmo=box_index, tab_config='efficient')
halotab_xi0 = database.read('AbacusSummit', 0.5, 'xi0', i_cosmo=box_index, tab_config='efficient')
halotab_xi2 = database.read('AbacusSummit', 0.5, 'xi2', i_cosmo=box_index, tab_config='efficient')
halotab_xi4 = database.read('AbacusSummit', 0.5, 'xi4', i_cosmo=box_index, tab_config='efficient')


from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import Zheng07Cens
from halotools.empirical_models import HeavisideAssembias
from halotools.empirical_models import AssembiasZheng07Sats

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
prim_haloprop_key='halo_m200m', sec_haloprop_key='halo_nfw_conc')
sats_occ_model = AssembiasZheng07Sats(
    prim_haloprop_key='halo_m200m', sec_haloprop_key='halo_nfw_conc')

model = HodModelFactory(centrals_occupation=cens_occ_model,
                        satellites_occupation=sats_occ_model, redshift=0.5)


prior = Prior() 

prior.add_parameter('logMmin', dist=(11.0, 15.0))
prior.add_parameter('logM0', dist=(11.0, 15.0))
prior.add_parameter('logM1', dist=(11.0, 15.0))
prior.add_parameter('sigma_logM', dist=(0.1, 1.0))
prior.add_parameter('alpha', dist=(0.5, 2.0))
prior.add_parameter('f_compl', dist=(0.5, 1.0))
prior.add_parameter(
    'mean_occupation_centrals_assembias_param1', dist=(-1, +1))
prior.add_parameter('mean_occupation_satellites_assembias_param1',
                    dist=(-1, +1))
prior.add_parameter('log_eta', dist=(
    np.amin(np.log10(config['conc_gal_bias_bins'])),
    np.amax(np.log10(config['conc_gal_bias_bins']))))
if(fit == 'rsd'):
    prior.add_parameter('alpha_c', dist=(np.amin(config['alpha_c_bins']),
                                         np.amax(config['alpha_c_bins'])))
prior.add_parameter('alpha_s', dist=(np.amin(config['alpha_s_bins']),
                                     np.amax(config['alpha_s_bins'])))

#%%
def likelihood_combined(param_dict):
        
        
    if(fit == 'wpds'):
        
        model.param_dict.update(param_dict)
        ngal_wp, wp_mod = halotab_wp.predict(model, check_consistency=False)
        wp_mod = wp_mod[rp_mask_wp]
        
        ngal_ds, ds_mod = halotab_ds.predict(model, check_consistency=False)
        ds_mod = ds_mod / (1e12)
        ds_mod = ds_mod[rp_mask_ds]
        
        ngal_mod = ngal_ds

        # precision matrix to apply hartlap correction:
        pre_wp_ds = (125 - len(cov_wp_ds_arr) - 2) / (125 - 1) * np.linalg.inv(cov_wp_ds_arr)
        
        chi_sq_wp_ds = np.inner(np.inner(np.concatenate([wp_mod, ds_mod]) - np.concatenate([wp_mock, ds_mock]), 
                                         pre_wp_ds), np.concatenate([wp_mod, ds_mod]) - np.concatenate([wp_mock, ds_mock]))

        chi_sq_ngal = np.inner(np.inner(ngal_mod - ngal, np.linalg.inv(ngal_err**2)), ngal_mod - ngal)
        chi_sq_ngal = chi_sq_ngal[0, 0]
        
        total_chi_sq = chi_sq_wp_ds + chi_sq_ngal
        return total_chi_sq * -0.5
    
    elif(fit == 'rsd'):

        model.param_dict.update(param_dict)
        ngal_0, xi0_mod = halotab_xi0.predict(model, check_consistency=False)
        ngal_2, xi2_mod = halotab_xi2.predict(model, check_consistency=False)
        ngal_4, xi4_mod = halotab_xi4.predict(model, check_consistency=False)
        ngal_ds, ds_mod = halotab_ds.predict(model, check_consistency=False)
        
        all_xi_mod = np.concatenate([list(xi0_mod), list(xi2_mod), list(xi4_mod)])
        all_xi_mod = all_xi_mod[tripled_rp_mask]

        ngal_mod = ngal_ds

        # precision matrix to apply hartlap correction:
        pre_xi = (125 - len(cov_xi_arr) - 2) / (125 - 1) * np.linalg.inv(cov_xi_arr)

        chi_sq_xi = np.inner(np.inner(all_xi_mod - all_xi_mock, pre_xi), all_xi_mod - all_xi_mock)
        chi_sq_ngal = np.inner(np.inner(ngal_mod - ngal, np.linalg.inv(ngal_err**2)), ngal_mod - ngal)
        chi_sq_ngal = chi_sq_ngal[0, 0]

        total_chi_sq = chi_sq_xi + chi_sq_ngal
        return total_chi_sq * -0.5


#%%
sampler = Sampler(prior, likelihood_combined, filepath='box_'+str(box_index)+'.hdf5', n_live=2000, resume=False)  
sampler.run(f_live = 1e-10, discard_exploration=True, verbose=True)
points, log_w, log_l = sampler.posterior(return_as_dict=True)

log_l_max = np.max(log_l)
print(log_l_max)

#%%
# save best fit parameters to be used for plotting in separate scripts
points, log_w, log_l = sampler.posterior(return_as_dict=True)

if(fit == 'wpds'):
    def save_as_csv(data, filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['logM1', 'logM0', 'alpha', 'logMmin', 'sigma_logM', 'alpha_s', 
                             'log_eta', 'f_compl', 'mean_occupation_centrals_assembias_param1', 
                            'mean_occupation_satellites_assembias_param1', 'log_w', 'log_l'])
            writer.writerows(data.T)

    data = np.asarray([points['logM1'], points['logM0'], points['alpha'], points['logMmin'], 
                       points['sigma_logM'], points['alpha_s'], points['log_eta'],
                       points['f_compl'], points['mean_occupation_centrals_assembias_param1'], 
                       points['mean_occupation_satellites_assembias_param1'], log_w, log_l])
    
    print(np.max(log_l))
    save_as_csv(data, 'sampler_post_wpds.csv')
    
elif(fit == 'rsd'):
    def save_as_csv(data, filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['logM1', 'logM0', 'alpha', 'logMmin', 'sigma_logM', 'alpha_s', 
                             'log_eta', 'alpha_c', 'f_compl', 'mean_occupation_centrals_assembias_param1', 
                            'mean_occupation_satellites_assembias_param1', 'log_w', 'log_l'])
            writer.writerows(data.T)

    data = np.asarray([points['logM1'], points['logM0'], points['alpha'], points['logMmin'], 
                       points['sigma_logM'], points['alpha_s'], points['log_eta'], points['alpha_c'],
                       points['f_compl'], points['mean_occupation_centrals_assembias_param1'], 
                       points['mean_occupation_satellites_assembias_param1'], log_w, log_l])
    
    print(np.max(log_l))
    save_as_csv(data, 'sampler_post_rsd.csv')

#%%
# Plot fit here
import os
os.environ["PATH"] += ":/Library/TeX/texbin"

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = 3.33, 2.5
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
# mpl.rcParams['backend'] = 'GTK3Agg'
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['legend.fancybox'] = False
mpl.rcParams['legend.framealpha'] = 0.1
# mpl.rcParams['legend.edgecolor'] = 0.0
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

points, log_w, log_l = sampler.posterior()
best_fit_index = np.argmax(log_l)

points, log_w, log_l = sampler.posterior(return_as_dict=True)

from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import Zheng07Cens
from halotools.empirical_models import HeavisideAssembias
from halotools.empirical_models import AssembiasZheng07Sats

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
prim_haloprop_key='halo_m200m', sec_haloprop_key='halo_nfw_conc')
sats_occ_model = AssembiasZheng07Sats(
    prim_haloprop_key='halo_m200m', sec_haloprop_key='halo_nfw_conc')

model = HodModelFactory(centrals_occupation=cens_occ_model,
                        satellites_occupation=sats_occ_model, redshift=0.5)


model.param_dict['logM1'] = points['logM1'][best_fit_index]
model.param_dict['logM0'] = points['logM0'][best_fit_index]
model.param_dict['alpha'] = points['alpha'][best_fit_index]
model.param_dict['logMmin'] = points['logMmin'][best_fit_index]
model.param_dict['sigma_logM'] = points['sigma_logM'][best_fit_index]
model.param_dict['alpha_s'] = points['alpha_s'][best_fit_index]
model.param_dict['log_eta'] = points['log_eta'][best_fit_index]
if(fit == 'rsd'):
    model.param_dict['alpha_c'] = points['alpha_c'][best_fit_index]
model.param_dict['f_compl'] = points['f_compl'][best_fit_index]
model.param_dict['mean_occupation_centrals_assembias_param1'] = points['mean_occupation_centrals_assembias_param1'][best_fit_index]
model.param_dict['mean_occupation_satellites_assembias_param1'] = points['mean_occupation_satellites_assembias_param1'][best_fit_index]

halotab_wp = database.read('AbacusSummit', 0.5, 'wp', i_cosmo=box_index, tab_config='efficient')
halotab_ds = database.read('AbacusSummit', 0.5, 'ds', i_cosmo=box_index, tab_config='efficient')
halotab_xi0 = database.read('AbacusSummit', 0.5, 'xi0', i_cosmo=box_index, tab_config='efficient')
halotab_xi2 = database.read('AbacusSummit', 0.5, 'xi2', i_cosmo=box_index, tab_config='efficient')
halotab_xi4 = database.read('AbacusSummit', 0.5, 'xi4', i_cosmo=box_index, tab_config='efficient')

if(fit == 'wpds'):
    
    wp_mock = table['wp'].data
    ds_mock = table['ds'].data
    ds_mock_cut = ds_mock[:13]
    rp_ave_cut = rp_ds_bins[:13]
    
    cov_mock_wp = mock_cov_all.iloc[:14, :14]
    cov_wp_arr = cov_mock_wp.to_numpy()
    cov_mock_ds = (mock_cov_all.iloc[14:27, 14:27])
    cov_ds_arr = cov_mock_ds.to_numpy()
    
    
    wp_mod_best_fit = halotab_wp.predict(model, check_consistency=False)[1]
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}) 
    ax1.axvspan(0, rp_lim_wp, facecolor='gray', alpha=0.5)
    ax2.axvspan(0, rp_lim_wp, facecolor='gray', alpha=0.5)

    ax1.errorbar(rp_wp_bins, rp_wp_bins * wp_mock, yerr= rp_wp_bins * np.sqrt(np.diag(cov_wp_arr)), fmt='.', color='blue', capsize=3, capthick=0.5)
    ax1.plot(rp_wp_bins, rp_wp_bins * wp_mod_best_fit, c='blue')
    ax1.set_xlabel(r'$r_p \ [h^{-1} \ \mathrm{Mpc}]$')
    ax1.set_ylabel(r'$r_p \times w_p \ [h^{-2} \ \mathrm{Mpc^2}]$')
    ax1.set_xscale('log')
    ax1.set_xticklabels([])
    
    delta_wp = np.asarray(wp_mock) - np.asarray(wp_mod_best_fit)
    sigma_wp = np.sqrt(np.diag(cov_wp_arr))
    d_wp = delta_wp/sigma_wp
    ax2.errorbar(rp_wp_bins, d_wp, yerr = 1, fmt='.', color='blue')
    ax2.plot(rp_wp_bins, np.zeros(len(rp_wp_bins)), ls='--', color='black')
    ax2.set_ylim(-3, 3)
    ax2.set_ylabel(r'$\delta w_p$')
    ax2.set_xscale('log')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
    ds_mod_best_fit = halotab_ds.predict(model, check_consistency=False)[1] / 1e12
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}) 
    ax1.axvspan(0, rp_lim_ds, facecolor='gray', alpha=0.5)
    ax2.axvspan(0, rp_lim_ds, facecolor='gray', alpha=0.5)
    
    ax1.errorbar(rp_ave_cut, rp_ave_cut * ds_mock[:13], yerr=rp_ave_cut * np.sqrt(np.diag(cov_ds_arr)), fmt='.', color='blue', capsize=5, capthick=0.5)
    ax1.plot(rp_ave_cut, rp_ave_cut * ds_mod_best_fit[:13], c='blue')
    ax1.set_xlabel(r'$r_{p} \ [h^{-1} \ \mathrm{Mpc}]$')
    ax1.set_ylabel(r'$r_{p} \times \Delta\Sigma \ [10^6 \, M_\odot / \mathrm{pc}]$')
    ax1.set_xscale('log')
    ax1.set_xticklabels([])
    
    d_ds = (np.asarray(ds_mock[:13]) - np.asarray(ds_mod_best_fit))/np.sqrt(np.diag(cov_ds_arr))
    ax2.errorbar(rp_ave_cut, d_ds, yerr = 1, fmt='.', color='blue')
    ax2.plot(rp_ave_cut, np.zeros(len(rp_ave_cut)), ls='--', color='black')
    ax2.set_ylim(-3, 3)
    ax2.set_ylabel(r'$\delta\Delta\Sigma$')
    ax2.set_xscale('log')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
elif(fit == 'rsd'):
    
    xi0_mock = table['xi0'].data
    xi2_mock = table['xi2'].data
    xi4_mock = table['xi4'].data
    
    cov_mock_xi0 = mock_cov_all.iloc[27:41, 27:41]
    cov_mock_xi2 = mock_cov_all.iloc[41:55, 41:55]
    cov_mock_xi4 = mock_cov_all.iloc[55:69, 55:69]
    cov_xi0_arr = cov_mock_xi0.to_numpy()
    cov_xi2_arr = cov_mock_xi2.to_numpy()
    cov_xi4_arr = cov_mock_xi4.to_numpy()
    
    xi0_mod_best_fit = halotab_xi0.predict(model, check_consistency=False)[1]
    xi2_mod_best_fit = halotab_xi2.predict(model, check_consistency=False)[1]
    xi4_mod_best_fit = halotab_xi4.predict(model, check_consistency=False)[1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}) 
    
    ax1.axvspan(0, rp_lim_xi, facecolor='gray', alpha=0.5)
    ax2.axvspan(0, rp_lim_xi, facecolor='gray', alpha=0.5)

    ax1.errorbar(s_bins, rp_wp_bins**1.5 * xi0_mock, yerr= rp_wp_bins**1.5 *np.sqrt(np.diag(cov_xi0_arr)), fmt='.', color='blue', capsize=5, capthick=0.5)
    ax1.plot(s_bins, rp_wp_bins**1.5 * xi0_mod_best_fit, c='blue', label=r'$\xi_0$')
    ax1.errorbar(s_bins, rp_wp_bins**1.5 * xi2_mock, yerr= rp_wp_bins**1.5 *np.sqrt(np.diag(cov_xi2_arr)), fmt='.', color='purple', capsize=5, capthick=0.5)
    ax1.plot(s_bins, rp_wp_bins**1.5 * xi2_mod_best_fit, c='purple', label=r'$\xi_2$')
    ax1.errorbar(s_bins, rp_wp_bins**1.5 * xi4_mock, yerr= rp_wp_bins**1.5 *np.sqrt(np.diag(cov_xi4_arr)), fmt='.', color='magenta', capsize=5, capthick=0.5)
    ax1.plot(s_bins, rp_wp_bins**1.5 * xi4_mod_best_fit, c='magenta', label=r'$\xi_4$')
    ax1.set_xlabel(r'$s \ [h^{-1} \ \mathrm{Mpc}]$')
    ax1.set_ylabel(r'$r^{1.5} \times\xi \ [h^{-1.5} \ \mathrm{Mpc^{1.5}}]$')
    ax1.set_xscale('log')
    ax1.set_xticklabels([])
    ax1.annotate(r'$\xi_0$', [0.65, -10], color = 'blue')
    ax1.annotate(r'$\xi_2$', [1, -10], color = 'purple')
    ax1.annotate(r'$\xi_4$', [1.5, -10], color = 'magenta')

    d_xi0 = (np.asarray(xi0_mock) - np.asarray(xi0_mod_best_fit))/np.sqrt(np.diag(cov_xi0_arr))
    d_xi2 = (np.asarray(xi2_mock) - np.asarray(xi2_mod_best_fit))/np.sqrt(np.diag(cov_xi2_arr))
    d_xi4 = (np.asarray(xi4_mock) - np.asarray(xi4_mod_best_fit))/np.sqrt(np.diag(cov_xi4_arr))

    ax2.errorbar(s_bins, d_xi0, yerr = 1, fmt='.', color='blue')
    ax2.errorbar(s_bins, d_xi2, yerr = 1, fmt='.', color='purple')
    ax2.errorbar(s_bins, d_xi4, yerr = 1, fmt='.', color='magenta')
    ax2.set_ylabel(r'$\delta\xi$')
    ax2.plot(s_bins, np.zeros(len(rp_wp_bins)), ls='--', color='black')
    ax2.set_ylim(-3, 3)
    ax2.set_xlabel(r'$s \ [h^{-1} \ \mathrm{Mpc}]$')
    ax2.set_xscale('log')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()



#%%
fig.savefig('mass_only_rsd_best.png', bbox_inches='tight')