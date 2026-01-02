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
from astropy.table import Table
import h5py
import corner
import os



os.environ['TABCORR_DATABASE'] = './tabcorr'

parser = argparse.ArgumentParser(description="Fit observational data for a specified Box index")
parser.add_argument('box_index', type=int, help='The Box index to fit')
parser.add_argument('data_fit', type=str, help='wpds or rsd')
args = parser.parse_args()
box_index = args.box_index
fit = args.data_fit

#box_index = 0
#fit = 'wpds' # options: wpds or rsd

rp_lim_ds = 2.5
rp_lim_xi = 2.5
rp_lim_wp = 1.0

config = database.configuration('efficient')
s_bins = np.sqrt(config['s_bins'][1:] * config['s_bins'][:-1])
rp_wp_bins = np.sqrt(config['rp_wp_bins'][1:] * config['rp_wp_bins'][:-1])
rp_ds_bins = np.sqrt(config['rp_ds_bins'][1:] * config['rp_ds_bins'][:-1])



# mock data
#mock_cov_all = pd.read_csv('./mocks/cov_model1_mass.csv', header=None)
mock_cov_all = pd.read_csv('./mocks/cov_model1.csv', header=None)
cov_xi_arr = np.array(mock_cov_all.iloc[27:69, 27:69])
cov_wp_ds_arr = np.array(mock_cov_all.iloc[:27, :27])

table = Table.read('mock_model1.fits')
#table = Table.read('mock_model1_mass.fits')
wp_mock = table['wp'].data
ds_mock = table['ds'].data
all_xi_mock = np.concatenate([table['xi0'].data,
                              table['xi2'].data,
                              table['xi4'].data])

ngal = table.meta['N']
ngal = np.reshape(ngal, [1, 1])
ngal_err = table.meta['N_ERR']
ngal_err = np.reshape(ngal_err, [1, 1])


#wp_mock = mock_measurements['wp'].values
#ds_mock = mock_measurements['ds'].values # ds_mock = ds_mock / 1e12
#all_xi_mock = np.concatenate([list(mock_measurements['xi0']),
#                              list(mock_measurements['xi2']),
#                              list(mock_measurements['xi4'])])


## number density of galaxies
#ngal_info = pd.read_csv('ngal.csv')
#ngal = ngal_info['ngal'].values
#ngal = np.reshape(ngal, [1, 1])
#ngal_err = ngal_info['ngal_err'].values
#ngal_err = np.reshape(ngal_err, [1, 1])

# apply scale cuts to data:

# First cut 14th element from lensing, for rp_ave_ds and ds_mock
# rp_ave_cut = rp_ave[:13]
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



halotab_wp = database.read('AbacusSummit', 0.5, 'wp', i_cosmo=box_index, tab_config='efficient')
halotab_ds = database.read('AbacusSummit', 0.5, 'ds', i_cosmo=box_index, tab_config='efficient')
halotab_xi0 = database.read('AbacusSummit', 0.5, 'xi0', i_cosmo=box_index, tab_config='efficient')
halotab_xi2 = database.read('AbacusSummit', 0.5, 'xi2', i_cosmo=box_index, tab_config='efficient')
halotab_xi4 = database.read('AbacusSummit', 0.5, 'xi4', i_cosmo=box_index, tab_config='efficient')


from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import Zheng07Cens
# from halotools.empirical_models import Zheng07Sats
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
    prior.add_parameter('alpha_c', dist=(np.amin(config['alpha_c_bins']), np.amax(config['alpha_c_bins'])))
prior.add_parameter('alpha_s', dist=(np.amin(config['alpha_s_bins']), np.amax(config['alpha_s_bins'])))

max_likelihoods = []
evidences = []
chi_squareds = []
                                     

def likelihood_combined(param_dict):
        
        
    if(fit == 'wpds'):
        
        model.param_dict.update(param_dict)
        ngal_wp, wp_mod = halotab_wp.predict(model, check_consistency=False)
        wp_mod = wp_mod[rp_mask_wp]
        
        ngal_ds, ds_mod = halotab_ds.predict(model, check_consistency=False)
        ds_mod = ds_mod / (1e12)
        ds_mod = ds_mod[rp_mask_ds]
        
        ngal_mod = ngal_ds


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


sampler = Sampler(prior, likelihood_combined, filepath=str(fit)+'_box_'+str(box_index)+'.hdf5', n_live=2000, resume=False)
sampler.run(f_live = 1e-10, discard_exploration=True, verbose=True)


points, log_w, log_l = sampler.posterior()
log_l_max = np.max(log_l)

max_likelihoods.append(log_l_max)
evidences.append(sampler.log_z)
file = open(str(fit)+'_output'+str(box_index)+'.txt', 'w')
    
file.write(f"Max Likelihood: {max_likelihoods[0]}\n")
file.write(f"Evidence: {evidences[0]}\n")
file.close()


