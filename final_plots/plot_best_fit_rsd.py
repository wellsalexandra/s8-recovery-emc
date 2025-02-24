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

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
# mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.figsize'] = 3.33, 2.5
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
# mpl.rcParams['backend'] = 'GTK3Agg'
print(mpl.rcParams['backend'].__init__)
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

folderpath_wp_ds = './fit_results/new_mock_wp_ds_Abacus/'
folderpath_rsd = './fit_results/new_mock_rsd_Abacus/'

table_rsd = Table.read('sampler_post_rsd.csv')
log_l = table_rsd['log_l']
log_w = table_rsd['log_w']
w = np.exp(log_w)
points = [table_rsd['logM1'], table_rsd['logM0'], table_rsd['alpha'], table_rsd['logMmin'], 
                   table_rsd['sigma_logM'], table_rsd['alpha_s'], table_rsd['log_eta'], table_rsd['alpha_c'],
                   table_rsd['f_compl'], table_rsd['mean_occupation_centrals_assembias_param1'], 
                   table_rsd['mean_occupation_satellites_assembias_param1']]


points = np.asarray(points).T
best_fit_index = np.argmax(log_l)
best_fit_params = points[best_fit_index]

print(np.max(log_l))
print(best_fit_index)
print(best_fit_params)

box_index = 103 # Best fit cosmology for RSDs
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

s_min = df['s_min']
s_max = df['s_max']
s_ave = 0.5 * (s_min + s_max)

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
model.param_dict['alpha_c'] = best_fit_params[7] # for RSDs
model.param_dict['f_compl'] = best_fit_params[8]
model.param_dict['mean_occupation_centrals_assembias_param1'] = best_fit_params[9]
model.param_dict['mean_occupation_satellites_assembias_param1'] = best_fit_params[10]


# code for plotting rsd multipoles
xi0_mod_best_fit = halotab_xi0.predict(model, check_consistency=False)[1]
xi2_mod_best_fit = halotab_xi2.predict(model, check_consistency=False)[1]
xi4_mod_best_fit = halotab_xi4.predict(model, check_consistency=False)[1]

cov_mock_xi0 = mock_cov_all.iloc[27:41, 27:41]
cov_mock_xi2 = mock_cov_all.iloc[41:55, 41:55]
cov_mock_xi4 = mock_cov_all.iloc[55:69, 55:69]
cov_xi0_arr = cov_mock_xi0.to_numpy()
cov_xi2_arr = cov_mock_xi2.to_numpy()
cov_xi4_arr = cov_mock_xi4.to_numpy()

# new mock data
mock_measurements = pd.read_csv('./mocks/mock_model1.csv', sep = ',')
xi0_mock = pd.Series(list(mock_measurements['xi0'])).values
xi2_mock = pd.Series(list(mock_measurements['xi2'])).values
xi4_mock = pd.Series(list(mock_measurements['xi4'])).values


ax = plt.gca()  
# s_lim = np.sqrt(np.pi**2 + 2.5**2)
s_lim = 2.5
ax.axvspan(0, s_lim, facecolor='gray', alpha=0.5)

# s = np.sqrt(np.pi**2 + (rp_ave.values)**2)

plt.errorbar(s_ave, rp_ave**1.5 * xi0_mock, yerr= rp_ave**1.5 *np.sqrt(np.diag(cov_xi0_arr)), fmt='.', color='blue', capsize=5, capthick=0.5)
plt.plot(s_ave, rp_ave**1.5 * xi0_mod_best_fit, c='blue', label=r'$\xi_0$')
plt.errorbar(s_ave, rp_ave**1.5 * xi2_mock, yerr= rp_ave**1.5 *np.sqrt(np.diag(cov_xi2_arr)), fmt='.', color='purple', capsize=5, capthick=0.5)
plt.plot(s_ave, rp_ave**1.5 * xi2_mod_best_fit, c='purple', label=r'$\xi_2$')
plt.errorbar(s_ave, rp_ave**1.5 * xi4_mock, yerr= rp_ave**1.5 *np.sqrt(np.diag(cov_xi4_arr)), fmt='.', color='magenta', capsize=5, capthick=0.5)
plt.plot(s_ave, rp_ave**1.5 * xi4_mod_best_fit, c='magenta', label=r'$\xi_4$')
plt.xlabel(r'$s \ [h^{-1} \ \mathrm{Mpc}]$')
plt.ylabel(r'$r^{1.5}\xi \ [h^{-1.5} \ \mathrm{Mpc^{1.5}}]$')
plt.xscale('log')
# plt.yscale('log')
plt.annotate(r'$\xi_0$', [0.65, -10], color = 'blue')
plt.annotate(r'$\xi_2$', [1, -10], color = 'purple')
plt.annotate(r'$\xi_4$', [1.5, -10], color = 'magenta')
# plt.legend(fontsize=12, frameon=False)
# plt.title('Best Fit of $w_p$', fontsize=15)
plt.show()
