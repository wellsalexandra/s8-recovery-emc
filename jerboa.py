from pathlib import Path

import numpy as np
from astropy.table import Table
from halotools.empirical_models import AssembiasZheng07Sats, HeavisideAssembias
from halotools.empirical_models import Zheng07Cens, HodModelFactory
from nautilus import Prior
from tabcorr import database

BASE_PATH = Path(__file__).absolute().parent
FITS_PATH = BASE_PATH / 'fits'
MOCK_PATH = BASE_PATH / 'mock'
WP_MIN = 1.0
DS_MIN = 2.5
XI_MIN = 2.5


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
            **kwargs
        )


def model():
    return HodModelFactory(
        centrals_occupation=AssembiasIncompleteZheng07Cens(),
        satellites_occupation=AssembiasZheng07Sats())


def dict_to_vector(**kwargs):
    theta = []
    for key in ['wp', 'ds', 'xi0', 'xi2', 'xi4']:
        if np.any(kwargs[f'{key}_use']):
            theta.append(kwargs[key][kwargs[f'{key}_use']])
    return np.concatenate(theta)


def observations(observations, assembly_bias):

    if observations not in ['ggl', 'rsd']:
        msg = f"`data` must be 'ggl' or 'rsd'. Got {observations}."
        raise ValueError(msg)

    obs = dict()
    suffix = '_mass_conc' if assembly_bias else '_mass'

    table = Table.read(MOCK_PATH / f'mock_model1{suffix}.fits')

    for key in ['wp', 'ds', 'xi0', 'xi2', 'xi4']:
        obs[key] = table[key].data
    obs['ds'] = obs['ds'][:-1]

    for key, r_min, tabcorr_key in zip(
            ['wp', 'ds', 'xi0', 'xi2', 'xi4'],
            [WP_MIN, DS_MIN, XI_MIN, XI_MIN, XI_MIN],
            ['rp_wp_bins', 'rp_ds_bins', 's_bins', 's_bins', 's_bins']):
        r = database.configuration('efficient')[tabcorr_key]
        obs[f'{key}_use'] = np.sqrt(r[1:] * r[:-1]) > r_min

    obs['n'] = table.meta['N']
    obs['n_err'] = table.meta['N_ERR']

    keys = ['wp', 'ds'] if observations == 'rsd' else ['xi0', 'xi2', 'xi4']
    for key in keys:
        obs[f'{key}_use'][:] = False

    use = np.concatenate([
        obs[f'{key}_use'] for key in ['wp', 'ds', 'xi0', 'xi2', 'xi4']])

    cov = np.genfromtxt(MOCK_PATH / f'cov_model1{suffix}.txt')

    i = 0
    for key, n in zip(['wp', 'ds', 'xi0', 'xi2', 'xi4'], [14, 13, 14, 14, 14]):
        obs[f'{key}_err'] = np.sqrt(np.diag(cov[i:i + n][:, i:i + n]))
        i += n

    obs['cov'] = cov[use, :][:, use]
    obs['pre'] = (125 - len(obs['cov']) - 2) / (125 - 1) * np.linalg.inv(
        obs['cov'])

    return obs


def prior(observations, assembly_bias):
    prior = Prior()
    config = database.configuration('efficient')
    prior.add_parameter('logMmin', dist=(11.0, 15.0))
    prior.add_parameter('logM0', dist=(11.0, 15.0))
    prior.add_parameter('logM1', dist=(11.0, 15.0))
    prior.add_parameter('sigma_logM', dist=(0.1, 1.0))
    prior.add_parameter('alpha', dist=(0.5, 2.0))
    prior.add_parameter('f_compl', dist=(0.5, 1.0))
    prior.add_parameter(
        'mean_occupation_centrals_assembias_param1',
        dist=(-1, +1) if assembly_bias else 0)
    prior.add_parameter(
        'mean_occupation_satellites_assembias_param1',
        dist=(-1, +1) if assembly_bias else 0)
    prior.add_parameter('log_eta', dist=(
        np.amin(np.log10(config['conc_gal_bias_bins'])),
        np.amax(np.log10(config['conc_gal_bias_bins']))))
    if observations == 'rsd':
        prior.add_parameter('alpha_c', dist=(np.amin(config['alpha_c_bins']),
                                             np.amax(config['alpha_c_bins'])))
    prior.add_parameter('alpha_s', dist=(np.amin(config['alpha_s_bins']),
                                         np.amax(config['alpha_s_bins'])))
    return prior


def halotab(observations, cosmology):
    keys = ['wp', 'ds'] if observations == 'ggl' else ['xi0', 'xi2', 'xi4']
    return {key: database.read(
        'AbacusSummit', 0.5, key, tab_config='efficient', i_cosmo=cosmology)
        for key in keys}


def statistics(observations, assembly_bias_mock, assembly_bias_model):
    rows = []
    for cosmology in [0, 1, 4, 13, 100, 101, 102, 103, 104, 105, 112, 113, 116,
                      117, 118, 119, 120, 125, 126, 130, 131, 132, 133, 134,
                      135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
                      146]:
        cosmo = database.cosmology('AbacusSummit', cosmology)
        path = (FITS_PATH /
                (f"{observations}{'_mock_gab' if assembly_bias_mock else ''}"
                 f"{'_model_gab' if assembly_bias_model else ''}") /
                f"{cosmology}.hdf5")
        meta = dict(Table.read(path).meta)
        rows.append(meta | dict(
            O_m=cosmo.Om0, sigma_8=cosmo.sigma8, n_s=cosmo.ns,
            H_0=cosmo.H0, O_b=cosmo.Ob0, N_eff=cosmo.Neff,
            sum_m_nu=np.sum(cosmo.m_nu), w_0=cosmo.w0, w_a=cosmo.wa,
            cosmology=cosmology))
    table = Table(rows)
    table['S_8'] = table['sigma_8'] * np.sqrt(table['O_m'] / 0.3)
    return table
