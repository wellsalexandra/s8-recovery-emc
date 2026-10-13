import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table
from nautilus import Sampler

import jerboa

parser = argparse.ArgumentParser(
    description='Fit a single AbacusSummit simulation to one data set.')
parser.add_argument('observations', help='observations to fit')
parser.add_argument('assembly_bias_mock', help='mock with assembly bias',
                    type=bool)
parser.add_argument('assembly_bias_model', help='model with assembly bias',
                    type=bool)
parser.add_argument('cosmology', help='AbacusSummit cosmology', type=int)
parser.add_argument('--n_jobs', help='how many processes to use', default=1,
                    type=int)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

prior = jerboa.prior(args.observations, args.assembly_bias_model)
obs = jerboa.observations(args.observations, args.assembly_bias_mock)
model = jerboa.model()
halotab = jerboa.halotab(args.observations, args.cosmology)


def likelihood(param_dict):

    model.param_dict.update(dict(param_dict))

    mod = {}

    for key in halotab.keys():
        n, xi = halotab[key].predict(model, check_consistency=False)
        if key == 'ds':
            xi /= 1e12
        if key in ['wp', 'xi0', 'xi2', 'xi4']:
            mod['n'] = n
        mod[key] = xi

    theta_obs = jerboa.dict_to_vector(**obs)
    theta_mod = jerboa.dict_to_vector(**(obs | mod))
    theta_diff = theta_obs - theta_mod
    chi_sq = np.inner(np.inner(theta_diff, obs['pre']), theta_diff)
    chi_sq += (obs['n'] - mod['n'])**2 / obs['n_err']**2

    return -chi_sq / 2, mod


if __name__ == '__main__':
    sampler = Sampler(prior, likelihood, pool=args.n_jobs, n_live=3000)
    sampler.run(f_live=1e-10, discard_exploration=True, verbose=args.verbose)
    points, log_w, log_l, blobs = sampler.posterior(
        return_as_dict=True, return_blobs=True)
    table = Table(points)
    for key in blobs[0].keys():
        table[key] = [b[key] for b in blobs]
    for key in table.colnames:
        table.meta[key] = table[key][np.argmax(log_l)]
    table['log_l'] = log_l
    table['log_w'] = log_w
    table.meta['log Z'] = sampler.log_z
    table.meta['log L_max'] = np.amax(log_l)

    w_min = 1e-5
    p = np.exp(log_w) / w_min
    table = table[np.random.random(len(table)) < p]
    table['log_w'] = np.maximum(table['log_w'], np.log(w_min))

    path = Path(
        f"{args.observations}{'_mock_gab' if args.assembly_bias_mock else ''}"
        f"{'_model_gab' if args.assembly_bias_model else ''}")
    path.mkdir(exist_ok=True)
    table.write(path / f"{args.cosmology}.hdf5", overwrite=True,
                path='posterior')
