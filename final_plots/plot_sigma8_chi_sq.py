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

### limited array with cosmologies that vary sigma8: 
import tabcorr

box_vary_sigma8 = [0, 4, 112, 113, 116, 125, 126]
vary_sigma8_val = []
vary_sigma8_chi_sq_wp_ds = []
vary_sigma8_chi_sq_rsd = []

for b in box_vary_sigma8:
    cosmo = tabcorr.database.cosmology('AbacusSummit', b)
    sigma_8 = cosmo.sigma_8
    vary_sigma8_val.append(sigma_8)
    
    f = open(folderpath_wp_ds+"output"+str(b)+".txt", "r")
    line1 = f.readline()
    line2 = f.readline()
    max_likelihood = float(line1[15:])
    vary_sigma8_chi_sq_wp_ds.append(-2.0 * max_likelihood)
    
    f = open(folderpath_rsd+"output"+str(b)+".txt", "r")
    line1 = f.readline()
    line2 = f.readline()
    max_likelihood = float(line1[15:])
    vary_sigma8_chi_sq_rsd.append(-2.0 * max_likelihood)

vary_sigma8 = np.asarray(vary_sigma8_val)
vary_sigma8_chisq_wp_ds = np.asarray(vary_sigma8_chi_sq_wp_ds)
vary_sigma8_chisq_rsd = np.asarray(vary_sigma8_chi_sq_rsd)

plt.scatter(vary_sigma8, vary_sigma8_chisq_wp_ds, label='$w_p+\Delta\Sigma$', color='red')
plt.scatter(vary_sigma8, vary_sigma8_chisq_rsd, label='RSDs', color='blue')
plt.xlabel('$\sigma_8$')
plt.ylabel('$\chi^2$')
plt.yscale('log')
plt.legend()
plt.title('$\chi^2$ vs $\sigma_8$')
plt.show()

