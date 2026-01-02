## This repository details an analysis pipeline used for extracting cosmological parameters from mock datasets of projected galaxy clustering, redshift-space clustering, and galaxy-galaxy lensing.

## Guide for using provided scripts: (*Mock data sets already obtained*)

**Step 1:** Run fitting script to compute likelihood values for each cosmology. See **likelihood_over_cosmologies** folder. Use **submit_job_ab_loop.sh** to run fitting script (**run_fit_ab.py**) over all applicable cosmologies for analysis (provided code uses LCDM cosmologies from AbacusSummit simulations). This will output max (log)likelihood values for each cosmology. 

**Step 2:** Take likelihood values from individual files to all in one file (**process_mock_fit_results.py**) and also extract what the best fit cosmology was for both wp+DS and RSD analyses.

**Step 3:** Use likelihood values to fit cosmological parameters and. 

