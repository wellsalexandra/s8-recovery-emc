#!/bin/bash
#SBATCH --partition=parallel
#SBATCH --account=PHS0336
#SBATCH --job-name=test_job
#SBATCH --nodes=3
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=my_first_job.out

module load python/3.9-2022.05
source /path/to/your/virtualenv/bin/activate

pip install --user astroML halotools pytest-astropy tabcorr nautilus-sampler corner

# run in parallel
srun -n1 python fit_data_test.py 0 2.0 &
srun -n1 python fit_data_test.py 1 2.0 &
srun -n1 python fit_data_test.py 2 2.0 &
srun -n1 python fit_data_test.py 3 2.0 &
srun -n1 python fit_data_test.py 4 2.0 &
srun -n1 python fit_data_test.py 5 2.0 &
srun -n1 python fit_data_test.py 6 2.0 &
srun -n1 python fit_data_test.py 7 2.0 &
srun -n1 python fit_data_test.py 8 2.0 &
srun -n1 python fit_data_test.py 9 2.0 &
srun -n1 python fit_data_test.py 10 2.0 &
srun -n1 python fit_data_test.py 11 2.0 &
srun -n1 python fit_data_test.py 12 2.0 &
srun -n1 python fit_data_test.py 13 2.0 &
srun -n1 python fit_data_test.py 14 2.0 &
srun -n1 python fit_data_test.py 15 2.0 &
srun -n1 python fit_data_test.py 16 2.0 &
srun -n1 python fit_data_test.py 17 2.0 &
srun -n1 python fit_data_test.py 18 2.0 &
srun -n1 python fit_data_test.py 19 2.0 &
srun -n1 python fit_data_test.py 20 2.0 &
srun -n1 python fit_data_test.py 21 2.0 &
srun -n1 python fit_data_test.py 22 2.0 &
srun -n1 python fit_data_test.py 23 2.0 &
srun -n1 python fit_data_test.py 24 2.0 &
srun -n1 python fit_data_test.py 25 2.0 &
srun -n1 python fit_data_test.py 26 2.0 &
srun -n1 python fit_data_test.py 27 2.0 &
srun -n1 python fit_data_test.py 28 2.0 &
srun -n1 python fit_data_test.py 29 2.0 &
srun -n1 python fit_data_test.py 30 2.0 &
srun -n1 python fit_data_test.py 31 2.0 &
srun -n1 python fit_data_test.py 32 2.0 &
srun -n1 python fit_data_test.py 33 2.0 &
srun -n1 python fit_data_test.py 34 2.0 &
srun -n1 python fit_data_test.py 35 2.0 &
srun -n1 python fit_data_test.py 36 2.0 &
srun -n1 python fit_data_test.py 37 2.0 &
srun -n1 python fit_data_test.py 38 2.0 &
srun -n1 python fit_data_test.py 39 2.0 &

wait
