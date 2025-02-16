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

pip install --user astroML halotools pytest-astropy nautilus-sampler corner
pip install git+https://github.com/johannesulf/TabCorr.git --ignore-installed --no-deps


# run in parallel
srun -n1 python fit_data_ab.py 0 2.5 1.0 &
srun -n1 python fit_data_ab.py 1 2.5 1.0 &
srun -n1 python fit_data_ab.py 4 2.5 1.0 &
srun -n1 python fit_data_ab.py 13 2.5 1.0 &
srun -n1 python fit_data_ab.py 100 2.5 1.0 &
srun -n1 python fit_data_ab.py 101 2.5 1.0 &
srun -n1 python fit_data_ab.py 102 2.5 1.0 &
srun -n1 python fit_data_ab.py 103 2.5 1.0 &
srun -n1 python fit_data_ab.py 104 2.5 1.0 &
srun -n1 python fit_data_ab.py 105 2.5 1.0 &
srun -n1 python fit_data_ab.py 112 2.5 1.0 &
srun -n1 python fit_data_ab.py 113 2.5 1.0 &
srun -n1 python fit_data_ab.py 116 2.5 1.0 &
srun -n1 python fit_data_ab.py 117 2.5 1.0 &
srun -n1 python fit_data_ab.py 118 2.5 1.0 &
srun -n1 python fit_data_ab.py 119 2.5 1.0 &
srun -n1 python fit_data_ab.py 120 2.5 1.0 &
srun -n1 python fit_data_ab.py 125 2.5 1.0 &
srun -n1 python fit_data_ab.py 126 2.5 1.0 &
srun -n1 python fit_data_ab.py 130 2.5 1.0 &
srun -n1 python fit_data_ab.py 131 2.5 1.0 &
srun -n1 python fit_data_ab.py 132 2.5 1.0 &
srun -n1 python fit_data_ab.py 133 2.5 1.0 &
srun -n1 python fit_data_ab.py 134 2.5 1.0 &
srun -n1 python fit_data_ab.py 135 2.5 1.0 &
srun -n1 python fit_data_ab.py 136 2.5 1.0 &
srun -n1 python fit_data_ab.py 137 2.5 1.0 &
srun -n1 python fit_data_ab.py 138 2.5 1.0 &
srun -n1 python fit_data_ab.py 139 2.5 1.0 &
srun -n1 python fit_data_ab.py 140 2.5 1.0 &
srun -n1 python fit_data_ab.py 141 2.5 1.0 &
srun -n1 python fit_data_ab.py 142 2.5 1.0 &
srun -n1 python fit_data_ab.py 143 2.5 1.0 &
srun -n1 python fit_data_ab.py 144 2.5 1.0 &
srun -n1 python fit_data_ab.py 145 2.5 1.0 &
srun -n1 python fit_data_ab.py 146 2.5 1.0 &

wait
