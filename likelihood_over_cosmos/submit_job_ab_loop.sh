#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=PHS0336
#SBATCH --job-name=test_job
#SBATCH --nodes=3
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=my_first_job.out

module load python/3.12
source ~/myenv/bin/activate


pip install numpy pandas matplotlib astroML halotools pytest-astropy nautilus-sampler corner weightedstats
#pip install git+https://github.com/johannesulf/TabCorr.git --ignore-installed --no-deps
pip install git+https://github.com/johannesulf/TabCorr.git

# Define the list of parameter values
indices=(0 1 4 13 100 101 102 103 104 105 112 113 116 117 118 119 120 125 126 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146)

# Run in parallel using a loop
for idx in "${indices[@]}"; do
    srun -n1 python run_fit_ab.py "$idx" "wpds" &
    srun -n1 python run_fit_ab.py "$idx" "rsd" &
done

wait
