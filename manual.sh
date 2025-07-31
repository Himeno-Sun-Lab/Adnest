#!/bin/bash
#SBATCH --partition=lmc
#SBATCH --account=RB230035
#SBATCH --job-name=test
#SBATCH -o /home/haruka74/Adnest/results/%j.o
#SBATCH -e /home/haruka74/Adnest/results/%j.e
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=6
#SBATCH --nodes=2
#SBATCH --mem=2880G
#SBATCH --time=02:00:00
#SBATCH --exclusive

source ~/.bashrc
micromamba activate humam
# echo $PATH
# which python3
# which python
source $HOME/nest/3.8/python3.12/bin/nest_vars.sh
srun python src/snakemake_simulation.py experiments/Abeta_default.py experiments/Abeta_default.network_hash experiments/Abeta_default.simulation_hash 192

