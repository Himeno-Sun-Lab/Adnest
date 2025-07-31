#!/bin/bash
#SBATCH --partition=mpc
#SBATCH --account=RB240007
#SBATCH --job-name=mpctest
#SBATCH -o /home/haruka74/Adnest/results/%j.o
#SBATCH -e /home/haruka74/Adnest/results/%j.e
#SBATCH --cpus-per-task=28
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=32
#SBATCH --mem=112G
#SBATCH --time=10:00:00
#SBATCH --exclusive

source ~/.bashrc
micromamba activate humam
# echo $PATH
# which python3
# which python

snakemake -s mpcSnakefile

