#!/bin/bash
#SBATCH --partition=lmc
#SBATCH --account=RB230035
#SBATCH --job-name=test
#SBATCH -o /home/haruka74/Adnest/results/%j.o
#SBATCH -e /home/haruka74/Adnest/results/%j.e
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=96
#SBATCH --nodes=2
#SBATCH --mem=2880G
#SBATCH --time=01:00:00
#SBATCH --exclusive

source ~/.bashrc
micromamba activate humam
# echo $PATH
# which python3
# which python

snakemake -s lmcSnakefile --cores 192

