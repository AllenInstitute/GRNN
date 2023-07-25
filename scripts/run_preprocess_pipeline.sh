#!/bin/bash
#SBATCH --job-name=PREPROCESS
#SBATCH -N1 # Run this across 1 nodes
#SBATCH -n1 # Run 1 tasks
#SBATCH --mail-user=calvin.yeung@alleninstitute.org # Send mail to your AI email
#SBATCH --mail-type=BEGIN,END,FAIL # Send mail on begin, end, fail
#SBATCH -p braintv
#SBATCH --mem=24gb                     # Job memory request (per node)
#SBATCH --time=36:00:00               # Time limit hrs:min:sec
#SBATCH -o ./output/output_preprocess_%j.out # output goes to gethostname_<JOBID>.out
#SBATCH -e ./output/error_preprocess_%j.err # error goes to gethostname_<JOBID>.err
source /home/calvin.yeung/.bash_profile
source /home/calvin.yeung/.bashrc
source /allen/ai/hpc/shared/utils.x86_64/anaconda3-2021.11/bin/activate /home/calvin.yeung/.conda/envs/pytorch-gpu
sbatch python3 -u preprocess_pipeline.py $1