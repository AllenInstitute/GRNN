#!/bin/bash
#SBATCH --job-name=train
#SBATCH -N1 # Run this across 1 nodes
#SBATCH -n1 # Run 1 tasks
#SBATCH --mail-user=calvin.yeung@alleninstitute.org # Send mail to your AI email
#SBATCH --mail-type=BEGIN,END,FAIL # Send mail on begin, end, fail
#SBATCH -p braintv
#SBATCH --mem=32gb                     # Job memory request (per node)
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH -o ./output/out_train_%j.out # output goes to gethostname_<JOBID>.out
#SBATCH -e ./output/err_train_%j.err # error goes to gethostname_<JOBID>.err
source /home/calvin.yeung/.bash_profile
source /home/calvin.yeung/.bashrc
source /allen/ai/hpc/shared/utils.x86_64/anaconda3-2021.11/bin/activate /home/calvin.yeung/.conda/envs/pytorch-gpu
python3 -u model_pipeline.py $1 $2 $3 $4 $5 $6