#!/bin/bash
#SBATCH -N1 # Run this across 1 nodes
#SBATCH -n1 # Run 1 tasks
#SBATCH --mem=5gb  #Job memory request
#SBATCH --mail-user=my.email@alleninstitute.org # Send mail to your AI email
#SBATCH --mail-type=BEGIN,END,FAIL # Send mail on begin, end, fail
#SBATCH -p braintv
#SBATCH -t 5 # Schedule n minute wallclock
#SBATCH -o ./hello_world_%j.out # output goes to gethostname_<JOBID>.out
#SBATCH -e ./hello_world_%j.err # error goes to gethostname_<JOBID>.err
module load anaconda/4.3.1
conda activate pytorch-gpu
python3 model_pipeline.py