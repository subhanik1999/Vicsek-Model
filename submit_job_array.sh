#!/bin/bash
#SBATCH -J Python_Job_Array_Test
#SBATCH -t 0:02:00
#SBATCH --array=1-3

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e arrayjob-%a.err
#SBATCH -o arrayjob-%a.out

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"
source ./bin/activate
python vicsek_model.py --job $SLURM_ARRAY_TASK_ID
deactivate
