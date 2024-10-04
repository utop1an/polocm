#! /bin/bash
#SBATCH --job-name=polocm_debug
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=u6797221@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=120:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --output=polocm_%j.log
# Standard output and error log
pwd; hostname; date
echo "--- running ---"

singularity exec /src \
    -B ~/polocm/data/:/data \
    -B ~/polocm/logs:/logs \
    -B ~/polocm/output/:/output \
    ~/polocm/polocm.sif python3 /src/experiment.py --i ./data/traces_combined_r1.json --o ./output \
    --c $SLURM_CPUS_PER_TASK \
    --debug --cplex /opt/cplex/cplex/bin/x86-64_linux/cplex

echo "--- done ---"
date
