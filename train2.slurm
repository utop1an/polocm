#! /bin/bash
#SBATCH --job-name=polocm
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=u6797221@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --output=%j_polocm_2.log
# Standard output and error log
pwd; hostname; date
echo "--- running ---"

# disable cplex log
export CPLEX_LOGFILE=/dev/null

for dod in 0.6 0.7 0.8 0.9 1; do
    echo "Running with --d=$dod"
    singularity exec -B ~/xLOCM/src/:/src \
        -B ~/xLOCM/data/:/data \
        -B ~/xLOCM/logs:/logs \
        -B ~/xLOCM/output/:/output \
        ~/xLOCM/polocm.sif python3 /src/experiment.py --i ./data/traces_combined_r1_dod0.json --o ./output \
        --c 8 \ 
        --d $dod \
        --cplex /opt/cplex/cplex/bin/x86-64_linux/cplex

    echo "Completed task with --d=$dod"
done

echo "--- done ---"
date
