#!/bin/bash

# sbatch setup
#SBATCH --job-name=ECC_test
#SBATCH --output=/user/ahagel14/output/%j_%N.out

#SBATCH --partition=eagle
#SBATCH --mem=200G
#SBATCH --cpus-per-task=48
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahagel14@student.aau.dk

# Job setup
export EXECUTION_PATH="/user/ahagel14/NoBackup/p10/code/" # Path to input data
export OUTPUT_DATA_PATH="$HOME/output/" # Path to place output
export PYTHON_VENV="/pack/anaconda-5.1.0-py36-cuda/bin" # Anaconda version to use
export PYTHONPATH="$PYTHONPATH:$EXECUTION_PATH"
export PATH="$PYTHON_VENV:$PATH"

# Job identification
echo "THIS IS THE MAIN JOB"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Job dependency: $SLURM_JOB_DEPENDENCY"
echo "Job partition: $SLURM_JOB_PARTITION"
echo "Allocated nodes: $SLURM_JOB_NODELIST"
echo "Allcoated CPUs on node: $SLURM_JOB_CPUS_PER_NODE"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Python virtual environment: $PYTHON_VENV"

# Host identification
echo -n "Host: "
hostname
echo -n "Begin: "
date
echo "================================================================"

# If the data is not present on the scratch parition of the server, move it there

srun --chdir=$EXECUTION_PATH "echo $(pwd)"
srun --chdir=$EXECUTION_PATH "$HOME/scripts/run_python.sh" "$EXECUTION_PATH/tests/ecc_test.py"

echo "================================================================"
echo -n "End: "
date
