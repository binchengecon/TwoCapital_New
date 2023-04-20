#! /bin/bash


######## login 
#SBATCH --job-name=pre-0
#SBATCH --output=./job-outs/comparisonwitholdpsi01/0.005_0.5/mercury_pre_0.out
#SBATCH --error=./job-outs/comparisonwitholdpsi01/0.005_0.5/mercury_pre_0.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Bin/abatement/predamage_spe_psi_name_moreiteration.py --xi_a 1000.0 --xi_g 1000.0 --psi_0 0.005 --psi_1 0.5 --name comparisonwitholdpsi01

echo "Program ends $(date)"

