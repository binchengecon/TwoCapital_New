#! /bin/bash

action_name="UploadDB_Parallel"
python_name="UploadDB4.py"
MotherRoot="/scratch/bincheng/abatement/data_2tech/"
DaughterRoot="/climatesocialpolicy/data_full/"

access_token="sl.BiEqok4dyURKMprbi01Ogie0G2xakrvb9V6eY_8EWJBIlwvZf-b9OC8i7yxF_2sJ8oB7gNCDU5WAMmEyasOgmtTtRGwehZMzZkiLp9MBsXVJnxHsAMkSbUfnoBY0hKQYNuLeAjWZz3De"
refresh_token="FwlECrv3_eQAAAAAAAAAAR2FyGLKIQa0FYMB26IyxZnsziQTtF6bH-i7hSMT_1TT"

declare -a FolderArray=("2jump_step_4.00,9.00_0.0,4.0_1.0,6.0_0.0,3.0_SS_0.2,0.1,0.1_LR_0.0025_FK_flow_phi0_0.5"
    "2jump_step_4.00,9.00_0.0,4.0_1.0,6.0_0.0,3.0_SS_0.2,0.1,0.1_LR_0.0025_FK_flow_phi0_0.1"
    "2jump_step_4.00,9.00_0.0,4.0_1.0,6.0_0.0,3.0_SS_0.2,0.1,0.1_LR_0.0025_FK_phi0_0.5"
    "2jump_step_4.00,9.00_0.0,4.0_1.0,6.0_0.0,3.0_SS_0.2,0.1,0.1_LR_0.0025_FK_phi0_0.1")

# Read the array values with space

count=0
for Folder in "${FolderArray[@]}"; do
    mkdir -p ./job-outs/${action_name}/

    if [ -f ./bash/${action_name}/${Folder}.sh ]; then
        rm ./bash/${action_name}/${Folder}.sh
    fi

    mkdir -p ./bash/${action_name}/

    touch ./bash/${action_name}/${Folder}.sh

    tee -a ./bash/${action_name}/${Folder}.sh <<EOF
#! /bin/bash


######## login 
#SBATCH --job-name=UDB${count}
#SBATCH --output=./job-outs/${action_name}/${Folder}.out
#SBATCH --error=./job-outs/${action_name}/${Folder}.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task


python3 /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --Folder ${Folder} --MotherRoot ${MotherRoot} --DaughterRoot ${DaughterRoot}  --access_token ${access_token} --refresh_token ${refresh_token}

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
    count=$(($count + 1))
    sbatch ./bash/${action_name}/${Folder}.sh
done
