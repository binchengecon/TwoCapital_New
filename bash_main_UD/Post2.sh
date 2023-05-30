#! /bin/bash

# epsilonarray=(0.1) 
# epsilonarray=(0.2) 
# epsilonarray=(0.05) 
# epsilonarray=(0.005) 
epsilonarray=(0.005) 
# epsilonarray=(0.01) 



python_name="postdamage_2jump_CRS2.py" # 3 dmg
# python_name="postdamage_2jump_CRS2_expo.py" # 3 dmg
# python_name="postdamage_2jump_CRS2_Mike.py" # 3 dmg
# python_name="postdamage_2jump_Short.py" # 3 dmg
# python_name="postdamage_2jump_CRS2_onlydamage.py" # 3 dmg
# python_name="postdamage_2jump_CRS2_onlytech.py" # 3 dmg
# python_name="postdamage_2jump_CRS2_bothdamagetech.py" # 3 dmg


# NUM_DAMAGE=20
# NUM_DAMAGE=10
# NUM_DAMAGE=3
NUM_DAMAGE=4

ID_MAX_DAMAGE=$((NUM_DAMAGE - 1))

# maxiterarr=(80000 200000)
maxiterarr=(2000000 4000000)
# maxiterarr=(10 10)

declare -A hXarr1=([0]=0.2 [1]=0.2 [2]=0.2)
declare -A hXarr2=([0]=0.1 [1]=0.1 [2]=0.1)
declare -A hXarr3=([0]=0.05 [1]=0.05 [2]=0.05)
declare -A hXarr4=([0]=0.2 [1]=0.01 [2]=0.2)
declare -A hXarr5=([0]=0.1 [1]=0.05 [2]=0.1)
declare -A hXarr6=([0]=0.1 [1]=0.025 [2]=0.1)
declare -A hXarr7=([0]=0.1 [1]=0.01 [2]=0.1)
declare -A hXarr8=([0]=0.2 [1]=0.1 [2]=0.2)
# hXarrays=(hXarr1 hXarr2 hXarr3)
hXarrays=(hXarr1)
# hXarrays=(hXarr2)
# hXarrays=(hXarr3)
# hXarrays=(hXarr4)
# hXarrays=(hXarr5)
# hXarrays=(hXarr6)
# hXarrays=(hXarr7)
# hXarrays=(hXarr8)


# Xminarr=(4.00 0.0 1.0 0.0)
# Xmaxarr=(9.00 4.0 6.0 3.0)

Xminarr=(4.00 1.2 1.0 0.0)
Xmaxarr=(9.00 4.0 6.0 3.0)

# xi_a=(100000. 100000. 100000.)
# xi_c=(0.050 100000. 100000.)
# xi_d=(100000. 0.050 100000.)
# xi_g=(100000. 100000. 0.050)


# xi_a=(100000. 100000. 100000.)
# xi_c=(0.025 0.050 100000.)
# xi_d=(0.025 0.050 100000.)
# xi_g=(0.025 0.050 100000.)


# xi_a=(100000. 100000. 100000.)
# xi_c=(0.250 0.500 1.000)
# xi_d=(0.025 0.025 0.025)
# xi_g=(0.025 0.025 0.025)


# xi_a=(100000. 100000. 100000.)
# xi_c=(0.025 0.050 0.100)
# xi_d=(0.025 0.025 0.025)
# xi_g=(0.025 0.025 0.025)

# xi_a=(100000. 100000. 100000.)
# xi_c=(0.025 0.050 0.100)
# xi_d=(0.025 0.050 0.100)
# xi_g=(0.025 0.050 0.100)


xi_a=(100000. 100000. 100000.)
xi_k=(0.050 0.025 100000.)
xi_c=(0.050 0.025 100000.)
xi_j=(0.050 0.025 100000.)
xi_d=(0.050 0.025 100000.)
xi_g=(0.050 0.025 100000.)



# xi_a=(100000. 100000. 100000. 100000. 100000.)
# xi_k=(0.050 100000. 100000. 100000. 0.050)
# xi_c=(100000. 0.050 100000. 100000. 0.050)
# xi_j=(100000. 100000. 0.050 100000. 0.050)
# xi_d=(100000. 100000. 100000. 0.050 0.050)
# xi_g=(100000. 100000. 0.050 100000. 0.050)

# xi_a=(100000. 100000.)
# xi_k=(100000. 100000.)
# xi_c=(100000. 100000.)
# xi_j=(100000. 0.050)
# xi_d=(100000. 100000.)
# xi_g=(0.050 100000.)

# xi_a=(100000.)
# xi_k=(100000.)
# xi_c=(100000.)
# xi_j=(0.05)
# xi_d=(100000.)
# xi_g=(0.05)

# xi_a=(100000. 100000.)
# xi_k=(100000. 100000.)
# xi_c=(100000. 100000.)
# xi_j=(0.050 0.1)
# xi_d=(100000. 100000.)
# xi_g=(0.1 0.050)



# xi_a=(100000.)
# xi_k=(100000.)
# xi_c=(100000.)
# xi_j=(0.050)
# xi_d=(100000.)
# xi_g=(0.050)

# xi_a=(100000.)
# xi_k=(0.050)
# xi_c=(0.050)
# xi_j=(0.050)
# xi_d=(0.050)
# xi_g=(0.050)

varrhoarr=(1120)
# varrhoarr=(448)

rhoarr=(0.66 1.00001 1.5)

# rhoarr=(0.66 0.77 0.88 0.99 1.00001 1.5)

# rhoarr=(0.66)
# rhoarr=(1.00001 1.5)

LENGTH_rho=$((${#rhoarr[@]} - 1))

psi0arr=(0.105830)

# phi0arr=(0.056 0.1 0.2 0.3 0.4 0.5)
phi0arr=(0.5)
LENGTH_phi0=$((${#phi0arr[@]} - 1))



psi1arr=(0.5)

LENGTH_psi=$((${#psi0arr[@]} - 1))
LENGTH_xi=$((${#xi_a[@]} - 1))


hXarr_SG=(0.2 0.2 0.2)
Xminarr_SG=(4.00 0.0 -5.5 0.0)
Xmaxarr_SG=(9.00 4.0 0.0 3.0)
interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.01"
fstr_SG="NearestNDInterpolator"

for epsilon in ${epsilonarray[@]}; do
	for hXarri in "${hXarrays[@]}"; do
        for phi0index in $(seq 0 $LENGTH_phi0); do
		count=0
		declare -n hXarr="$hXarri"

		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_geneal_phi0_${phi0arr[$phi0index]}"
		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_general_longer_phi0_${phi0arr[$phi0index]}"
		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_general_longer3_phi0_${phi0arr[$phi0index]}"
		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_general_longer4dmg_phi0_${phi0arr[$phi0index]}"
		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_4dmg_phi0_${phi0arr[$phi0index]}"
		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_general_longer_Mike_phi0_${phi0arr[$phi0index]}"
		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_expo_phi0_${phi0arr[$phi0index]}"
		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_new_phi0_${phi0arr[$phi0index]}"
		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_newdmg_phi0_${phi0arr[$phi0index]}"
		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_newtech_phi0_${phi0arr[$phi0index]}"
		action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_plugin_phi0_${phi0arr[$phi0index]}"
		# action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}_plugin_expo_phi0_${phi0arr[$phi0index]}"


		
		epsilonarr=(0.005 ${epsilon})
		fractionarr=(0.005 ${epsilon})

		# epsilonarr=(0.01 0.0025)
		# fractionarr=(0.01 0.0025)

		# i=1
		for i in $(seq 0 $ID_MAX_DAMAGE); do
			for PSI_0 in ${psi0arr[@]}; do
				for PSI_1 in ${psi1arr[@]}; do
					for varrho in ${varrhoarr[@]}; do
						for j in $(seq 0 $LENGTH_xi); do
							for rho in ${rhoarr[@]}; do

								mkdir -p ./job-outs/${action_name}/Post/xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rho}/

								if [ -f ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rho}_ID_${i}.sh ]; then
									rm ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rho}_ID_${i}.sh
								fi

								mkdir -p ./bash/${action_name}/

								touch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rho}_ID_${i}.sh

								tee -a ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rho}_ID_${i}.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_${rho}_${i}_${epsilon}
#SBATCH --output=./job-outs/${action_name}/Post/xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rho}/mercury_post_$i.out
#SBATCH --error=./job-outs/${action_name}/Post/xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rho}/mercury_post_$i.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=mcn53,mcn52,mcn51

####### load modules
module load python/booth/3.8  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u /home/bcheng4/TwoCapital_Shrink/abatement_UD/$python_name --num_gamma $NUM_DAMAGE --xi_a ${xi_a[$j]} --xi_k ${xi_k[$j]} --xi_c ${xi_c[$j]} --xi_j ${xi_j[$j]}  --xi_d ${xi_d[$j]} --xi_g ${xi_g[$j]}  --epsilonarr ${epsilonarr[@]}  --fractionarr ${fractionarr[@]}   --maxiterarr ${maxiterarr[@]}  --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name} --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]}  --varrho ${varrho}  --phi_0 ${phi0arr[$phi0index]}  --rho ${rho}

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"
# echo ${hXarr[@]}

EOF
									count=$(($count + 1))
									sbatch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rho}_ID_${i}.sh
								done
							done
						done
					done
				done
			done
		done
	done
done