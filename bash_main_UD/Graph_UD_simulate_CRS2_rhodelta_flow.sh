#! /bin/bash

actiontime=1
# epsilonarraypost=(0.1) # Computation of fine grid and psi10.8, post
# epsilonarraypost=(0.05) # Computation of fine grid and psi10.8, post
# epsilonarraypost=(0.01) # Computation of fine grid and psi10.8, post
# epsilonarraypost=(0.005) # Computation of fine grid and psi10.8, post
# epsilonarraypost=(0.025) # Computation of fine grid and psi10.8, post
# epsilonarraypost=(0.004) # Computation of fine grid and psi10.8, post
epsilonarraypost=(0.0025) # Computation of fine grid and psi10.8, post
# epsilonarraypost=(0.001) # Computation of fine grid and psi10.8, post

NUM_DAMAGE=20

declare -A hXarr1=([0]=0.2 [1]=0.2 [2]=0.2)
declare -A hXarr2=([0]=0.1 [1]=0.1 [2]=0.1)
declare -A hXarr3=([0]=0.05 [1]=0.05 [2]=0.05)
declare -A hXarr4=([0]=0.2 [1]=0.1 [2]=0.2)
declare -A hXarr5=([0]=0.2 [1]=0.1 [2]=0.1)

# hXarrays=(hXarr1 hXarr2 hXarr3)
# hXarrays=(hXarr1)
# hXarrays=(hXarr2)
# hXarrays=(hXarr3)
# hXarrays=(hXarr4)
hXarrays=(hXarr5)


Xminarr=(4.00 0.0 1.0 0.0)
Xmaxarr=(9.00 4.0 6.0 3.0)



# xi_a=(100000. 100000. 100000. 100000.)
# xi_k=(0.050 100000. 100000. 100000.)
# xi_c=(100000. 0.050 100000. 100000.)
# xi_j=(100000. 100000. 0.050 100000.)
# xi_d=(100000. 100000. 100000. 0.050)
# xi_g=(100000. 100000. 0.050 100000.)


# xi_a=(100000. 100000. 100000.)
# xi_k=(0.025  0.050 100000.)
# xi_c=(0.025  0.050 100000.)
# xi_j=(0.025  0.050 100000.)
# xi_d=(0.025  0.050 100000.)
# xi_g=(0.025  0.050 100000.)




# xi_a=(100000. 100000. 100000.)
# xi_k=(0.075  0.150 100000.)
# xi_c=(0.075  0.150 100000.)
# xi_j=(0.075  0.150 100000.)
# xi_d=(0.075  0.150 100000.)
# xi_g=(0.075  0.150 100000.)


xi_a=(100000. 100000. 100000. 100000.)
xi_k=(0.150 100000. 100000. 100000.)
xi_c=(100000. 0.150 100000. 100000.)
xi_j=(100000. 100000. 0.150 100000.)
xi_d=(100000. 100000. 100000. 0.150)
xi_g=(100000. 100000. 0.150 100000.)





varrhoarr=(1120)
# varrhoarr=(448)


# psi01arr=(0.052915)
# psi02arr=(0.052915)

psi01arr=(0.105830)
psi02arr=(0)


# psi0arr=(0.000001)
psi1arr=(0.5)


# rhoarr=(0.66 1 1.5)
# deltaarr=(0.010 0.010 0.010)


rhoarr=(1)
deltaarr=(0.010)

# rhoarr=(0.66 0.77 0.88 0.99 0.66 0.77 0.88 0.93)
# deltaarr=(0.015 0.015 0.015 0.015 0.010 0.010 0.010 0.010)


# rhoarr=(0.99 0.9999 1 1.0001)
# deltaarr=(0.010 0.010 0.010 0.010)

# rhoarr=(0.66 0.77 0.88 0.99)
# deltaarr=(0.015 0.015 0.015 0.015)

# rhoarr=(0.66)
# deltaarr=(0.03)

# rhoarr=(0.88)
# deltaarr=(0.02)

# rhoarr=(0.66 0.77 0.88 0.99 1.00001 1.5)

LENGTH_rho=$((${#rhoarr[@]} - 1))

# phi0arr=(0.05 0.1 0.2 0.3 0.4 0.5)

# phi0arr=(0.05 0.1 0.2 0.5)
# phi0arr=(0.05 0.1 0.3 0.5)
# phi0arr=(0.05 0.1 0.2 0.5)
# phi0arr=(0.1 0.5)
phi0arr=(0.5)
# phi0arr=(0.1)
LENGTH_phi0=$((${#phi0arr[@]} - 1))


# python_name_unit="Result_2jump_UD_simulate_CRS.py"
# python_name_unit="Result_2jump_UD_simulate_CRS_long_rhodelta.py"
python_name_unit="Result_2jump_UD_simulate_CRS_long_rhodelta2_flow.py"
# python_name_unit="Result_2jump_UD_simulate_CRS_MulJump.py"


server_name="mercury"

LENGTH_psi=$((${#psi01arr[@]} - 1))
LENGTH_xi=$((${#xi_a[@]} - 1))

hXarr_SG=(0.2 0.2 0.2)
Xminarr_SG=(4.00 0.0 -5.5 0.0)
Xmaxarr_SG=(9.00 4.0 0.0 3.0)
interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.01"
fstr_SG="NearestNDInterpolator"

auto=1
# year=25
year=40

# scheme_array=("macroannual" "newway" "newway" "newway" "check")
# HJBsolution_array=("simple" "iterative_partial" "iterative_fix" "n_iterative_fix" "iterative_partial")

# scheme_array=("macroannual" "newway" "newway" "newway")
# HJBsolution_array=("simple" "iterative_partial" "iterative_fix" "n_iterative_fix")
# scheme_array=("newway")
# HJBsolution_array=("n_iterative_fix")

# scheme_array=("newway" "newway" "newway" "check")
# HJBsolution_array=("iterative_partial" "iterative_fix" "n_iterative_fix" "iterative_partial")
# scheme_array=("newway" "check")
# HJBsolution_array=("iterative_fix" "iterative_partial")

scheme_array=("direct")
HJBsolution_array=("direct")


LENGTH_scheme=$((${#scheme_array[@]} - 1))




for epsilonpost in ${epsilonarraypost[@]}; do
    for hXarri in "${hXarrays[@]}"; do
        for phi0index in $(seq 0 $LENGTH_phi0); do

        count=0
        declare -n hXarr="$hXarri"

        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_CRS"
        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_CRS_MulJump"
        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_oldmodel"
        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_postadd_phi0_${phi0arr[$phi0index]}"
        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_deltarho_phi0_${phi0arr[$phi0index]}"
        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_new_deltarho_phi0_${phi0arr[$phi0index]}"
        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_testbar_phi0_${phi0arr[$phi0index]}"
        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_testalgo_phi0_${phi0arr[$phi0index]}"
        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_FK_flow_phi0_${phi0arr[$phi0index]}"
        action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_FK_flow_sub_phi0_${phi0arr[$phi0index]}"
        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_${Xminarr[3]},${Xmaxarr[3]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_deltarho_testlargedelta_phi0_${phi0arr[$phi0index]}"

        for PSI_01 in ${psi01arr[@]}; do
        for PSI_02 in ${psi02arr[@]}; do
            for PSI_1 in ${psi1arr[@]}; do
                for varrho in ${varrhoarr[@]}; do
                    for j in $(seq 0 $LENGTH_xi); do
                        for k in $(seq 0 $LENGTH_scheme); do
							for kk in $(seq 0 $LENGTH_rho); do

                    mkdir -p ./job-outs/${action_name}/Graph_Simulate/scheme_${scheme_array[$k]}_HJB_${HJBsolution_array[$k]}/xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI01_${PSI_01}_PSI02_${PSI_02}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rhoarr[$kk]}_delta_${deltaarr[$kk]}/

                    if [ -f ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI01_${PSI_01}_PSI02_${PSI_02}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rhoarr[$kk]}_delta_${deltaarr[$kk]}_Graph.sh ]; then
                        rm ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI01_${PSI_01}_PSI02_${PSI_02}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rhoarr[$kk]}_delta_${deltaarr[$kk]}_Graph.sh
                    fi
                    mkdir -p ./bash/${action_name}/

                    touch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI01_${PSI_01}_PSI02_${PSI_02}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rhoarr[$kk]}_delta_${deltaarr[$kk]}_Graph.sh

                    tee -a ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI01_${PSI_01}_PSI02_${PSI_02}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rhoarr[$kk]}_delta_${deltaarr[$kk]}_Graph.sh <<EOF
#! /bin/bash


######## login 
#SBATCH --job-name=sim_${year}
#SBATCH --output=./job-outs/${action_name}/Graph_Simulate/scheme_${scheme_array[$k]}_HJB_${HJBsolution_array[$k]}/xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI01_${PSI_01}_PSI02_${PSI_02}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rhoarr[$kk]}_delta_${deltaarr[$kk]}/graph_simulate_${year}.out
#SBATCH --error=./job-outs/${action_name}/Graph_Simulate/scheme_${scheme_array[$k]}_HJB_${HJBsolution_array[$k]}/xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI01_${PSI_01}_PSI02_${PSI_02}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rhoarr[$kk]}_delta_${deltaarr[$kk]}/graph_simulate_${year}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8  gcc/9.2.0


echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python3 /home/bcheng4/TwoCapital_Shrink/abatement_UD/${python_name_unit} --dataname  ${action_name} --pdfname ${server_name} --psi01 ${PSI_01} --psi02 ${PSI_02} --psi1 ${PSI_1}  --xiaarr ${xi_a[$j]} --xikarr ${xi_k[$j]} --xicarr ${xi_c[$j]}  --xijarr ${xi_j[$j]} --xidarr ${xi_d[$j]} --xigarr ${xi_g[$j]}   --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]} --auto $auto --IntPeriod ${year} --scheme ${scheme_array[$k]}  --HJB_solution ${HJBsolution_array[$k]}  --varrho ${varrho}   --phi_0 ${phi0arr[$phi0index]}  --rhoarr ${rhoarr[$kk]} --delta ${deltaarr[$kk]}

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF

                    sbatch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xik_${xi_k[$j]}_xic_${xi_c[$j]}_xij_${xi_j[$j]}_xid_${xi_d[$j]}_xig_${xi_g[$j]}_PSI01_${PSI_01}_PSI02_${PSI_02}_PSI1_${PSI_1}_varrho_${varrho}_rho_${rhoarr[$kk]}_delta_${deltaarr[$kk]}_Graph.sh

                        done
                    done
                done
            done
        done
    done
done
done
done
done