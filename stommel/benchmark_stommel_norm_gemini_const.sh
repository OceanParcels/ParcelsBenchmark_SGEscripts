#!/bin/sh
# SGE Options
#$ -S /bin/bash
# Shell environment forwarding
#$ -V
# Job Name
#$ -N benchmark_stommel_norm

# Notifications
#$ -M <fill-in-mail-address>

# When notified
#$ -m es
# Set memory limit
#$ -l h_vmem=60G
# Set runtime limit
#$ -l h_rt=96:00:00

# run the job on the queue for long-running processes (parameterization depends on your cluster!):
#$ -q <queue-name>

echo 'Initialize environment'

export SCRIPT_LOCATION = #<location-of-these-runscripts>#
cd ${SCRIPT_LOCATION}
export PARCELS_HEAD= #<location your your parcels checkout from github; main folder>#
export TARGET_HEAD= #<location where your benchmark results shall be stored>#

echo '======== JIT (Just-in-Time) experiments ========'
echo ' ---- constant particle number ---- '
python3.6 -m cProfile -o ${TARGET_HEAD}/benchmark_stommel_GEMINI_noMPI_constP_2pow10_wGC_jit.prof ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**10 -i stommel_GEMINI_noMPI_constP-2pow10_wGC_jit.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**10 -i stommel_GEMINI_noMPI_constP-2pow10_wGC_jit.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**11 -i stommel_GEMINI_noMPI_constP-2pow11_wGC_jit.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**12 -i stommel_GEMINI_noMPI_constP-2pow12_wGC_jit.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**13 -i stommel_GEMINI_noMPI_constP-2pow13_wGC_jit.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**14 -i stommel_GEMINI_noMPI_constP-2pow14_wGC_jit.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**15 -i stommel_GEMINI_noMPI_constP-2pow15_wGC_jit.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**16 -i stommel_GEMINI_noMPI_constP-2pow16_wGC_jit.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**17 -i stommel_GEMINI_noMPI_constP-2pow17_wGC_jit.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**18 -i stommel_GEMINI_noMPI_constP-2pow18_wGC_jit.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_stommel.py -t 365 -G -N 2**19 -i stommel_GEMINI_noMPI_constP-2pow19_wGC_jit.png

echo 'Finished program execution.'
