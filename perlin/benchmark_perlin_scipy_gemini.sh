#!/bin/sh
# SGE Options
#$ -S /bin/bash
# Shell environment forwarding
#$ -V
# Job Name
#$ -N benchmark_perlin_wGC_scipy

# Notifications
#$ -M <fill-in-mail-address>

# When notified
#$ -m es
# Set memory limit
#$ -l h_vmem=50G
# Set runtime limit
#$ -l h_rt=336:00:00

# run the job on the queue for long-running processes (parameterization depends on your cluster!):
#$ -q <queue-name>

echo 'Initialize environment'

export SCRIPT_LOCATION = #<location-of-these-runscripts>#
cd ${SCRIPT_LOCATION}
export PARCELS_HEAD= #<location your your parcels checkout from github; main folder>#
export TARGET_HEAD= #<location where your benchmark results shall be stored>#

echo '======== SciPy experiments (ONLY ON GEMINI) - N=2^10 ========'
echo ' ---- constant particle number ---- '
python3.6 -m cProfile -o ${TARGET_HEAD}/benchmark_perlin_GEMINI_noMPI_constP_2pow10_wGC_scipy.prof ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -t 365 -G -N 2**10 -i perlin_GEMINI_noMPI_constP-2pow10_wGC_scipy.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -t 365 -G -N 2**10 -i perlin_GEMINI_noMPI_constP-2pow10_wGC_scipy.png
echo ' ---- dynamically removing particles (aging with t_max=14 days) ---- '
python3.6 -m cProfile -o ${TARGET_HEAD}/benchmark_perlin_GEMINI_noMPI_ageP_2pow10_wGC_scipy.prof ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -a -t 365 -G -N 2**10 -i perlin_GEMINI_noMPI_ageP-2pow10_wGC_scipy.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -a -t 365 -G -N 2**10 -i perlin_GEMINI_noMPI_ageP-2pow10_wGC_scipy.png
echo ' ---- dynamically adding particles (adaptive release rate) ---- '
python3.6 -m cProfile -o ${TARGET_HEAD}/benchmark_perlin_GEMINI_noMPI_addP_2pow10_wGC_scipy.prof ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -t 365 -G -r -sN 128 -N 2**10 -i perlin_GEMINI_noMPI_addP-2pow10_wGC_scipy.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -t 365 -G -r -sN 128 -N 2**10 -i perlin_GEMINI_noMPI_addP-2pow10_wGC_scipy.png
echo ' ---- dynamically adding and removing particles ---- '
python3.6 -m cProfile -o ${TARGET_HEAD}/benchmark_perlin_GEMINI_noMPI_ageAddP_2pow10_wGC_scipy.prof ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -a -t 365 -G -r -sN 128 -N 2**10 -i perlin_GEMINI_noMPI_ageAddP-2pow10_wGC_scipy.png
python3.6 ${PARCELS_HEAD}/performance/benchmark_perlin.py -m scipy -a -t 365 -G -r -sN 128 -N 2**10 -i perlin_GEMINI_noMPI_ageAddP-2pow10_wGC_scipy.png

echo 'Finished program execution.'
