#!/bin/bash
#$ -S /bin/bash
# Shell environment forwarding
#$ -V
## Job Name
#$ -N benchmark_parcels_jobarray
# task array option
##$ -t 1-44 -tc 4
# join stdout and stderr
#$ -j y

# Notifications
#$ -M <fill-in-mail-address>

# When notified
#$ -m es
# Set memory limit
#$ -l h_vmem=60G
# Set runtime limit - max. 168:00:00
#$ -l h_rt=120:00:00

# run the job on the queue for long-running processes (parameterization depends on your cluster!):
#$ -q <queue-name>

# ===== EXECUTE SCRIPT USING THE SCHEDULED JOB ARRAY ===== #
# == call: qsub -V -t 1-44 -tc 4 exec_parcels_benchmark == #
# ======================================================== #

export SCRIPT_LOCATION = #<location-of-these-runscripts>#
cd ${SCRIPT_LOCATION}
export PARCELS_HEAD= #<location your your parcels checkout from github; main folder>#
export TARGET_HEAD= #<location where your benchmark results shall be stored>#

joblist="$1"
if [ -z "$joblist" ] 
then
	joblist=${SCRIPT_LOCATION}/exec_benchmark_perlin_norm.txt
	echo "No joblist given - joblist source set to ${joblist}"
 
else
	echo "Source job list is: ${joblist}"
fi


QSUB_PARCELS_START_SCRIPT=`awk "NR==$SGE_TASK_ID" ${joblist}`
echo `date`
echo "Starting benchmark ..."
echo $QSUB_PARCELS_START_SCRIPT
eval $QSUB_PARCELS_START_SCRIPT
echo "Fished benchmark."
