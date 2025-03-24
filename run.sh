#!/bin/bash
#SBATCH -n 1
#SBATCH -o slurm-output/winograd-job-%j.out
#SBATCH -e slurm-error/winograd-job-%j.err
#SBATCH -c 64
#SBATCH --exclusive
#SBATCH --exclude hepnode0

# Note: How to run this script on slurm: `sbatch ./run.sh'.
# Note: see `man sbatch' for more options.

# Note: Manual control number of omp threads
export OMP_NUN_THREADS=64

# Note: numactl - Control NUMA policy for processes or shared memory, see `man numactl'.`
# Note: perf-stat - Run a command and gather performance counter statistics, see `man perf stat'.

echo "Parameters: [version:vx.x.x] [input_config(0/1)]"

directory=result_data
version=$1

if [ "$2" == "1" ]; then
	#Use vgg16.conf
	CONFIG_FILE="conf/vgg16.conf"
	OUTPUT_FILE="$directory/$version-big.out"
elif [ "$2" == "0" ]; then
	#Use small.conf
	CONFIG_FILE="conf/small.conf"
	OUTPUT_FILE="$directory/$version-small.out"
else
	echo "Wrong Parameters!"

fi


numactl --cpunodebind=0 --membind=0 perf stat -ddd ./winograd $CONFIG_FILE > $OUTPUT_FILE 
