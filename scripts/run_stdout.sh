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

echo "This will run program on hepnode0, are you sure? (1/0)"
read answer
if [ "$answer" == 1 ] ;then
	echo "Yes"
else
	echo "No"
	exit
fi

directory=result_data

if [ "$1" == "1" ]; then
	#Use vgg16.conf
	CONFIG_FILE="conf/vgg16.conf"
elif [ "$1" == "0" ]; then
	#Use small.conf
	CONFIG_FILE="conf/small.conf"
else
	echo "Wrong Parameters!"

fi


numactl --cpunodebind=0 --membind=0 perf stat -ddd ./winograd $CONFIG_FILE 
