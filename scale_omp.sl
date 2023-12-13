#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:00:59
#SBATCH --output=scale_mpi.out
#SBATCH --account=anakano_429

counter=1
while [ $counter -lt 5 ]; do
  echo "***** threads $counter*****"
  ./main_omp daip -N $((counter*100000)) -threads $counter
  let counter*=2
done
