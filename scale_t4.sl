#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:00:59
#SBATCH --output=scale_mpi.out
#SBATCH --account=anakano_429

counter=1
while [ $counter -lt 5 ]; do
  echo "***** counter $counter *****"
  mpirun -bind-to none -n $counter ./main_mpi daip -N 100000 -threads 4
  let counter*=2
done
