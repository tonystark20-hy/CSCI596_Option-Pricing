#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:04:59
#SBATCH --output=scale_t1.out
#SBATCH --account=anakano_429

counter=1
while [ $counter -lt 5 ]; do
  echo "***** counter $counter *****"
  mpirun -bind-to none -n $counter ./main_mpi daip -N 5000000 -threads 1
  let counter*=2
done

counter=1
while [ $counter -lt 5 ]; do
  echo "***** counter $counter *****"
  mpirun -bind-to none -n $counter ./main_mpi daip -N 5000000 -threads 2
  let counter*=2
done

counter=1
while [ $counter -lt 5 ]; do
  echo "***** counter $counter *****"
  mpirun -bind-to none -n $counter ./main_mpi daip -N 5000000 -threads 4
  let counter*=2
done
