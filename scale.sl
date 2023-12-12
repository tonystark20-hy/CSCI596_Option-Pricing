#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:00:59
#SBATCH --output=scale.out
#SBATCH --account=anakano_429

counter=0
while [ $counter -lt 3 ]; do
  echo "***** CUDA *****"
  mpirun -n $SLURM_NTASKS ./main
  echo "***** CUDA+OMP *****"
  mpirun -n $SLURM_NTASKS ./main_omp
  echo "***** CUDA+OMP+MPI *****"
  mpirun -n $SLURM_NTASKS ./main_mpi
  let counter+=1
done
