#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:00:59
#SBATCH --output=scale_cuda.out
#SBATCH --account=anakano_429

counter=1
while [ $counter -lt 5 ]; do
  echo "***** threads $counter*****"
  ./main daip -N $((counter*100000)) 
  let counter*=2
done
