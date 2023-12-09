// commit test
#include "mpi.h"

#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "dev_array.h"
#include <curand.h>

using namespace std;

const size_t N_PATHS = 100000;

int main(int argc,char *argv[])
{
    try
    {
        // declare variables and constants
        const size_t N_STEPS = 365;
        const size_t N_NORMALS = N_PATHS * N_STEPS;
        const float T = 1.0f;
        const float K = 100.0f;
        const float B = 95.0f;
        const float S0 = 100.0f;
        const float sigma = 0.2f;
        const float mu = 0.1f;
        const float r = 0.05f;
        float dt = float(T) / float(N_STEPS);
        float sqrdt = sqrt(dt);

        int nprocs;  /* Number of processes */
        int myid;    /* My rank */

        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

        // Calculate workload for each MPI process
        // size_t paths_per_process = N_PATHS / nprocs;
        size_t paths_per_process = N_PATHS / nprocs;
        size_t start_idx = myid * paths_per_process;
        size_t end_idx = start_idx + paths_per_process;

        // generate arrays
        vector<float> s(N_PATHS);
        dev_array<float> d_s(N_PATHS);
        dev_array<float> d_normals(N_NORMALS);

        // generate random numbers
        curandGenerator_t curandGenerator;
        curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
        curandSetPseudoRandomGeneratorSeed(curandGenerator, 0);
        curandGenerateNormal(curandGenerator, d_normals.getData(), N_NORMALS, 0.0f, sqrdt);
        double t2 = double(clock()) / CLOCKS_PER_SEC;

        // call the kernel
        mc_dao_call(d_s.getData(), T, K, B, S0, sigma, mu, r, dt, d_normals.getData(), N_STEPS, N_PATHS);
        cudaDeviceSynchronize();

        // copy results from device to host
        d_s.get(&s[0], N_PATHS);

        if (myid == nprocs - 1) {
            end_idx = N_PATHS;
        }

        // compute the payoff average
        // double temp_sum = 0.0;

        // MPI variables for local sum and final sum
        double local_sum = 0.0;
        double total_sum = 0.0;

        // for (size_t i = 0; i < N_PATHS; i++)
        for (size_t i = start_idx; i < end_idx; i++)
        {
            local_sum += s[i];
        }
        // temp_sum /= N_PATHS;

        MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (myid == 0)
        {
            total_sum /= (N_PATHS);
        
            double t4 = double(clock()) / CLOCKS_PER_SEC;

            // init variables for CPU Monte Carlo
            vector<float> normals(N_NORMALS);
            d_normals.get(&normals[0], N_NORMALS);
            double sum = 0.0;
            float s_curr = 0.0;

            // CPU Monte Carlo Simulation
            for (size_t i = 0; i < N_PATHS; i++)
            {
                int n_idx = i * N_STEPS;

                s_curr = S0;
                int n = 0;

                do
                {
                    s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * normals[n_idx];
                    n_idx++;
                    n++;
                } while (n < N_STEPS && s_curr > B);

                double payoff = (s_curr > K ? s_curr - K : 0.0);
                sum += exp(-r * T) * payoff;
            }

            sum /= N_PATHS;

            double t5 = double(clock()) / CLOCKS_PER_SEC;

            cout << "****************** INFO ******************\n";
            cout << "Number of Paths: " << N_PATHS << "\n";
            cout << "Underlying Initial Price: " << S0 << "\n";
            cout << "Strike: " << K << "\n";
            cout << "Barrier: " << B << "\n";
            cout << "Time to Maturity: " << T << " years\n";
            cout << "Risk-free Interest Rate: " << r << "%\n";
            cout << "Annual drift: " << mu << "%\n";
            cout << "Volatility: " << sigma << "%\n";
            cout << "****************** PRICE ******************\n";
            cout << "Option Price (GPU+MPI): " << total_sum << "\n";
            cout << "Option Price (CPU): " << sum << "\n";
            cout << "******************* TIME *****************\n";
            cout << "GPU Monte Carlo Computation: " << (t4 - t2) * 1e3 << " ms\n";
            cout << "CPU Monte Carlo Computation: " << (t5 - t4) * 1e3 << " ms\n";
            cout << "******************* END *****************\n";

            // destroy generator
            curandDestroyGenerator(curandGenerator);
        }
    }
    catch (exception &e)
    {
        cout << "exception: " << e.what() << "\n";
    }
}
