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
#include <omp.h>
using namespace std;

int main()
{
    try
    {
        // declare variables and constants
        const size_t N_PATHS = 100000;
        const size_t N_STEPS = 365;
        const size_t N_NORMALS = N_PATHS * N_STEPS;
        int N_THREADS;
        cudaGetDeviceCount(&N_THREADS);
        const float T = 1.0f;
        const float K = 100.0f;
        const float B = 95.0f;
        const float S0 = 100.0f;
        const float sigma = 0.2f;
        const float mu = 0.1f;
        const float r = 0.05f;
        float dt = float(T) / float(N_STEPS);
        float sqrdt = sqrt(dt);
        int thread_count;
        double sum;
        cout<<"Total number of CPUs: "<<omp_get_num_procs()<<"\n";
        cout<<"max threads available: "<<omp_get_max_threads()<<" "<<N_THREADS<<"\n";

        double t2 = double(clock()) / CLOCKS_PER_SEC;
// generate arrays
#pragma omp parallel reduction(+ : sum)
        {
#pragma omp single
            {
                thread_count = omp_get_num_threads();
            }
            int thread_paths = N_PATHS/thread_count;
            int thread_normals = N_NORMALS/thread_count;
            cudaSetDevice(omp_get_thread_num());
            vector<float> s(thread_paths);
            dev_array<float> d_s(thread_paths);
            dev_array<float> d_normals(thread_normals);

            curandGenerator_t curandGenerator;
            curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
            curandSetPseudoRandomGeneratorSeed(curandGenerator, omp_get_thread_num());
            curandGenerateNormal(curandGenerator, d_normals.getData(), thread_normals, 0.0f, sqrdt);

            // call the kernel
            mc_dao_call(d_s.getData(), T, K, B, S0, sigma, mu, r, dt, d_normals.getData(), N_STEPS, thread_paths);
            cudaDeviceSynchronize();

            // copy results from device to host
            d_s.get(&s[0], thread_paths);

            // compute the payoff average
            sum = 0.0;
            for (size_t i = 0; i < thread_paths; i++)
            {
                sum += s[i];
            }
            sum /= thread_paths;
            cout<<"thread "<<omp_get_thread_num()<<" option price: "<<sum<<endl;
            curandDestroyGenerator(curandGenerator);
        }

        sum /= thread_count;
        double t4 = double(clock()) / CLOCKS_PER_SEC;
        // init variables for CPU Monte Carlo

        cout << "****************** INFO ******************\n";
        cout << "Number of Threads and devices: " << thread_count <<" "<< N_THREADS <<"\n";
        cout << "Number of Paths: " << N_PATHS << "\n";
        cout << "Underlying Initial Price: " << S0 << "\n";
        cout << "Strike: " << K << "\n";
        cout << "Barrier: " << B << "\n";
        cout << "Time to Maturity: " << T << " years\n";
        cout << "Risk-free Interest Rate: " << r << "%\n";
        cout << "Annual drift: " << mu << "%\n";
        cout << "Volatility: " << sigma << "%\n";
        cout << "****************** PRICE ******************\n";
        cout << "Option Price (MPI): " << sum << "\n";
        cout << "******************* TIME *****************\n";
        cout << "MPI Monte Carlo Computation: " << (t4 - t2) * 1e3 << " ms\n";
        cout << "******************* END *****************\n";

        // destroy generator
    }
    catch (exception &e)
    {
        cout << "exception: " << e.what() << "\n";
    }
}