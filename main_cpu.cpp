// commit test

#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <string.h>

using namespace std;

int main(int argc, char *argv[])
{
    try
    {
        // declare variables and constants

        size_t N_PATHS = 100000;
        size_t N_STEPS = 365;

        float T = 1.0f;
        float K = 100.0f;
        float B = 95.0f;
        float S0 = 100.0f;
        float sigma = 0.2f;
        float mu = 0.1f;
        float r = 0.05f;
        float dt = float(T) / float(N_STEPS);
        float sqrdt = sqrt(dt);
        const std::vector<char *> args(argv + 1, argv + argc);
        for (auto it = args.begin(), end = args.end(); it != end; ++it)
        {
            if (strcmp("daoc", *it) == 0)
            {
                mc_call = mc_daoc_call;
                mu = 0.1f;
                B = 95.0f;
            }
            if (strcmp("uaop", *it) == 0)
            {
                mc_call = mc_uaop_call;
                mu = -0.1f;
                B = 105.0f;
            }
            if (strcmp("uaic", *it) == 0)
            {
                mc_call = mc_uaic_call;
                mu = 0.1f;
                B = 105.0f;
            }
            if (strcmp("daip", *it) == 0)
            {
                mc_call = mc_daip_call;
                mu = -0.1f;
                B = 95.0f;
            }
            if (strcmp("-B", *it) == 0)
                if (it + 1 != end)
                    B = stof(*(it + 1));
            if (strcmp("-K", *it) == 0)
                if (it + 1 != end)
                {

                    K = stof(*(it + 1));
                    S0 = K;
                }
            if (strcmp("-N", *it) == 0)
                if (it + 1 != end)
                {
                    N_PATHS = stof(*(it + 1));
                }
        }
        // generate arrays
        size_t N_NORMALS = N_PATHS * N_STEPS;

        double t4 = double(clock()) / CLOCKS_PER_SEC;

        init variables for CPU Monte Carlo
        vector<float> normals(N_NORMALS);
        d_normals.get(&normals[0], N_NORMALS);
        double sum = 0.0;
        float s_curr = 0.0;

        CPU Monte Carlo Simulation
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
        // cout << "Option Price (GPU): " << temp_sum << "\n";
        cout << "Option Price (CPU): " << sum << "\n";
        cout << "******************* TIME *****************\n";
        // cout << "GPU Monte Carlo Computation: " << (t4 - t2) * 1e3 << " ms\n";
        cout << "CPU Monte Carlo Computation: " << (t5 - t4) * 1e3 << " ms\n";
        cout << "******************* END *****************\n";

        // destroy generator
        // curandDestroyGenerator(curandGenerator);
    }
    catch (exception &e)
    {
        cout << "exception: " << e.what() << "\n";
    }
}
