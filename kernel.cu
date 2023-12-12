#include "kernel.h"
#include <stdio.h>

__global__ void mc_daoc_kernel( // Down and out call
    float *d_s,
    float T,
    float K,
    float B,
    float S0,
    float sigma,
    float mu,
    float r,
    float dt,
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned bsz = blockDim.x;
    int s_idx = tid + bid * bsz;
    int n_idx = tid + bid * bsz;
    float s_curr = S0;
    if (s_idx < N_PATHS)
    {
        int n = 0;
        do
        {
            s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * d_normals[n_idx];
            n_idx++;
            n++;
        } while (n < N_STEPS && s_curr > B);
        double payoff = (s_curr > K ? s_curr - K : 0.0);
        __syncthreads();
        d_s[s_idx] = exp(-r * T) * payoff;
    }
}

__global__ void mc_uaop_kernel( // Up and out put
    float *d_s,
    float T,
    float K,
    float B,
    float S0,
    float sigma,
    float mu,
    float r,
    float dt,
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned bsz = blockDim.x;
    int s_idx = tid + bid * bsz;
    int n_idx = tid + bid * bsz;
    float s_curr = S0;
    if (s_idx < N_PATHS)
    {
        int n = 0;
        do
        {
            s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * d_normals[n_idx];
            n_idx++;
            n++;
        } while (n < N_STEPS && s_curr < B);
        double payoff = (s_curr < K ? K - s_curr : 0.0);
        __syncthreads();
        d_s[s_idx] = exp(-r * T) * payoff;
    }
}

__global__ void mc_uaic_kernel( // Up and in call
    float *d_s,
    float T,
    float K,
    float B,
    float S0,
    float sigma,
    float mu,
    float r,
    float dt,
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned bsz = blockDim.x;
    int s_idx = tid + bid * bsz;
    int n_idx = tid + bid * bsz;
    float s_curr = S0;
    if (s_idx < N_PATHS)
    {
        int n = 0;
        bool b_crossed = false;
        while (n < N_STEPS)
        {
            s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * d_normals[n_idx];
            n_idx++;
            n++;
            if (s_curr > B)
                b_crossed = true;
        }
        double payoff = (b_crossed && s_curr > K ? s_curr - K : 0.0);
        __syncthreads();
        d_s[s_idx] = exp(-r * T) * payoff;
    }
}

__global__ void mc_daip_kernel( // Down and in put
    float *d_s,
    float T,
    float K,
    float B,
    float S0,
    float sigma,
    float mu,
    float r,
    float dt,
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned bsz = blockDim.x;
    int s_idx = tid + bid * bsz;
    int n_idx = tid + bid * bsz;
    float s_curr = S0;
    if (s_idx < N_PATHS)
    {
        int n = 0;
        bool b_crossed = false;
        while (n < N_STEPS)
        {
            s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * d_normals[n_idx];
            n_idx++;
            n++;
            if (s_curr < B)
                b_crossed = true;
        }
        double payoff = (b_crossed && s_curr < K ? K - s_curr : 0.0);
        __syncthreads();
        d_s[s_idx] = exp(-r * T) * payoff;
    }
}

void mc_daoc_call(
    float *d_s,
    float T,
    float K,
    float B,
    float S0,
    float sigma,
    float mu,
    float r,
    float dt,
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    if (B > S0)
    {
        printf("error: B > S0.\n");
        return;
    }
    const unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
    mc_daoc_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        d_s, T, K, B, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
}

void mc_uaop_call(
    float *d_s,
    float T,
    float K,
    float B,
    float S0,
    float sigma,
    float mu,
    float r,
    float dt,
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    if (B < S0)
    {
        printf("error: B < S0.\n");
        return;
    }
    const unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
    mc_uaop_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        d_s, T, K, B, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
}

void mc_uaic_call(
    float *d_s,
    float T,
    float K,
    float B,
    float S0,
    float sigma,
    float mu,
    float r,
    float dt,
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    if (B < S0)
    {
        printf("error: B < S0.\n");
        return;
    }
    const unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
    mc_uaic_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        d_s, T, K, B, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
}

void mc_daip_call(
    float *d_s,
    float T,
    float K,
    float B,
    float S0,
    float sigma,
    float mu,
    float r,
    float dt,
    float *d_normals,
    unsigned N_STEPS,
    unsigned N_PATHS)
{
    if (B > S0)
    {
        printf("error: B > S0.\n");
        return;
    }
    const unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
    mc_daip_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        d_s, T, K, B, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
}