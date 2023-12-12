#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

void mc_daoc_call(float * d_s, float T, float K, float B, float S0, float sigma, float mu, float r, float dt, float* d_normals, unsigned N_STEPS, unsigned N_PATHS);
void mc_uaop_call(float * d_s, float T, float K, float B, float S0, float sigma, float mu, float r, float dt, float* d_normals, unsigned N_STEPS, unsigned N_PATHS);
void mc_uaic_call(float * d_s, float T, float K, float B, float S0, float sigma, float mu, float r, float dt, float* d_normals, unsigned N_STEPS, unsigned N_PATHS);
void mc_daip_call(float * d_s, float T, float K, float B, float S0, float sigma, float mu, float r, float dt, float* d_normals, unsigned N_STEPS, unsigned N_PATHS);
#endif