#define _CRT_SECURE_NO_WARNINGS // 避免 MSVC 对 strcpy 等函数的警告
#define _USE_MATH_DEFINES		// 使 <math.h> 中包含 M_PI 等常量 (Windows)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h> // 为了使用 bool
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include <new>
#include <limits.h>
#include <errno.h>
#include <ctype.h>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#define PI 3.1415926535
#define e 2.718281828
#ifndef NZ
#define NZ 940
#endif
#ifndef NZZ
#define NZZ 900
#endif
#ifndef NY
#define NY 900
#endif
#ifndef NX
#define NX 900
#endif
#ifndef NP
#define NP 30
#endif
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 1

// Labels 1..9: core; label 10: original average/background medium.
struct MaterialTable {
    float rho[11], mu[11], lambda[11], lambda2u[11];
};
__constant__ MaterialTable g_materialTable;
MaterialTable makeMaterialTable(const char* materialsPath);

#define CUDA_CHECK(stmt)                                      \
    do {                                                      \
        cudaError_t __err = (stmt);                           \
        if (__err != cudaSuccess) {                           \
            fprintf(stderr,"CUDA error %s at %s:%d\n",        \
                    cudaGetErrorString(__err), __FILE__,__LINE__); \
            return EXIT_FAILURE;                              \
        }                                                     \
    } while(0)

int readTextDat(uint8_t* output, size_t totalSize, const char* filename, int coordinateBase);
int divUp(int a, int b) { return (a + b - 1) / b; }
/**********************************************************************/
/*  辅助函数：把 (ix,iy,iz) 映射到紧凑 PML 数组的一维下标              */
/**********************************************************************/
/* 一侧面大小：NX_ext*NY_ext, NX_ext*NZ_ext, NY_ext*NZ_ext 均已乘 NP          */
__device__ __forceinline__
int idx_pml_x(int ix, int iy, int iz,
    int NX_ext, int NY_ext, int NZ_ext, int PLANE)
{
    if (ix < NP)                   // 左
        return ix + iy * NP + iz * NP * NY_ext;
    if (ix >= NX_ext - NP)           // 右
        return PLANE + ((iz * NY_ext) + iy) * NP + (ix - (NX_ext - NP));
    return -1;                     // 不在 X-PML
}

__device__ __forceinline__
int idx_pml_y(int ix, int iy, int iz,
    int NX_ext, int NY_ext, int NZ_ext, int PLANE)
{
    if (iy < NP)
        return ix + iy * NX_ext + iz * NP * NX_ext;
    if (iy >= NY_ext - NP)
        return PLANE + ix + iz * NP * NX_ext + (iy - (NY_ext - NP)) * NX_ext;
    return -1;
}
__device__ __forceinline__
int idx_pml_z(int ix, int iy, int iz,
    int NX_ext, int NY_ext, int NZ_ext, int PLANE)
{
    if (iz < NP)
        return ix + iy * NX_ext + iz * NX_ext * NY_ext;
    if (iz >= NZ_ext - NP)          // == NZ_ext-NP
        return PLANE + ix + iy * NX_ext + (iz - (NZ_ext - NP)) * NX_ext * NY_ext;
    return -1;
}
//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------
// 计算震源
// One source point only; avoid scanning the full volume on every time step.
__global__ void Source(float* txx, float* tyy, float* tzz, float amplitude, int sn)
{
    txx[sn] += amplitude;
    tyy[sn] += amplitude;
    tzz[sn] += amplitude;
}

// Launched after a complete update kernel in the same stream. Updating history
// inside FD_V/FD_T would race with other blocks reading history at neighboring cells.
__global__ void SaveStressHistory(const float* xx, const float* yy, const float* zz,
    const float* xy, const float* xz, const float* yz,
    float* oldxx, float* oldyy, float* oldzz, float* oldxy, float* oldxz, float* oldyz,
    size_t count)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
        i += (size_t)blockDim.x * gridDim.x) {
        oldxx[i] = xx[i]; oldyy[i] = yy[i]; oldzz[i] = zz[i];
        oldxy[i] = xy[i]; oldxz[i] = xz[i]; oldyz[i] = yz[i];
    }
}

__global__ void SaveVelocityHistory(const float* x, const float* y, const float* z,
    float* oldx, float* oldy, float* oldz, size_t count)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
        i += (size_t)blockDim.x * gridDim.x) {
        oldx[i] = x[i]; oldy[i] = y[i]; oldz[i] = z[i];
    }
}

__global__ void GatherOutput(const float* txx, const float* vux, const float* vuy,
    const float* vuz, float* trace, int sx, int sy, int NX_ext, int NY_ext,
    int NZ_ext)
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    if (iz >= NZ_ext)
        return;

    size_t offset = (size_t)iz * NX_ext * NY_ext + (size_t)sy * NX_ext + sx;
    trace[(size_t)iz * 4] = txx[offset];
    trace[(size_t)iz * 4 + 1] = vux[offset];
    trace[(size_t)iz * 4 + 2] = vuy[offset];
    trace[(size_t)iz * 4 + 3] = vuz[offset];
}

__global__ void FD_V(float* vux, float* vuy, float* vuz, const uint8_t* model_codes,
    float* txx, float* tyy, float* tzz, float* txz, float* txy, float* tyz,
    float* pmlxSxx, float* pmlySxy, float* pmlzSxz, float* pmlxSxy, float* pmlySyy, float* pmlzSyz, float* pmlxSxz, float* pmlySyz, float* pmlzSzz, float* midtxx, float* midtyy, float* midtzz, float* midtxy, float* midtxz, float* midtyz, float DT, float H)
{
    // Extended dimensions including PML
    const int NX_ext = NX + 2 * NP;
    const int NY_ext = NY + 2 * NP;
    const int NZ_ext = NZ + 2 * NP;
    const float DT2 = -DT * 0.5f;
    float e_dxi;
    float e_dxi2;
    float e_dyj;
    float e_dyj2;
    float e_dzk;
    float e_dzk2;
    float dxi;
    float dxi2;
    float dyj;
    float dyj2;
    float dzk;
    float dzk2;
    const double DP = NP * (double)H;
    float RC = 1.0 * pow(10.0, -6);
    const double v0 = 4000.0;
    const double d0 = 3 * v0 * log(1.0 / RC) / (2.0 * DP);

    const int PLANE_X = NP * NY_ext * NZ_ext;     // 一个侧壳面大小
    const int PLANE_Y = NP * NX_ext * NZ_ext;
    const int PLANE_Z = NP * NX_ext * NY_ext;
    // 1) Compute each thread's starting index in x,y,z
    int sx = threadIdx.x + blockIdx.x * blockDim.x;
    int sy = threadIdx.y + blockIdx.y * blockDim.y;
    int sz = threadIdx.z + blockIdx.z * blockDim.z;

    // 2) Compute the stepping stride in each dimension
    int step_x = blockDim.x * gridDim.x;
    int step_y = blockDim.y * gridDim.y;
    int step_z = blockDim.z * gridDim.z;

    // 3) Triple loop to cover the entire domain by strides
    for (int iz = sz; iz < NZ_ext; iz += step_z)
    {
        for (int iy = sy; iy < NY_ext; iy += step_y)
        {
            for (int ix = sx; ix < NX_ext; ix += step_x)
            {

                // 4) Compute flat offsets
                int base = ix + iy * NX_ext + iz * NX_ext * NY_ext;
                int off_r = base + 1;
                int off_l = base - 1;
                int off_q = base + NX_ext;
                int off_h = base - NX_ext;
                int off_u = base + NX_ext * NY_ext;
                int off_b = base - NX_ext * NY_ext;

                // only interior points
                if (ix == 0 || iy == 0 || iz == 0 ||
                    ix == NX_ext - 1 || iy == NY_ext - 1 || iz == NZ_ext - 1)
                    continue;

                float rho_tempx = 2 / (g_materialTable.rho[model_codes[base]] + g_materialTable.rho[model_codes[off_r]]);
                float rho_tempy = 2 / (g_materialTable.rho[model_codes[base]] + g_materialTable.rho[model_codes[off_q]]);
                float rho_tempz = 2 / (g_materialTable.rho[model_codes[base]] + g_materialTable.rho[model_codes[off_u]]);


                // 5) Spatial derivatives
                float x1 = (txx[off_r] - txx[base]) / H;
                float x2 = (txy[base] - txy[off_h]) / H;
                float x3 = (txz[base] - txz[off_b]) / H;

                float y1 = (tyy[off_q] - tyy[base]) / H;
                float y2 = (txy[base] - txy[off_l]) / H;
                float y3 = (tyz[base] - tyz[off_b]) / H;

                float z1 = (tzz[off_u] - tzz[base]) / H;
                float z2 = (txz[base] - txz[off_l]) / H;
                float z3 = (tyz[base] - tyz[off_h]) / H;

                float xx1 = (midtxx[off_r] - midtxx[base]) / H;
                float xx2 = (midtxy[base] - midtxy[off_h]) / H;
                float xx3 = (midtxz[base] - midtxz[off_b]) / H;

                float yy1 = (midtyy[off_q] - midtyy[base]) / H;
                float yy2 = (midtxy[base] - midtxy[off_l]) / H;
                float yy3 = (midtyz[base] - midtyz[off_b]) / H;

                float zz1 = (midtzz[off_u] - midtzz[base]) / H;
                float zz2 = (midtxz[base] - midtxz[off_l]) / H;
                float zz3 = (midtyz[base] - midtyz[off_h]) / H;

                // --- PML coefficients: reset then plug if/else‑if per axis ---
                dxi = dxi2 = dyj = dyj2 = dzk = dzk2 = 0.0f;   // ① reset

                /* 原始 off_*、导数、阻尼系数……  完全照搬你之前的代码  */
                /* --------- σ 系数计算略 ---------- */


                // ‑‑ X 方向边界 ‑‑
                if (ix < NP) {
                    float rx = (NP - ix) * H;
                    float rx2 = rx - 0.5f * H;
                    dxi = d0 * pow(rx / DP, 2);
                    dxi2 = d0 * pow(rx2 / DP, 2);
                    e_dxi = exp(-dxi * DT);
                    e_dxi2 = exp(-dxi2 * DT);


                }
                else if (ix >= NX_ext - NP) {
                    float rx = (ix - (NX_ext - 1 - NP)) * H;
                    float rx2 = rx + 0.5f * H;
                    dxi = d0 * pow(rx / DP, 2);
                    dxi2 = d0 * pow(rx2 / DP, 2);
                    e_dxi = exp(-dxi * DT);
                    e_dxi2 = exp(-dxi2 * DT);

                }


                // ‑‑ Y 方向边界 ‑‑
                if (iy < NP) {
                    float ry = (NP - iy) * H;
                    float ry2 = ry - 0.5f * H;
                    dyj = d0 * pow(ry / DP, 2);
                    dyj2 = d0 * pow(ry2 / DP, 2);
                    e_dyj = exp(-dyj * DT);
                    e_dyj2 = exp(-dyj2 * DT);

                }
                else if (iy >= NY_ext - NP) {
                    float ry = (iy - (NY_ext - 1 - NP)) * H;
                    float ry2 = ry + 0.5f * H;
                    dyj = d0 * pow(ry / DP, 2);
                    dyj2 = d0 * pow(ry2 / DP, 2);
                    e_dyj = exp(-dyj * DT);
                    e_dyj2 = exp(-dyj2 * DT);

                }
                // ‑‑ Z 方向边界 ‑‑
                if (iz < NP) {
                    float rz = (NP - iz) * H;
                    float rz2 = rz - 0.5f * H;
                    dzk = d0 * pow(rz / DP, 2);
                    dzk2 = d0 * pow(rz2 / DP, 2);
                    e_dzk = exp(-dzk * DT);
                    e_dzk2 = exp(-dzk2 * DT);
                }
                else if (iz >= NZ_ext - NP) {
                    float rz = (iz - (NZ_ext - 1 - NP)) * H;
                    float rz2 = rz + 0.5f * H;
                    dzk = d0 * pow(rz / DP, 2);
                    dzk2 = d0 * pow(rz2 / DP, 2);
                    e_dzk = exp(-dzk * DT);
                    e_dzk2 = exp(-dzk2 * DT);
                }


                int idX = idx_pml_x(ix, iy, iz, NX_ext, NY_ext, NZ_ext, PLANE_X);
                int idY = idx_pml_y(ix, iy, iz, NX_ext, NY_ext, NZ_ext, PLANE_Y);
                int idZ = idx_pml_z(ix, iy, iz, NX_ext, NY_ext, NZ_ext, PLANE_Z);
                if ((ix < NP || ix >= (NX_ext - 1 - NP)) || (iy < NP || iy >= (NY_ext - 1 - NP)) || (iz < NP || iz >= (NZ_ext - 1 - NP)))
                {
                    if (idX >= 0)
                    {
                        pmlxSxx[idX] = pmlxSxx[idX] * e_dxi2 + DT2 * dxi2 * (e_dxi2 * xx1 + x1);
                        x1 += pmlxSxx[idX];

                        pmlxSxy[idX] = pmlxSxy[idX] * e_dxi + DT2 * dxi * (e_dxi * yy2 + y2);
                        y2 += pmlxSxy[idX];

                        pmlxSxz[idX] = pmlxSxz[idX] * e_dxi + DT2 * dxi * (e_dxi * zz2 + z2);
                        z2 += pmlxSxz[idX];
                    }

                    if (idY >= 0)
                    {
                        pmlySxy[idY] = pmlySxy[idY] * e_dyj + DT2 * dyj * (e_dyj * xx2 + x2);
                        x2 += pmlySxy[idY];

                        pmlySyy[idY] = pmlySyy[idY] * e_dyj2 + DT2 * dyj2 * (e_dyj2 * yy1 + y1);
                        y1 += pmlySyy[idY];

                        pmlySyz[idY] = pmlySyz[idY] * e_dyj + DT2 * dyj * (e_dyj * zz3 + z3);
                        z3 += pmlySyz[idY];
                    }

                    if (idZ >= 0)
                    {
                        pmlzSxz[idZ] = pmlzSxz[idZ] * e_dzk + DT2 * dzk * (e_dzk * xx3 + x3);
                        x3 += pmlzSxz[idZ];

                        pmlzSyz[idZ] = pmlzSyz[idZ] * e_dzk + DT2 * dzk * (e_dzk * yy3 + y3);
                        y3 += pmlzSyz[idZ];

                        pmlzSzz[idZ] = pmlzSzz[idZ] * e_dzk2 + DT2 * dzk2 * (e_dzk2 * zz1 + z1);
                        z1 += pmlzSzz[idZ];
                    }
                }

                // 9) Update velocities
                vux[base] += DT * rho_tempx * (x1 + x2 + x3);
                vuy[base] += DT * rho_tempy * (y1 + y2 + y3);
                vuz[base] += DT * rho_tempz * (z1 + z2 + z3);
            }
        }
    }
}

__global__ void FD_T(float* vux, float* vuy, float* vuz, const uint8_t* model_codes,
    float* txx, float* tzz, float* tyy, float* txz, float* txy, float* tyz,
    float* pmlxVux, float* pmlyVuy, float* pmlzVuz, float* pmlxVuy, float* pmlyVux, float* pmlyVuz, float* pmlzVuy, float* pmlzVux, float* pmlxVuz, float* midvx, float* midvy, float* midvz, float DT, float H)
{
    // extended grid sizes
    const int NX_ext = NX + 2 * NP;
    const int NY_ext = NY + 2 * NP;
    const int NZ_ext = NZ + 2 * NP;
    const float DT2 = -DT * 0.5f;
    float RC = 1.0 * pow(10.0, -8);
    float DP = NP * H;
    float e_dxi;
    float e_dxi2;
    float e_dyj;
    float e_dyj2;
    float e_dzk;
    float e_dzk2;
    float dxi;
    float dxi2;
    float dyj;
    float dyj2;
    float dzk;
    float dzk2;
    const double v0 = 4000.0;
    const double d0 = 3 * v0 * log(1.0 / RC) / (2.0 * DP);

    // 1) each thread’s starting indices
    int sx = threadIdx.x + blockIdx.x * blockDim.x;
    int sy = threadIdx.y + blockIdx.y * blockDim.y;
    int sz = threadIdx.z + blockIdx.z * blockDim.z;

    // 2) strides in each dimension
    int step_x = blockDim.x * gridDim.x;
    int step_y = blockDim.y * gridDim.y;
    int step_z = blockDim.z * gridDim.z;

    const int PLANE_X = NP * NY_ext * NZ_ext;     // 一个侧壳面大小
    const int PLANE_Y = NP * NX_ext * NZ_ext;
    const int PLANE_Z = NP * NX_ext * NY_ext;
    float muxy, muxz, muyz;
    // 3) triple-loop with strides
    for (int iz = sz; iz < NZ_ext; iz += step_z)
    {
        for (int iy = sy; iy < NY_ext; iy += step_y)
        {
            for (int ix = sx; ix < NX_ext; ix += step_x)
            {

                // 4) flat indices
                int base = ix + iy * NX_ext + iz * NX_ext * NY_ext;
                int off_b = base - NX_ext * NY_ext;  // z-1
                int off_u = base + NX_ext * NY_ext;  // z+1
                int off_r = base + 1;              // x+1
                int off_l = base - 1;              // x-1
                int off_q = base + NX_ext;         // y+1
                int off_h = base - NX_ext;         // y-1
                int off_ur = base + 1 + NX_ext * NY_ext; // x+1 z+1
                int off_uq = base + NX_ext + NX_ext * NY_ext; // z+1 y+1
                int off_rq = base + 1 + NX_ext;

                // skip boundaries
                if (ix == 0 || iy == 0 || iz == 0 ||
                    ix == NX_ext - 1 || iy == NY_ext - 1 || iz == NZ_ext - 1) continue;

                const float mu_base = g_materialTable.mu[model_codes[base]];
                const float mu_r = g_materialTable.mu[model_codes[off_r]];
                const float mu_u = g_materialTable.mu[model_codes[off_u]];
                const float mu_ur = g_materialTable.mu[model_codes[off_ur]];
                const float mu_q = g_materialTable.mu[model_codes[off_q]];
                const float mu_uq = g_materialTable.mu[model_codes[off_uq]];
                const float mu_rq = g_materialTable.mu[model_codes[off_rq]];
                if (mu_base == 0 || mu_r == 0 || mu_u == 0 || mu_ur == 0)
                    muxz = 0.0;
                else
                    muxz = 1 / (0.25 * (1 / mu_base + 1 / mu_r + 1 / mu_u + 1 / mu_ur));

                if (mu_base == 0 || mu_q == 0 || mu_u == 0 || mu_uq == 0)
                    muyz = 0.0;
                else
                    muyz = 1 / (0.25 * (1 / mu_base + 1 / mu_q + 1 / mu_u + 1 / mu_uq));

                if (mu_base == 0 || mu_r == 0 || mu_q == 0 || mu_rq == 0)
                    muxy = 0.0;
                else
                    muxy = 1 / (0.25 * (1 / mu_base + 1 / mu_r + 1 / mu_q + 1 / mu_rq));

                // 5) compute strain increments
                float uxx = (vux[base] - vux[off_l]) / H;
                float uyy = (vuy[base] - vuy[off_h]) / H;
                float uzz = (vuz[base] - vuz[off_b]) / H;

                float uxy = (vux[off_q] - vux[base]) / H;
                float uyx = (vuy[off_r] - vuy[base]) / H;

                float uxz = (vux[off_u] - vux[base]) / H;
                float uzx = (vuz[off_r] - vuz[base]) / H;

                float uyz = (vuy[off_u] - vuy[base]) / H;
                float uzy = (vuz[off_q] - vuz[base]) / H;

                float uuxx = (midvx[base] - midvx[off_l]) / H;
                float uuyy = (midvy[base] - midvy[off_h]) / H;
                float uuzz = (midvz[base] - midvz[off_b]) / H;

                float uuxy = (midvx[off_q] - midvx[base]) / H;
                float uuyx = (midvy[off_r] - midvy[base]) / H;

                float uuxz = (midvx[off_u] - midvx[base]) / H;
                float uuzx = (midvz[off_r] - midvz[base]) / H;

                float uuyz = (midvy[off_u] - midvy[base]) / H;
                float uuzy = (midvz[off_q] - midvz[base]) / H;

                // 6) PML on cross-couplings
                // --- PML coefficients: reset then plug if/else‑if per axis ---
                dxi = dxi2 = dyj = dyj2 = dzk = dzk2 = 0.0f;   // ① reset

                // ‑‑ X 方向边界 ‑‑
                if (ix < NP) {
                    float rx = (NP - ix) * H;
                    float rx2 = rx - 0.5f * H;
                    dxi = d0 * pow(rx / DP, 2);
                    dxi2 = d0 * pow(rx2 / DP, 2);
                    e_dxi = exp(-dxi * DT);
                    e_dxi2 = exp(-dxi2 * DT);
                }
                else if (ix >= NX_ext - NP) {
                    float rx = (ix - (NX_ext - 1 - NP)) * H;
                    float rx2 = rx + 0.5f * H;
                    dxi = d0 * pow(rx / DP, 2);
                    dxi2 = d0 * pow(rx2 / DP, 2);
                    e_dxi = exp(-dxi * DT);
                    e_dxi2 = exp(-dxi2 * DT);
                }

                // ‑‑ Y 方向边界 ‑‑
                if (iy < NP) {
                    float ry = (NP - iy) * H;
                    float ry2 = ry - 0.5f * H;
                    dyj = d0 * pow(ry / DP, 2);
                    dyj2 = d0 * pow(ry2 / DP, 2);
                    e_dyj = exp(-dyj * DT);
                    e_dyj2 = exp(-dyj2 * DT);
                }
                else if (iy >= NY_ext - NP) {
                    float ry = (iy - (NY_ext - 1 - NP)) * H;
                    float ry2 = ry + 0.5f * H;
                    dyj = d0 * pow(ry / DP, 2);
                    dyj2 = d0 * pow(ry2 / DP, 2);
                    e_dyj = exp(-dyj * DT);
                    e_dyj2 = exp(-dyj2 * DT);

                }


                // ‑‑ Z 方向边界 ‑‑
                if (iz < NP) {
                    float rz = (NP - iz) * H;
                    float rz2 = rz - 0.5f * H;
                    dzk = d0 * pow(rz / DP, 2);
                    dzk2 = d0 * pow(rz2 / DP, 2);
                    e_dzk = exp(-dzk * DT);
                    e_dzk2 = exp(-dzk2 * DT);
                }
                else if (iz >= NZ_ext - NP) {
                    float rz = (iz - (NZ_ext - 1 - NP)) * H;
                    float rz2 = rz + 0.5f * H;
                    dzk = d0 * pow(rz / DP, 2);
                    dzk2 = d0 * pow(rz2 / DP, 2);
                    e_dzk = exp(-dzk * DT);
                    e_dzk2 = exp(-dzk2 * DT);
                }


                int idX = idx_pml_x(ix, iy, iz, NX_ext, NY_ext, NZ_ext, PLANE_X);
                int idY = idx_pml_y(ix, iy, iz, NX_ext, NY_ext, NZ_ext, PLANE_Y);
                int idZ = idx_pml_z(ix, iy, iz, NX_ext, NY_ext, NZ_ext, PLANE_Z);
                // --- 计算阻尼指数 (保持原变量名不变) ---
                if ((ix < NP || ix >= (NX_ext - 1 - NP)) || (iy < NP || iy >= (NY_ext - 1 - NP)) || (iz < NP || iz >= (NZ_ext - 1 - NP)))
                {
                    if (idX >= 0)
                    {
                        pmlxVux[idX] = pmlxVux[idX] * e_dxi + DT2 * dxi * (e_dxi * uuxx + uxx);
                        uxx += pmlxVux[idX];

                        pmlyVux[idX] = pmlyVux[idX] * e_dxi2 + DT2 * dxi2 * (e_dxi2 * uuyx + uyx);
                        uyx += pmlyVux[idX];

                        pmlzVux[idX] = pmlzVux[idX] * e_dxi2 + DT2 * dxi2 * (e_dxi2 * uuzx + uzx);
                        uzx += pmlzVux[idX];

                    }

                    if (idY >= 0)
                    {
                        pmlyVuy[idY] = pmlyVuy[idY] * e_dyj + DT2 * dyj * (e_dyj * uuyy + uyy);
                        uyy += pmlyVuy[idY];

                        pmlxVuy[idY] = pmlxVuy[idY] * e_dyj2 + DT2 * dyj2 * (e_dyj2 * uuxy + uxy);
                        uxy += pmlxVuy[idY];

                        pmlzVuy[idY] = pmlzVuy[idY] * e_dyj2 + DT2 * dyj2 * (e_dyj2 * uuzy + uzy);
                        uzy += pmlzVuy[idY];
                    }

                    if (idZ >= 0)
                    {
                        pmlzVuz[idZ] = pmlzVuz[idZ] * e_dzk + DT2 * dzk * (e_dzk * uuzz + uzz);
                        uzz += pmlzVuz[idZ];

                        pmlxVuz[idZ] = pmlxVuz[idZ] * e_dzk2 + DT2 * dzk2 * (e_dzk2 * uuxz + uxz);
                        uxz += pmlxVuz[idZ];

                        pmlyVuz[idZ] = pmlyVuz[idZ] * e_dzk2 + DT2 * dzk2 * (e_dzk2 * uuyz + uyz);
                        uyz += pmlyVuz[idZ];

                    }
                }

                // 7) update stress components
                const uint8_t material = model_codes[base];
                const float lambda = g_materialTable.lambda[material];
                const float lambda2u = g_materialTable.lambda2u[material];
                txx[base] += DT * (lambda * (uyy + uzz) + lambda2u * uxx);
                tyy[base] += DT * (lambda * (uxx + uzz) + lambda2u * uyy);
                tzz[base] += DT * (lambda * (uxx + uyy) + lambda2u * uzz);
                txy[base] += muxy * DT * (uxy + uyx);
                txz[base] += muxz * DT * (uxz + uzx);
                tyz[base] += muyz * DT * (uyz + uzy);
            }
        }
    }
}

// Own every successful device allocation, including partial initialization paths.
class DeviceBuffers {
    void* pointers[48] = {};
    int count = 0;
public:
    DeviceBuffers() = default;
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;
    ~DeviceBuffers() {
        for (int i = count - 1; i >= 0; --i)
            cudaFree(pointers[i]);
    }
    template<class T> cudaError_t allocate(T** ptr, size_t bytes) {
        if (count == 48) return cudaErrorInvalidValue;
        cudaError_t status = cudaMalloc(ptr, bytes);
        if (status == cudaSuccess) pointers[count++] = *ptr;
        return status;
    }
};

using FileHandle = std::unique_ptr<FILE, int(*)(FILE*)>;
const int MODEL_CHUNK_DEPTH = 8;
const int OUTPUT_INTERVAL = 500;
const double GIB = 1024.0 * 1024.0 * 1024.0;

uint8_t modelCodeAt(const uint8_t* codes, int z, int y, int x,
    int nz, int ny, int nx)
{
    z -= (nz - NZZ) / 2;
    y -= (ny - NY) / 2;
    x -= (nx - NX) / 2;
    if (z < 0 || z >= NZZ || y < 0 || y >= NY || x < 0 || x >= NX)
        return 10; // Average/background: reserve 7..9 for microporous minerals.
    return codes[((size_t)z * NY + y) * NX + x];
}

// Keep the original text format: one row per receiver (or z slice).
int writeMatrix(const char* filename, const float* data, int rows, int columns)
{
    FileHandle file(fopen(filename, "wt"), fclose);
    if (!file) {
        fprintf(stderr, "Cannot open output '%s'.\n", filename);
        return 0;
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < columns; ++col)
            fprintf(file.get(), "%e ", data[(size_t)row * columns + col]);
        fprintf(file.get(), "\n\n");
    }
    bool ok = !ferror(file.get());
    if (fclose(file.release()) != 0) ok = false;
    if (!ok) fprintf(stderr, "Failed to write output '%s'.\n", filename);
    return ok;
}

int uploadModel(const uint8_t* codes, int nz, int ny, int nx,
    uint8_t* d_model_codes, const MaterialTable& material)
{
    const size_t plane = (size_t)ny * nx;
    const size_t capacity = (size_t)MODEL_CHUNK_DEPTH * plane;
    std::vector<uint8_t> expanded(capacity);
    FileHandle rhoFile(fopen("rho_ext.dat", "wt"), fclose);
    FileHandle codeFile(fopen("elastic_pre_all.dat", "wt"), fclose);
    if (!rhoFile || !codeFile) {
        fprintf(stderr, "Cannot create model diagnostic output files.\n");
        return 0;
    }
    for (int z = 0; z < nz; z += MODEL_CHUNK_DEPTH) {
        const int depth = (z + MODEL_CHUNK_DEPTH <= nz) ? MODEL_CHUNK_DEPTH : nz - z;
        for (int localZ = 0; localZ < depth; ++localZ)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                    expanded[(size_t)localZ * plane + (size_t)y * nx + x] =
                    modelCodeAt(codes, z + localZ, y, x, nz, ny, nx);
        const size_t offset = (size_t)z * plane;
        const size_t bytes = (size_t)depth * plane * sizeof(uint8_t);
        // This helper returns a boolean, so do not use main's CUDA_CHECK macro.
        cudaError_t status = cudaMemcpy(d_model_codes + offset, expanded.data(), bytes, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            fprintf(stderr, "Material code upload failed at z=%d: %s\n", z, cudaGetErrorString(status));
            return 0;
        }
        for (int localZ = 0; localZ < depth; ++localZ) {
            const uint8_t* row = expanded.data() + (size_t)localZ * plane + (size_t)(ny / 2) * nx;
            for (int x = 0; x < nx; ++x) fprintf(rhoFile.get(), "%e ", material.rho[row[x]]);
            fprintf(rhoFile.get(), "\n\n");
            for (int y = 0; y < ny; ++y) {
                char textRow[4096];
                size_t used = 0;
                for (int x = 0; x < nx; ++x) {
                    const int written = snprintf(textRow + used, sizeof(textRow) - used, "%u ",
                        (unsigned)modelCodeAt(codes, z + localZ, y, x, nz, ny, nx));
                    if (written < 0 || (size_t)written >= sizeof(textRow) - used) {
                        fprintf(stderr, "Failed to format model diagnostic row.\n");
                        return 0;
                    }
                    used += (size_t)written;
                }
                if (fwrite(textRow, 1, used, codeFile.get()) != used ||
                    fputc('\n', codeFile.get()) == EOF) {
                    fprintf(stderr, "Failed to write model diagnostic output.\n");
                    return 0;
                }
            }
            if (fputc('\n', codeFile.get()) == EOF) {
                fprintf(stderr, "Failed to write model diagnostic output.\n");
                return 0;
            }
        }
    }
    bool ok = !ferror(rhoFile.get()) && !ferror(codeFile.get());
    if (fclose(rhoFile.release()) != 0) ok = false;
    if (fclose(codeFile.release()) != 0) ok = false;
    if (!ok) fprintf(stderr, "Failed to write model diagnostic output.\n");
    return ok;
}

int runSimulation(int argc, char** argv)
{
    // 四列文本：z y x 元素编号；按实际文件位置修改。
    const char* modelPath = "C:/model/第三次/model_random5_900x900x900.dat";
    std::string materialsPath; // 留空则自动使用模型同名的 _materials.csv
    const int modelCoordinateBase = 0; // 坐标 0..899 填 0；坐标 1..900 填 1。
    bool memoryReportOnly = false;
    int NT = 7100;
    for (int arg = 1; arg < argc; ++arg) {
        if (strcmp(argv[arg], "--memory-report") == 0) {
            memoryReportOnly = true;
        }
        else if (strcmp(argv[arg], "--model") == 0 && arg + 1 < argc) {
            modelPath = argv[++arg];
        }
        else if (strcmp(argv[arg], "--materials") == 0 && arg + 1 < argc) {
            materialsPath = argv[++arg];
        }
        else if (strcmp(argv[arg], "--steps") == 0 && arg + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long value = strtol(argv[++arg], &end, 10);
            if (errno == ERANGE || end == argv[arg] || *end != '\0' || value <= 0 || value > INT_MAX) {
                fprintf(stderr, "--steps requires a positive integer.\n");
                return EXIT_FAILURE;
            }
            NT = (int)value;
        }
        else {
            fprintf(stderr, "Usage: %s [--memory-report] [--model input.dat] [--materials input_materials.csv] [--steps N]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (materialsPath.empty()) {
        materialsPath = modelPath;
        const size_t dot = materialsPath.find_last_of('.');
        const size_t slash = materialsPath.find_last_of("/\\");
        if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
            materialsPath.erase(dot);
        materialsPath += "_materials.csv";
    }
    static_assert(NX > 0 && NY > 0 && NZ > 0 && NP > 0 && NZZ > 0 && NZZ <= NZ,
        "Invalid model dimensions.");
    static_assert((size_t)(NX + 2 * NP) * (NY + 2 * NP) * (NZ + 2 * NP) <= INT_MAX,
        "Grid exceeds the int indexing range in FD_V/FD_T.");
    const int NX_ext = NX + 2 * NP, NY_ext = NY + 2 * NP, NZ_ext = NZ + 2 * NP;
    const size_t gridSize = (size_t)NZ_ext * NY_ext * NX_ext;
    const size_t mem_size = gridSize * sizeof(float);
    const size_t mem_size_pmlx = sizeof(float) * 2u * NP * (size_t)NY_ext * NZ_ext;
    const size_t mem_size_pmly = sizeof(float) * 2u * NP * (size_t)NX_ext * NZ_ext;
    const size_t mem_size_pmlz = sizeof(float) * 2u * NP * (size_t)NX_ext * NY_ext;
    const size_t traceBytes = (size_t)NZ_ext * 4 * sizeof(float);
    // 9 current fields, 9 history fields, one byte per material code, and 18 compact PML arrays.
    const size_t requiredDeviceBytes = 18 * mem_size + gridSize +
        6 * (mem_size_pmlx + mem_size_pmly + mem_size_pmlz) + traceBytes;
    const size_t modelSize = (size_t)NZZ * NY * NX;
    const size_t materialChunkBytes = (size_t)MODEL_CHUNK_DEPTH * NY_ext * NX_ext * sizeof(uint8_t);
    const size_t buildHostBytes = modelSize + materialChunkBytes + 65536;
    const size_t runHostBytes = (size_t)NZ_ext * NT * 4 * sizeof(float) +
        (size_t)NZ_ext * NX_ext * sizeof(float) + traceBytes;
    const size_t hostPeakBytes = buildHostBytes > runHostBytes ? buildHostBytes : runHostBytes;
    printf("Grid: %d x %d x %d (%zu cells)\n", NX_ext, NY_ext, NZ_ext, gridSize);
    printf("Estimated GPU arrays: %.3f GiB; peak host arrays: %.3f GiB\n",
        requiredDeviceBytes / GIB, hostPeakBytes / GIB);
    printf("Material staging: %.2f MiB (%d z layers). Runtime/driver overhead excluded.\n",
        materialChunkBytes / (1024.0 * 1024.0), MODEL_CHUNK_DEPTH);
    fflush(stdout);
    if (memoryReportOnly) return EXIT_SUCCESS;

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t freeDeviceBytes = 0, totalDeviceBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeDeviceBytes, &totalDeviceBytes));
    printf("Device: %s; free %.2f / total %.2f GiB\n",
        prop.name, freeDeviceBytes / GIB, totalDeviceBytes / GIB);
    fflush(stdout);
    const size_t deviceReserve = 256u * 1024 * 1024;
    if (requiredDeviceBytes > freeDeviceBytes ||
        deviceReserve > freeDeviceBytes - requiredDeviceBytes) {
        fprintf(stderr, "Insufficient GPU memory before model loading: %.2f GiB arrays + "
            "0.25 GiB reserve required, %.2f GiB free.\n"
            "The current full-grid solver needs domain decomposition, multiple GPUs, "
            "or a larger GPU for this model.\n", requiredDeviceBytes / GIB, freeDeviceBytes / GIB);
        return EXIT_FAILURE;
    }
#ifdef _WIN32
    MEMORYSTATUSEX memoryStatus = {};
    memoryStatus.dwLength = sizeof(memoryStatus);
    if (GlobalMemoryStatusEx(&memoryStatus)) {
        const size_t hostReserve = (size_t)1024 * 1024 * 1024;
        if (hostPeakBytes + hostReserve > memoryStatus.ullAvailPhys) {
            fprintf(stderr, "Insufficient physical RAM: %.2f GiB arrays + 1 GiB reserve "
                "required, %.2f GiB available.\n", hostPeakBytes / GIB, memoryStatus.ullAvailPhys / GIB);
            return EXIT_FAILURE;
        }
    }
#endif

    const float H = 1.0 * pow(10.0, -5);
    const float DT = 5.0 * pow(10.0, -10);
    const float F0 = 10 * pow(10.0, 6);
    const float T0 = 1.2 / F0;
    const float pressure[] = { 0.1f };
    const float best_dt = 6.0 * H / (7.0 * sqrt(2.0) * 5560.6f);
    if (DT >= best_dt) fprintf(stderr, "Warning: DT exceeds stability estimate %e.\n", best_dt);
    if (1639.0f / (F0 * H) < 15) fprintf(stderr, "Warning: fewer than 15 cells per S wavelength.\n");
    const int sx = NX_ext / 2, sy = NY_ext / 2, sz = NP + 5;
    if (sz >= NZ_ext - 1) {
        fprintf(stderr, "Source lies outside the grid interior.\n");
        return EXIT_FAILURE;
    }
    const int sn = sx + sy * NX_ext + sz * NX_ext * NY_ext;
    const dim3 Blocksize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    const dim3 Gridsize(divUp(NX_ext, BLOCK_SIZE_X), divUp(NY_ext, BLOCK_SIZE_Y),
        divUp(NZ_ext, BLOCK_SIZE_Z) < 100 ? divUp(NZ_ext, BLOCK_SIZE_Z) : 100);

    for (size_t prenumber = 0; prenumber < sizeof(pressure) / sizeof(pressure[0]); ++prenumber) {
        printf("Pressure = %f\n", pressure[prenumber]);
        const int namepressure = (int)pressure[prenumber];
        const MaterialTable materialTable = makeMaterialTable(materialsPath.c_str());
        CUDA_CHECK(cudaMemcpyToSymbol(g_materialTable, &materialTable, sizeof(materialTable)));
        std::vector<uint8_t> modelCodes(modelSize);
        if (!readTextDat(modelCodes.data(), modelSize, modelPath, modelCoordinateBase))
            return EXIT_FAILURE;

        DeviceBuffers buffers;
        uint8_t* d_model_codes;
        float* d_vux, * d_vuy, * d_vuz;
        float* d_txx, * d_tyy, * d_tzz, * d_txy, * d_tyz, * d_txz;
        float* d_midtxx, * d_midtyy, * d_midtzz, * d_midtyz, * d_midtxy, * d_midtxz;
        float* d_midvx, * d_midvy, * d_midvz, * d_trace;
        float* d_pmlxSxx, * d_pmlySxy, * d_pmlzSxz, * d_pmlxSxy, * d_pmlySyy;
        float* d_pmlzSyz, * d_pmlxSxz, * d_pmlySyz, * d_pmlzSzz;
        float* d_pmlxVux, * d_pmlyVuy, * d_pmlzVuz, * d_pmlxVuy, * d_pmlyVux;
        float* d_pmlxVuz, * d_pmlzVux, * d_pmlyVuz, * d_pmlzVuy;
        CUDA_CHECK(buffers.allocate(&d_model_codes, gridSize * sizeof(uint8_t)));
        if (!uploadModel(modelCodes.data(), NZ_ext, NY_ext, NX_ext,
            d_model_codes, materialTable))
            return EXIT_FAILURE;
        // Release input and staging before allocating records or starting iteration.
        std::vector<uint8_t>().swap(modelCodes);

        float** fields[] = { &d_vux, &d_vuy, &d_vuz, &d_txx, &d_tyy, &d_tzz,
            &d_txy, &d_txz, &d_tyz, &d_midtxx, &d_midtyy, &d_midtzz,
            &d_midtyz, &d_midtxy, &d_midtxz, &d_midvx, &d_midvy, &d_midvz };
        for (auto ptr : fields) {
            CUDA_CHECK(buffers.allocate(ptr, mem_size));
            CUDA_CHECK(cudaMemset(*ptr, 0, mem_size));
        }
        // Size by derivative/index axis, not the first letter in the variable name.
        struct PmlSpan { float** ptr; size_t bytes; };
        PmlSpan pml[] = {
            {&d_pmlxSxx, mem_size_pmlx}, {&d_pmlxSxy, mem_size_pmlx},
            {&d_pmlxSxz, mem_size_pmlx}, {&d_pmlxVux, mem_size_pmlx},
            {&d_pmlyVux, mem_size_pmlx}, {&d_pmlzVux, mem_size_pmlx},
            {&d_pmlySxy, mem_size_pmly}, {&d_pmlySyy, mem_size_pmly},
            {&d_pmlySyz, mem_size_pmly}, {&d_pmlyVuy, mem_size_pmly},
            {&d_pmlxVuy, mem_size_pmly}, {&d_pmlzVuy, mem_size_pmly},
            {&d_pmlzSxz, mem_size_pmlz}, {&d_pmlzSyz, mem_size_pmlz},
            {&d_pmlzSzz, mem_size_pmlz}, {&d_pmlzVuz, mem_size_pmlz},
            {&d_pmlxVuz, mem_size_pmlz}, {&d_pmlyVuz, mem_size_pmlz}
        };
        for (auto span : pml) {
            CUDA_CHECK(buffers.allocate(span.ptr, span.bytes));
            CUDA_CHECK(cudaMemset(*span.ptr, 0, span.bytes));
        }
        CUDA_CHECK(buffers.allocate(&d_trace, traceBytes));
        std::vector<float> h_trace((size_t)NZ_ext * 4);
        std::vector<float> h_snapshot((size_t)NZ_ext * NX_ext);
        const size_t recordSize = (size_t)NZ_ext * NT;
        std::vector<float> sis_x(recordSize), sis_vx(recordSize), sis_vy(recordSize), sis_vz(recordSize);
        const clock_t start = clock();
        for (int it = 0; it < NT; ++it) {
            const float tt = it * DT;
            if (it % 100 == 0) printf("it=%d\n", it);
            const float I_sou = -(1 - 2 * PI * PI * F0 * F0 * (tt - T0) * (tt - T0)) *
                exp(-PI * PI * F0 * F0 * (tt - T0) * (tt - T0));
            Source << <1, 1 >> > (d_txx, d_tyy, d_tzz, I_sou, sn);
            CUDA_CHECK(cudaGetLastError());
            FD_V << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_model_codes,
                d_txx, d_tyy, d_tzz, d_txz, d_txy, d_tyz,
                d_pmlxSxx, d_pmlySxy, d_pmlzSxz, d_pmlxSxy, d_pmlySyy,
                d_pmlzSyz, d_pmlxSxz, d_pmlySyz, d_pmlzSzz,
                d_midtxx, d_midtyy, d_midtzz, d_midtxy, d_midtxz, d_midtyz, DT, H);
            CUDA_CHECK(cudaGetLastError());
            SaveStressHistory << <4096, 256 >> > (d_txx, d_tyy, d_tzz, d_txy, d_txz, d_tyz,
                d_midtxx, d_midtyy, d_midtzz, d_midtxy, d_midtxz, d_midtyz, gridSize);
            CUDA_CHECK(cudaGetLastError());
            FD_T << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_model_codes,
                d_txx, d_tzz, d_tyy, d_txz, d_txy, d_tyz,
                d_pmlxVux, d_pmlyVuy, d_pmlzVuz,
                d_pmlxVuy, d_pmlyVux, d_pmlyVuz, d_pmlzVuy, d_pmlzVux, d_pmlxVuz,
                d_midvx, d_midvy, d_midvz, DT, H);
            CUDA_CHECK(cudaGetLastError());
            SaveVelocityHistory << <4096, 256 >> > (d_vux, d_vuy, d_vuz, d_midvx, d_midvy, d_midvz, gridSize);
            CUDA_CHECK(cudaGetLastError());
            GatherOutput << <divUp(NZ_ext, 256), 256 >> > (d_txx, d_vux, d_vuy, d_vuz,
                d_trace, sx, sy, NX_ext, NY_ext, NZ_ext);
            CUDA_CHECK(cudaGetLastError());
            // Default-stream ordering plus this synchronous copy completes all updates.
            CUDA_CHECK(cudaMemcpy(h_trace.data(), d_trace, traceBytes, cudaMemcpyDeviceToHost));
            for (int z = 0; z < NZ_ext; ++z) {
                const size_t index = (size_t)z * NT + it;
                sis_x[index] = h_trace[(size_t)z * 4];
                sis_vx[index] = h_trace[(size_t)z * 4 + 1];
                sis_vy[index] = h_trace[(size_t)z * 4 + 2];
                sis_vz[index] = h_trace[(size_t)z * 4 + 3];
            }
            if (it % OUTPUT_INTERVAL == 0) {
                CUDA_CHECK(cudaMemcpy2D(h_snapshot.data(), (size_t)NX_ext * sizeof(float),
                    d_txx + (size_t)(NY_ext / 2) * NX_ext,
                    (size_t)NX_ext * NY_ext * sizeof(float), (size_t)NX_ext * sizeof(float),
                    NZ_ext, cudaMemcpyDeviceToHost));
                char filename[80];
                snprintf(filename, sizeof(filename), "txx%d %d.dat", it, namepressure);
                if (!writeMatrix(filename, h_snapshot.data(), NZ_ext, NX_ext)) return EXIT_FAILURE;
                snprintf(filename, sizeof(filename), "sisx %d.dat", namepressure);
                if (!writeMatrix(filename, sis_x.data(), NZ_ext, NT)) return EXIT_FAILURE;
            }
        }
        const char* names[] = { "sisx", "sisvx", "sisvy", "sisvz" };
        const float* records[] = { sis_x.data(), sis_vx.data(), sis_vy.data(), sis_vz.data() };
        for (int component = 0; component < 4; ++component) {
            char filename[80];
            snprintf(filename, sizeof(filename), "%s %d.dat", names[component], namepressure);
            if (!writeMatrix(filename, records[component], NZ_ext, NT)) return EXIT_FAILURE;
        }
        printf("Completed %d steps in %.3f seconds.\n", NT, (double)(clock() - start) / CLOCKS_PER_SEC);
    }
    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    try {
        return runSimulation(argc, argv);
    }
    catch (const std::exception& error) {
        fprintf(stderr, "Host allocation/runtime failure: %s\n", error.what());
        return EXIT_FAILURE;
    }
}

// Read the actual connect.m output for this model and alpha, without applying MT twice.
static std::vector<std::string> csvFields(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') { field += '"'; ++i; }
            else quoted = !quoted;
        } else if (c == ',' && !quoted) { fields.push_back(field); field.clear(); }
        else if (c != '\r') field += c;
    }
    if (quoted) throw std::runtime_error("Unclosed CSV quote.");
    fields.push_back(field);
    return fields;
}
static double csvNumber(const std::string& field)
{
    char* end = nullptr;
    errno = 0;
    const double v = strtod(field.c_str(), &end);
    if (end == field.c_str() || errno == ERANGE || !isfinite(v))
        throw std::runtime_error("Invalid material CSV number.");
    while (isspace((unsigned char)*end)) ++end;
    if (*end) throw std::runtime_error("Invalid material CSV number suffix.");
    return v;
}
MaterialTable makeMaterialTable(const char* materialsPath)
{
    std::ifstream file(materialsPath);
    if (!file) throw std::runtime_error(std::string("Cannot open material CSV: ") + materialsPath);
    std::string line;
    if (!std::getline(file, line)) throw std::runtime_error("Empty material CSV.");
    if (line.compare(0, 3, "\xef\xbb\xbf") == 0) line.erase(0, 3);
    const std::vector<std::string> header = csvFields(line);
    const char* required[] = { "Label", "Vp_m_s", "Vs_m_s", "Density_kg_m3" };
    size_t column[4];
    for (int k = 0; k < 4; ++k) {
        column[k] = header.size();
        for (size_t j = 0; j < header.size(); ++j)
            if (header[j] == required[k]) column[k] = j;
        if (column[k] == header.size())
            throw std::runtime_error(std::string("Missing material CSV column: ") + required[k]);
    }
    MaterialTable table = {};
    bool seen[11] = {};
    printf("Material CSV: %s\nLabel  Vp(m/s)  Vs(m/s)  Density(kg/m3)\n", materialsPath);
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        const std::vector<std::string> fields = csvFields(line);
        if (fields.size() != header.size()) throw std::runtime_error("Material CSV column count mismatch.");
        const double raw = csvNumber(fields[column[0]]);
        if (raw < 1 || raw > 9 || floor(raw) != raw)
            throw std::runtime_error("Material CSV requires labels 1..9 only.");
        const int label = (int)raw;
        if (seen[label]) throw std::runtime_error("Duplicate material CSV label.");
        const float vp = (float)csvNumber(fields[column[1]]);
        const float vs = (float)csvNumber(fields[column[2]]);
        const float rho = (float)csvNumber(fields[column[3]]);
        if (!isfinite(vp) || !isfinite(vs) || !isfinite(rho) || vp <= 0 || vs < 0 || rho <= 0 ||
            (double)vp * vp <= (4.0 / 3.0) * vs * vs)
            throw std::runtime_error("Invalid material speed/density or nonpositive bulk modulus.");
        table.rho[label] = rho;
        table.mu[label] = rho * vs * vs;
        table.lambda2u[label] = rho * vp * vp;
        table.lambda[label] = table.lambda2u[label] - 2.0f * table.mu[label];
        if (!isfinite(table.lambda2u[label]) || !isfinite(table.mu[label]))
            throw std::runtime_error("Material elastic coefficients overflow.");
        seen[label] = true;
        printf("%d  %.7g  %.7g  %.7g\n", label, vp, vs, rho);
    }
    if (file.bad()) throw std::runtime_error("Material CSV I/O failure.");
    for (int label = 1; label <= 9; ++label)
        if (!seen[label]) throw std::runtime_error("Material CSV must contain all labels 1..9.");
    // 原平均介质从 7 移到 10，物性保持原值。
    table.rho[10] = 2292.7f;
    table.mu[10] = 2292.7f * 1700.0f * 1700.0f;
    table.lambda2u[10] = 2292.7f * 3000.0f * 3000.0f;
    table.lambda[10] = table.lambda2u[10] - 2.0f * table.mu[10];
    fflush(stdout);
    return table;
}

// 四列文本导入；不要求行顺序，内部仍采用 x 最快的一维 uint8_t 存储。
// 元素范围 1..9（填充介质 10 由程序内部生成），0 用于识别尚未读入的网格。
int readTextDat(uint8_t* output, size_t totalSize, const char* filename, int coordinateBase)
{
    if (!output || !filename || totalSize != (size_t)NZZ * NY * NX ||
        (coordinateBase != 0 && coordinateBase != 1)) {
        fprintf(stderr, "Invalid text model reader arguments.\n");
        return 0;
    }
    FileHandle file(fopen(filename, "rt"), fclose);
    if (!file) {
        fprintf(stderr, "Cannot open model '%s': ", filename);
        perror(nullptr);
        return 0;
    }
    // 缓冲区先于文件析构而失效的问题通过使用 stdio 自有缓冲区避免。
    setvbuf(file.get(), nullptr, _IOFBF, 65536);
    memset(output, 0, totalSize);
    printf("Reading text model: %s\nExpected %zu cells; coordinates start at %d.\n",
        filename, totalSize, coordinateBase);
    fflush(stdout);
    char line[1024];
    size_t lineNumber = 0, count = 0;
    while (fgets(line, sizeof(line), file.get())) {
        ++lineNumber;
        if (!strchr(line, '\n') && !feof(file.get()) && strlen(line) == sizeof(line) - 1) {
            fprintf(stderr, "Model line %zu is too long.\n", lineNumber);
            return 0;
        }
        char* cursor = line;
        // 允许文本文件开头有 UTF-8 BOM。
        if (lineNumber == 1 && strlen(line) >= 3 &&
            (unsigned char)line[0] == 0xef && (unsigned char)line[1] == 0xbb &&
            (unsigned char)line[2] == 0xbf) cursor += 3;
        while (isspace((unsigned char)*cursor)) ++cursor;
        if (!*cursor) continue;
        double value[4];
        for (int col = 0; col < 4; ++col) {
            char* end = nullptr;
            errno = 0;
            value[col] = strtod(cursor, &end);
            if (end == cursor || errno == ERANGE || !isfinite(value[col]) ||
                floor(value[col]) != value[col] || (*end && !isspace((unsigned char)*end))) {
                fprintf(stderr, "Invalid integer at line %zu, column %d.\n", lineNumber, col + 1);
                return 0;
            }
            cursor = end;
        }
        while (isspace((unsigned char)*cursor)) ++cursor;
        if (*cursor) {
            fprintf(stderr, "Expected exactly four columns at line %zu.\n", lineNumber);
            return 0;
        }
        const int limits[3] = { NZZ, NY, NX };
        for (int axis = 0; axis < 3; ++axis) {
            if (value[axis] < coordinateBase || value[axis] >= (double)limits[axis] + coordinateBase) {
                fprintf(stderr, "Coordinate out of range at line %zu (z y x; base=%d).\n",
                    lineNumber, coordinateBase);
                return 0;
            }
        }
        if (value[3] < 1 || value[3] > 9) {
            fprintf(stderr, "Material at line %zu is outside 1..9 (core material table).\n", lineNumber);
            return 0;
        }
        const size_t z = (size_t)(value[0] - coordinateBase);
        const size_t y = (size_t)(value[1] - coordinateBase);
        const size_t x = (size_t)(value[2] - coordinateBase);
        const size_t index = (z * NY + y) * NX + x;
        if (output[index] != 0) {
            fprintf(stderr, "Duplicate model coordinate at line %zu.\n", lineNumber);
            return 0;
        }
        output[index] = (uint8_t)value[3];
        ++count;
        if (count % 1000000 == 0) {
            printf("Model input: %zu / %zu (%.2f%%)\n", count, totalSize, 100.0 * count / totalSize);
            fflush(stdout);
        }
    }
    if (ferror(file.get())) {
        fprintf(stderr, "I/O error reading model '%s'.\n", filename);
        return 0;
    }
    if (count != totalSize) {
        fprintf(stderr, "Incomplete model: read %zu cells, expected %zu; missing %zu.\n",
            count, totalSize, totalSize - count);
        return 0;
    }
    printf("Model input complete: %zu cells [z y x]=[%d %d %d].\n", count, NZZ, NY, NX);
    fflush(stdout);
    return 1;
}
