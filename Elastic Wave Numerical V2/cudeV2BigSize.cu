#define _CRT_SECURE_NO_WARNINGS // 避免 MSVC 对 strcpy 等函数的警告
#define _USE_MATH_DEFINES		// 使 <math.h> 中包含 M_PI 等常量 (Windows)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h> // 为了使用 bool
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma argsused
#include <string.h>
#include <iostream>
#define PI 3.1415926535
#define e 2.718281828
#define NZ 540
#define NZZ 500
#define NY 500
#define NX 500
#define NP 30
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 1

#define CUDA_CHECK(stmt)                                      \
    do {                                                      \
        cudaError_t __err = (stmt);                           \
        if (__err != cudaSuccess) {                           \
            fprintf(stderr,"CUDA error %s at %s:%d\n",        \
                    cudaGetErrorString(__err), __FILE__,__LINE__); \
            return EXIT_FAILURE;                              \
        }                                                     \
    } while(0)

float** space2d(int nr, int nc);
float*** space3d(int nr, int ny, int nc);
void free_space2d(float** a, int nr);
void free_space3d(float*** b, int nr, int ny);
void wfile(char filename[], float** data, int nr, int nc);
void wfile3d(char filename[], float*** data, int nr, int ny, int nc);
void wfile1d(char filename[], float* data, int nr, int ny, int nc);
void create_model(float*** vp, float*** vs, float* rho, float*** vf, float*** rhof, float* lamda, float* lamda2u, float* mu, int nr, int ny, int nc, float pressure, float*** elastic_pre);
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
__global__ void Source(float* txx, float* tyy, float* tzz, float I_sou, int sn, int NX_ext, int NY_ext, int NZ_ext)
{
	// 1) each thread’s starting indices in x,y,z
	int sx = threadIdx.x + blockIdx.x * blockDim.x;
	int sy = threadIdx.y + blockIdx.y * blockDim.y;
	int sz = threadIdx.z + blockIdx.z * blockDim.z;

	// 2) strides in each dimension
	int step_x = blockDim.x * gridDim.x;
	int step_y = blockDim.y * gridDim.y;
	int step_z = blockDim.z * gridDim.z;

	// 3) triple-loop to cover entire domain
	for (int ix = sx; ix < NX_ext; ix += step_x) {
		for (int iy = sy; iy < NY_ext; iy += step_y) {
			for (int iz = sz; iz < NZ_ext; iz += step_z) {
				int offset = ix
					+ iy * NX_ext
					+ iz * (NX_ext * NY_ext);

				// add or subtract source at the two locations
				if (offset == sn) {
					txx[offset] += I_sou;
					tyy[offset] += I_sou;
					tzz[offset] += I_sou;
				}
			}
		}
	}
}

__global__ void FD_V(float* vux, float* vuy, float* vuz, float* rho_ext,
	float* txx, float* tyy, float* tzz, float* txz, float* txy, float* tyz,
	float* pmlxSxx, float* pmlySxy, float* pmlzSxz, float* pmlxSxy, float* pmlySyy, float* pmlzSyz, float* pmlxSxz, float* pmlySyz, float* pmlzSzz, float* midtxx, float* midtyy, float* midtzz, float* midtxy, float* midtxz, float* midtyz, float DT, float H)
{
	// Extended dimensions including PML
	const int NX_ext = NX + 2 * NP;
	const int NY_ext = NY + 2 * NP;
	const int NZ_ext = NZ + 2 * NP;
	const float DT2 = -DT * 0.5f;
	float x, y, z;
	float xoleft, xoright, yoleft, yoright, zoleft, zoright;
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

	xoleft = DP;
	xoright = (NX_ext - 1) * H - DP;
	yoleft = DP;
	yoright = (NY_ext - 1) * H - DP;
	zoleft = DP;
	zoright = (NZ_ext - 1) * H - DP;
	const int PLANE_X = NP * NY_ext * NZ_ext;     // 一个侧壳面大小
	const int PLANE_Y = NP * NX_ext * NZ_ext;
	const int PLANE_Z = NP * NX_ext * NY_ext;
	const int PML_STRIDE_X = 2 * PLANE_X;
	const int PML_STRIDE_Y = 2 * PLANE_Y;
	const int PML_STRIDE_Z = 2 * PLANE_Z;
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

				float rho_tempx = 2 / (rho_ext[base] + rho_ext[off_r]);
				float rho_tempy = 2 / (rho_ext[base] + rho_ext[off_q]);
				float rho_tempz = 2 / (rho_ext[base] + rho_ext[off_u]);


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
	for (int ix = sx; ix < NX_ext; ix += step_x) {
		for (int iy = sy; iy < NY_ext; iy += step_y) {
			for (int iz = sz; iz < NZ_ext; iz += step_z) {
				int base = ix + iy * NX_ext + iz * NX_ext * NY_ext;
				midtxx[base] = txx[base];
				midtyy[base] = tyy[base];
				midtzz[base] = tzz[base];
				midtxy[base] = txy[base];
				midtxz[base] = txz[base];
				midtyz[base] = tyz[base];
			}
		}
	}
}

__global__ void FD_T(float* vux, float* vuy, float* vuz, float* mu_ext,
	float* txx, float* tzz, float* tyy, float* txz, float* txy, float* tyz, float* lamda2u_ext, float* lamda_ext,
	float* pmlxVux, float* pmlyVuy, float* pmlzVuz, float* pmlxVuy, float* pmlyVux, float* pmlyVuz, float* pmlzVuy, float* pmlzVux, float* pmlxVuz, float* midvx, float* midvy, float* midvz, float DT, float H)
{
	// extended grid sizes
	const int NX_ext = NX + 2 * NP;
	const int NY_ext = NY + 2 * NP;
	const int NZ_ext = NZ + 2 * NP;
	const float DT2 = -DT * 0.5f;
	float x, y, z;
	float RC = 1.0 * pow(10.0, -8);
	float DP = NP * H;
	float xoleft, xoright, yoleft, yoright, zoleft, zoright;
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
	xoleft = DP;
	xoright = (NX_ext - 1) * H - DP;
	yoleft = DP;
	yoright = (NY_ext - 1) * H - DP;
	zoleft = DP;
	zoright = (NZ_ext - 1) * H - DP;

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
	const int PML_STRIDE_X = 2 * PLANE_X;
	const int PML_STRIDE_Y = 2 * PLANE_Y;
	const int PML_STRIDE_Z = 2 * PLANE_Z;
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

				if (mu_ext[base] == 0 || mu_ext[off_r] == 0 || mu_ext[off_u] == 0 || mu_ext[off_ur] == 0)
					muxz = 0.0;
				else
					muxz = 1 / (0.25 * (1 / mu_ext[base] + 1 / mu_ext[off_r] + 1 / mu_ext[off_u] + 1 / mu_ext[off_ur]));

				if (mu_ext[base] == 0 || mu_ext[off_q] == 0 || mu_ext[off_u] == 0 || mu_ext[off_uq] == 0)
					muyz = 0.0;
				else
					muyz = 1 / (0.25 * (1 / mu_ext[base] + 1 / mu_ext[off_q] + 1 / mu_ext[off_u] + 1 / mu_ext[off_uq]));

				if (mu_ext[base] == 0 || mu_ext[off_r] == 0 || mu_ext[off_q] == 0 || mu_ext[off_rq] == 0)
					muxy = 0.0;
				else
					muxy = 1 / (0.25 * (1 / mu_ext[base] + 1 / mu_ext[off_r] + 1 / mu_ext[off_q] + 1 / mu_ext[off_rq]));

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
				txx[base] += DT * (lamda_ext[base] * (uyy + uzz)
					+ lamda2u_ext[base] * uxx);
				tyy[base] += DT * (lamda_ext[base] * (uxx + uzz)
					+ lamda2u_ext[base] * uyy);
				tzz[base] += DT * (lamda_ext[base] * (uxx + uyy)
					+ lamda2u_ext[base] * uzz);
				txy[base] += muxy * DT * (uxy + uyx);
				txz[base] += muxz * DT * (uxz + uzx);
				tyz[base] += muyz * DT * (uyz + uzy);
			}
		}
	}
	for (int ix = sx; ix < NX_ext; ix += step_x) {
		for (int iy = sy; iy < NY_ext; iy += step_y) {
			for (int iz = sz; iz < NZ_ext; iz += step_z) {
				int base = ix + iy * NX_ext + iz * NX_ext * NY_ext;
				midvx[base] = vux[base];
				midvy[base] = vuy[base];
				midvz[base] = vuz[base];
			}
		}
	}
}
// 请在发射 Source<<<…>>> 之前调用
void checkSourceResources(dim3 blockDim, dim3 gridDim, int dynamicShared = 0) {
	// 1) 先清除任何残留的错误状态
	cudaGetLastError();
	// 1. 查询设备极限
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Device limits: regsPerBlock=%d, sharedMemPerBlock=%zu\n",
		prop.regsPerBlock, prop.sharedMemPerBlock);

	// 2. 查询 Source kernel 属性
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, Source);
	printf("Source: numRegs/thread=%d, staticShared=%zu bytes\n",
		attr.numRegs, attr.sharedSizeBytes);

	// 3. 计算本次 launch 需要的资源
	int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
	int regsPerBlock = attr.numRegs * threadsPerBlock;
	size_t sharedPerBlock = attr.sharedSizeBytes + dynamicShared;
	printf("Source launch needs: regsPerBlock=%d, sharedPerBlock=%zu bytes\n",
		regsPerBlock, sharedPerBlock);

	// 4. 对比设备极限
	if (regsPerBlock > prop.regsPerBlock) {
		fprintf(stderr,
			"ERROR: Source regsPerBlock (%d) > device limit (%d)\n",
			regsPerBlock, prop.regsPerBlock);
	}
	if (sharedPerBlock > prop.sharedMemPerBlock) {
		fprintf(stderr,
			"ERROR: Source sharedPerBlock (%zu) > device limit (%zu)\n",
			sharedPerBlock, prop.sharedMemPerBlock);
	}

	// 5. 打印 launch 配置以作核对
	printf("Launching Source with grid=(%d,%d,%d), block=(%d,%d,%d), dynShared=%d\n",
		gridDim.x, gridDim.y, gridDim.z,
		blockDim.x, blockDim.y, blockDim.z,
		dynamicShared);
}

// 请在发射 FD_V<<<…>>> 之前调用
void checkFD_VResources(dim3 blockDim, dim3 gridDim, int dynamicShared = 0) {
	// 1) 先清除任何残留的错误状态
	cudaGetLastError();
	// 1. 查询设备极限
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Device limits: regsPerBlock=%d, sharedMemPerBlock=%zu\n",
		prop.regsPerBlock, prop.sharedMemPerBlock);

	// 2. 查询 FD_V kernel 属性
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, FD_V);
	printf("FD_V:   numRegs/thread=%d, staticShared=%zu bytes\n",
		attr.numRegs, attr.sharedSizeBytes);

	// 3. 计算本次 launch 需要的资源
	int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
	int regsPerBlock = attr.numRegs * threadsPerBlock;
	size_t sharedPerBlock = attr.sharedSizeBytes + dynamicShared;
	printf("FD_V launch needs: regsPerBlock=%d, sharedPerBlock=%zu bytes\n",
		regsPerBlock, sharedPerBlock);

	// 4. 对比设备极限
	if (regsPerBlock > prop.regsPerBlock) {
		fprintf(stderr,
			"ERROR: FD_V regsPerBlock (%d) > device limit (%d)\n",
			regsPerBlock, prop.regsPerBlock);
	}
	if (sharedPerBlock > prop.sharedMemPerBlock) {
		fprintf(stderr,
			"ERROR: FD_V sharedPerBlock (%zu) > device limit (%zu)\n",
			sharedPerBlock, prop.sharedMemPerBlock);
	}

	// 5. 打印 launch 配置
	printf("Launching FD_V with grid=(%d,%d,%d), block=(%d,%d,%d), dynShared=%d\n",
		gridDim.x, gridDim.y, gridDim.z,
		blockDim.x, blockDim.y, blockDim.z,
		dynamicShared);
}

// 请在发射 FD_T<<<…>>> 之前调用
void checkFD_TResources(dim3 blockDim, dim3 gridDim, int dynamicShared = 0) {
	// 清除残留错误
	cudaGetLastError();

	// 查询设备极限
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	// 查询 FD_T 属性
	cudaFuncAttributes attr;
	cudaError_t ferr = cudaFuncGetAttributes(&attr, FD_T);
	if (ferr != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaFuncGetAttributes(FD_T) failed: %s\n",
			cudaGetErrorString(ferr));
		return;
	}

	// 正常打印
	printf("FD_T: numRegs/thread=%d, staticShared=%zu bytes\n",
		attr.numRegs, attr.sharedSizeBytes);

	// 计算需求
	int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
	int regsPerBlock = attr.numRegs * threadsPerBlock;
	size_t sharedPerBlock = attr.sharedSizeBytes + dynamicShared;

	// 对比
	if (regsPerBlock > prop.regsPerBlock) {
		fprintf(stderr,
			"ERROR: FD_T regsPerBlock (%d) > device limit (%d)\n",
			regsPerBlock, prop.regsPerBlock);
	}
	if (sharedPerBlock > prop.sharedMemPerBlock) {
		fprintf(stderr,
			"ERROR: FD_T sharedPerBlock (%zu) > device limit (%zu)\n",
			sharedPerBlock, prop.sharedMemPerBlock);
	}

	printf("Launching FD_T with grid=(%d,%d,%d), block=(%d,%d,%d)\n",
		gridDim.x, gridDim.y, gridDim.z,
		blockDim.x, blockDim.y, blockDim.z);
}


//---------------------------------------------------------------------------------------------------------------
// 主函数
int main()
{
	int dev = 0;
	cudaDeviceProp prop;
	cudaError_t err = cudaGetDeviceProperties(&prop, dev);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
		return 1;
	}

	printf("Device %d: %s\n", dev, prop.name);
	printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("  Max grid dimensions:\n");
	printf("    x: %d\n", prop.maxGridSize[0]);
	printf("    y: %d\n", prop.maxGridSize[1]);
	printf("    z: %d\n", prop.maxGridSize[2]);

	// 给定参数
	float best_dt;
	int ix, iy, iz, it;
	int NX_ext = NX + 2 * NP;
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	int sx = NX_ext / 2;	// 震源坐标点号
	int sy = NY_ext / 2;
	int sz = NP + 5;
	int sn = sx + sy * NX_ext + sz * NX_ext * NY_ext;
	int NT = 4500; // 时间层数wo
	int MM = 1;
	float H = 1.0 * pow(10.0, -5);//空间步长
	float RC = 1.0 * pow(10.0, -6);
	float DT = 5.0 * pow(10.0, -10);	    //时间步长
	int averageType = 7;    //平均介质的参数
	float DP = NP * H;
	float DT_H = DT / H;
	float F0 = 10 * pow(10.0, 6);		//震源主频,主频太高会造成频散
	float T0 = 1.2 / F0;
	float Vpmax = 5560.6;    //模型最大纵波速度,用于稳定性计算
	float Vsmin = 1639.0;
	dim3 Blocksize = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 Gridsize = dim3(100, 100, 100);
	size_t mem_size = NZ_ext * NY_ext * NX_ext * sizeof(float); // 内存大小
	//---------------------------------------------------------------------------
	//---------------------------------------------------------------------------
	//参数开辟内存
	float*** elastic_pre;
	elastic_pre = space3d(NZZ, NY, NX);
	//定义围压
	//float pressure[15] = { 0.1, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400 };
	float pressure[1] = { 0.1 };
	printf("input begin");
	// 打开文件
	FILE* file11;
	file11 = fopen("E:/500modelCalElastic/500modelCalElastic/model_2_500x500x500.dat", "r");
	if (file11 == NULL) {
		fprintf(stderr, "无法打开文件\n");
		return 1;
	}

	// 读取文件内容并存储到数组中
	int INPUTnz, INPUTny, INPUTnx;
	float INPUTvalue;
	while (fscanf(file11, "%d %d %d %f", &INPUTnz, &INPUTny, &INPUTnx, &INPUTvalue) != EOF) {
		if (INPUTnz >= 0 && INPUTnz < NZZ && INPUTny >= 0 && INPUTny < NY && INPUTnx >= 0 && INPUTnx < NX) {
			elastic_pre[INPUTnz][INPUTny][INPUTnx] = INPUTvalue;
		}
		else {
			fprintf(stderr, "索引超出范围: nz=%d, ny=%d, nx=%d\n", INPUTnz, INPUTny, INPUTnx);
		}
	}

	// 关闭文件
	fclose(file11);
	// 打印数组的某些值以验证
	printf("elastic_pre[50][50][50] = %f\n", elastic_pre[50][50][50]);
	printf("elastic_pre[20][50][50] = %f\n", elastic_pre[20][50][50]);
	//input结束
	printf("input finish\n");

	//扩展input
	float*** elastic_pre_all;
	elastic_pre_all = space3d(NZ_ext, NY_ext, NX_ext);
	int Zdif = (NZ_ext - NZZ) * 0.5;
	int Ydif = (NY_ext - NY) * 0.5;
	int Xdif = (NX_ext - NX) * 0.5;
	for (int k = 0; k < NZ_ext; k++)
		for (int j = 0; j < NY_ext; j++)
			for (int i = 0; i < NX_ext; i++)
			{
				if (k >= Zdif && k < (NZZ + Zdif) && j >= Ydif && j < (NY + Ydif) && i >= Xdif && i < (NX + Xdif))
				{
					elastic_pre_all[k][j][i] = elastic_pre[k - Zdif][j - Ydif][i - Xdif];
				}
				else
				{
					elastic_pre_all[k][j][i] = averageType;
				}
			}
	char elastic_pre_allname[] = "elastic_pre_all.dat";
	wfile3d(elastic_pre_allname, elastic_pre_all, NZ_ext, NY_ext, NX_ext);
	//---------------------------------------------------------------------------
	//多次围压循环
	for (int prenumber = 0; prenumber < 1; prenumber++)
	{
		float pressure0 = pressure[prenumber];
		printf("presuure = %f\n", pressure0);
		int namepressure = pressure[prenumber];

		float** sis_x;  //地震记录z分量
		float** sis_y;  //地震记录z分量
		float** sis_z;  //地震记录z分量
		sis_x = space2d(NZ_ext, NT);
		sis_y = space2d(NZ_ext, NT);
		sis_z = space2d(NZ_ext, NT);
		float** sis_vx;
		float** sis_vy;
		float** sis_vz;
		sis_vx = space2d(NZ_ext, NT);
		sis_vy = space2d(NZ_ext, NT);
		sis_vz = space2d(NZ_ext, NT);

		// 参数开辟内存
		float*** vs_ext;
		float*** vp_ext;
		float*** vf_ext;
		float*** rhof_ext;
		vs_ext = space3d(NZ_ext, NY_ext, NX_ext);
		vp_ext = space3d(NZ_ext, NY_ext, NX_ext);
		rhof_ext = space3d(NZ_ext, NY_ext, NX_ext);
		vf_ext = space3d(NZ_ext, NY_ext, NX_ext);
		// 定义密度，粘度系数分量
		// 在主机端CPU定义参数，分配内存
		float* rho_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* mu_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_lamda_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_lamda2u_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_vux = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_vuy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_vuz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_txx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_tyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_txy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_tyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_tzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_txz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));

		float* h_midtxx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_midtyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_midtzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_midtxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_midtxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_midtyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_midvx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_midvy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_midvz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		//----------------------------------------------------------------------------------
		// 创建模型赋值
		//创建模型赋值
		create_model(vp_ext, vs_ext, rho_ext, vf_ext, rhof_ext, h_lamda_ext, h_lamda2u_ext, mu_ext, NZ_ext, NY_ext, NX_ext, pressure0, elastic_pre_all);
		char rhof_extname[] = "rho_ext";
		char rhof_extName[50];
		sprintf(rhof_extName, "%s.dat", rhof_extname);
		wfile1d(rhof_extName, rho_ext, NZ_ext, NY_ext, NX_ext);
		//----------------------------------------------------------------------------------
		// 应力和速度分量内存开辟
		float*** vux;
		float*** vuy;
		float*** vuz;
		float*** txx;
		float*** txx1000;
		vux = space3d(NZ_ext, NY_ext, NX_ext);
		vuy = space3d(NZ_ext, NY_ext, NX_ext);
		vuz = space3d(NZ_ext, NY_ext, NX_ext);
		txx = space3d(NZ_ext, NY_ext, NX_ext);
		txx1000 = space3d(NZ_ext, NY_ext, NX_ext);
		//-----------------------------------------------并行计算-------------------------------------------------------------------------
		//------------------------------------------------------------------------------------------------------------
		// 在设备端GPU定义参数，分配内存
		float* d_lamda_ext, * d_lamda2u_ext, * d_rho_ext, * d_mu_ext; // 初始模型密度
		float* d_vux, * d_vuy, * d_vuz;
		float* d_txx, * d_tyy, * d_tzz, * d_txy, * d_tyz, * d_txz;
		float* d_pmlxSxx, * d_pmlySxy, * d_pmlzSxz, * d_pmlxSxy, * d_pmlySyy, * d_pmlzSyz, * d_pmlxSxz, * d_pmlySyz, * d_pmlzSzz;
		float* d_pmlxVux, * d_pmlyVuy, * d_pmlzVuz, * d_pmlxVuy, * d_pmlyVux, * d_pmlxVuz, * d_pmlzVux, * d_pmlyVuz, * d_pmlzVuy;
		size_t mem_size_pmlx = sizeof(float) * 2 * NP * NY_ext * NZ_ext;
		size_t mem_size_pmly = sizeof(float) * 2 * NP * NX_ext * NZ_ext;
		size_t mem_size_pmlz = sizeof(float) * 2 * NP * NX_ext * NY_ext;
		CUDA_CHECK(cudaMalloc(&d_lamda_ext, mem_size));
		CUDA_CHECK(cudaMalloc(&d_lamda2u_ext, mem_size));
		CUDA_CHECK(cudaMalloc(&d_rho_ext, mem_size));
		CUDA_CHECK(cudaMalloc(&d_mu_ext, mem_size));
		CUDA_CHECK(cudaMalloc(&d_vux, mem_size));
		CUDA_CHECK(cudaMalloc(&d_vuy, mem_size));
		CUDA_CHECK(cudaMalloc(&d_vuz, mem_size));
		CUDA_CHECK(cudaMalloc(&d_txx, mem_size));
		CUDA_CHECK(cudaMalloc(&d_tyy, mem_size));
		CUDA_CHECK(cudaMalloc(&d_tzz, mem_size));
		CUDA_CHECK(cudaMalloc(&d_txz, mem_size));
		CUDA_CHECK(cudaMalloc(&d_txy, mem_size));
		CUDA_CHECK(cudaMalloc(&d_tyz, mem_size));
		CUDA_CHECK(cudaMalloc(&d_pmlxSxx, mem_size_pmlx));
		CUDA_CHECK(cudaMalloc(&d_pmlxSxy, mem_size_pmlx));
		CUDA_CHECK(cudaMalloc(&d_pmlxSxz, mem_size_pmlx));
		CUDA_CHECK(cudaMalloc(&d_pmlxVux, mem_size_pmlx));
		CUDA_CHECK(cudaMalloc(&d_pmlxVuy, mem_size_pmly));
		CUDA_CHECK(cudaMalloc(&d_pmlxVuz, mem_size_pmlz));
		CUDA_CHECK(cudaMalloc(&d_pmlySxy, mem_size_pmly));
		CUDA_CHECK(cudaMalloc(&d_pmlySyy, mem_size_pmly));
		CUDA_CHECK(cudaMalloc(&d_pmlySyz, mem_size_pmly));
		CUDA_CHECK(cudaMalloc(&d_pmlyVuy, mem_size_pmly));
		CUDA_CHECK(cudaMalloc(&d_pmlyVuz, mem_size_pmlz));
		CUDA_CHECK(cudaMalloc(&d_pmlyVux, mem_size_pmlx));
		CUDA_CHECK(cudaMalloc(&d_pmlzSxz, mem_size_pmlz));
		CUDA_CHECK(cudaMalloc(&d_pmlzSyz, mem_size_pmlz));
		CUDA_CHECK(cudaMalloc(&d_pmlzSzz, mem_size_pmlz));
		CUDA_CHECK(cudaMalloc(&d_pmlzVuz, mem_size_pmlz));
		CUDA_CHECK(cudaMalloc(&d_pmlzVux, mem_size_pmlx));
		CUDA_CHECK(cudaMalloc(&d_pmlzVuy, mem_size_pmly));
		//---------------------------------------------------------------------------------------------
		best_dt = 6.0 * H / (7.0 * sqrt(2.0) * Vpmax);
		if (DT >= best_dt)
			printf("时间步数大 %f\n", best_dt);
		if (Vsmin / (F0 * H) < 15)
			printf("网格过大\n");
		clock_t start, finish;
		start = clock();
		//--------------------------------------------------------------------------------------------
		// 进行参数传递，将主机端CPU参数传递到设备端GPU
		cudaMemcpy(d_lamda_ext, h_lamda_ext, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_lamda2u_ext, h_lamda2u_ext, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rho_ext, rho_ext, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_mu_ext, mu_ext, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vux, h_vux, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vuy, h_vuy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vuz, h_vuz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_txx, h_txx, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tyy, h_tyy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tzz, h_tzz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_txy, h_txy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_txz, h_txz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tyz, h_tyz, mem_size, cudaMemcpyHostToDevice);
		free(h_lamda_ext);
		free(mu_ext);
		free(rho_ext);
		free(h_lamda2u_ext);
		free(h_tyy);
		free(h_txz);
		free(h_txy);
		free(h_tyz);
		int dynShared = 0; // 如果没有动态共享内存，就用 0
		float tt;
		//---------------------------------------------------------------------------------------------------------------------
		// 在时间上进行迭代
		for (it = 0; it < NT; it++)
		{


			float tt = it * DT;
			if (it % 100 == 0)
			{
				printf("it=%d\n", it);
			}
			if (it % MM == 0)
			{
				// 加震源
				float I_sou;
				I_sou = -(1 - 2 * PI * PI * F0 * F0 * (tt - T0) * (tt - T0)) * exp(-PI * PI * F0 * F0 * (tt - T0) * (tt - T0));
				//checkSourceResources(Blocksize, Gridsize, dynShared);
				Source << <Gridsize, Blocksize >> > (d_txx, d_tyy, d_tzz, I_sou, sn, NX_ext, NY_ext, NZ_ext);
				cudaDeviceSynchronize();
			}

			//----------------------------------------------------------------------------------------------------
			// 计算速度
			//checkFD_VResources(Blocksize, Gridsize, dynShared);
			float* d_midtxx, * d_midtyy, * d_midtzz, * d_midtyz, * d_midtxy, * d_midtxz;
			CUDA_CHECK(cudaMalloc(&d_midtxx, mem_size));
			CUDA_CHECK(cudaMalloc(&d_midtyy, mem_size));
			CUDA_CHECK(cudaMalloc(&d_midtzz, mem_size));
			CUDA_CHECK(cudaMalloc(&d_midtyz, mem_size));
			CUDA_CHECK(cudaMalloc(&d_midtxy, mem_size));
			CUDA_CHECK(cudaMalloc(&d_midtxz, mem_size));
			cudaMemcpy(d_midtxx, h_midtxx, mem_size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_midtyy, h_midtyy, mem_size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_midtzz, h_midtzz, mem_size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_midtxy, h_midtxy, mem_size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_midtxz, h_midtxz, mem_size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_midtyz, h_midtyz, mem_size, cudaMemcpyHostToDevice);
			FD_V << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_rho_ext, d_txx, d_tyy, d_tzz, d_txz, d_txy, d_tyz, d_pmlxSxx, d_pmlySxy, d_pmlzSxz, d_pmlxSxy, d_pmlySyy, d_pmlzSyz, d_pmlxSxz, d_pmlySyz, d_pmlzSzz, d_midtxx, d_midtyy, d_midtzz, d_midtxy, d_midtxz, d_midtyz, DT, H);
			cudaDeviceSynchronize();
			cudaMemcpy(h_midtxx, d_midtxx, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_midtyy, d_midtyy, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_midtzz, d_midtzz, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_midtxy, d_midtxy, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_midtxz, d_midtxz, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_midtyz, d_midtyz, mem_size, cudaMemcpyDeviceToHost);
			cudaFree(d_midtxx);
			cudaFree(d_midtyy);
			cudaFree(d_midtzz);
			cudaFree(d_midtxy);
			cudaFree(d_midtxz);
			cudaFree(d_midtyz);
			//----------------------------------------------------------------------------------------------------------
			// 计算应力
			//checkFD_TResources(Blocksize, Gridsize, dynShared);
			float* d_midvx, * d_midvy, * d_midvz;
			CUDA_CHECK(cudaMalloc(&d_midvx, mem_size));
			CUDA_CHECK(cudaMalloc(&d_midvy, mem_size));
			CUDA_CHECK(cudaMalloc(&d_midvz, mem_size));
			cudaMemcpy(d_midvx, h_midvx, mem_size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_midvy, h_midvy, mem_size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_midvz, h_midvz, mem_size, cudaMemcpyHostToDevice);
			FD_T << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_mu_ext, d_txx, d_tzz, d_tyy, d_txz, d_txy, d_tyz, d_lamda2u_ext, d_lamda_ext, d_pmlxVux, d_pmlyVuy, d_pmlzVuz, d_pmlxVuy, d_pmlyVux, d_pmlyVuz, d_pmlzVuy, d_pmlzVux, d_pmlxVuz, d_midvx, d_midvy, d_midvz, DT, H);
			cudaDeviceSynchronize();
			cudaMemcpy(h_midvx, d_midvx, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_midvy, d_midvy, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_midvz, d_midvz, mem_size, cudaMemcpyDeviceToHost);
			cudaFree(d_midvx);
			cudaFree(d_midvy);
			cudaFree(d_midvz);
			//----------------------------------------------------------------------------------------------------------
			// 将速度，应力从设备端拷贝到主机
			cudaMemcpy(h_txx, d_txx, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_tzz, d_tzz, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_vux, d_vux, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_vuy, d_vuy, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_vuz, d_vuz, mem_size, cudaMemcpyDeviceToHost);
			// 记录地震记录
			for (iz = 0; iz < NZ_ext; iz++)
			{
				sis_x[iz][it] = h_txx[iz * NX_ext * NY_ext + sx + sy * NX_ext];
				//sis_z[iz][it] = h_tzz[iz * NX_ext * NY_ext + sx + sy * NX_ext];
				sis_vx[iz][it] = h_vux[iz * NX_ext * NY_ext + sx + sy * NX_ext];
				sis_vy[iz][it] = h_vuy[iz * NX_ext * NY_ext + sx + sy * NX_ext];
				sis_vz[iz][it] = h_vuz[iz * NX_ext * NY_ext + sx + sy * NX_ext];
			}

			if (it % 500 == 0)
			{
				for (int k = 0; k < NZ_ext; k++)
				{
					for (int j = 0; j < NY_ext; j++)
					{
						for (int i = 0; i < NX_ext; i++)
						{
							txx1000[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
						}
					}
				}
				int name = it;
				char txxname[] = "txx";
				char txxName[50];
				sprintf(txxName, "%s%d %d.dat", txxname, name, namepressure);
				wfile3d(txxName, txx1000, NZ_ext, NY_ext, NX_ext);
			}
		}
		finish = clock();
		printf("%f seconds\n", (float)(finish - start) / CLOCKS_PER_SEC);
		//--------------------------------------------------------------------------------------------------------------
		// 输出波场快照
		printf("NZ_ext = %d NY_ext = %d NX_ext = %d\n", NZ_ext, NY_ext, NX_ext);
		// 输出地震记录
		char sisxname[] = "sisx";
		char sisxxname[50];
		sprintf(sisxxname, "%s %d.dat", sisxname, namepressure);
		wfile(sisxxname, sis_x, NZ_ext, NT);

		char sisvxname[] = "sisvx";
		char sisvxxname[50];
		sprintf(sisvxxname, "%s %d.dat", sisvxname, namepressure);
		wfile(sisvxxname, sis_vx, NZ_ext, NT);

		char sisvyname[] = "sisvy";
		char sisvyyname[50];
		sprintf(sisvyyname, "%s %d.dat", sisvyname, namepressure);
		wfile(sisvyyname, sis_vy, NZ_ext, NT);

		char sisvzname[] = "sisvz";
		char sisvzzname[50];
		sprintf(sisvzzname, "%s %d.dat", sisvzname, namepressure);
		wfile(sisvzzname, sis_vz, NZ_ext, NT);

		//---------------------------------------------------------------------------------------------------------------
		// 主机端释放内存
		free_space3d(txx1000, NZ_ext, NY_ext);
		free_space3d(txx, NZ_ext, NY_ext);
		free_space3d(vux, NZ_ext, NY_ext);
		free_space3d(vuy, NZ_ext, NY_ext);
		free_space3d(vuz, NZ_ext, NY_ext);
		free_space2d(sis_x, NZ_ext);
		free_space2d(sis_y, NZ_ext);
		free_space2d(sis_z, NZ_ext);
		free_space2d(sis_vx, NZ_ext);
		free_space2d(sis_vy, NZ_ext);
		free_space2d(sis_vz, NZ_ext);
		free(h_txx);
		free(h_tzz);
		free(h_vux);
		free(h_vuy);
		free(h_vuz);
		free(h_midtxx);
		free(h_midtyy);
		free(h_midtzz);
		free(h_midtxy);
		free(h_midtxz);
		free(h_midtyz);
		free(h_midvx);
		free(h_midvy);
		free(h_midvz);
		// 设备端释放内存
		cudaFree(d_lamda2u_ext);
		cudaFree(d_lamda_ext);
		cudaFree(d_mu_ext);
		cudaFree(d_pmlxSxx);
		cudaFree(d_pmlxSxy);
		cudaFree(d_pmlxSxz);
		cudaFree(d_pmlxVux);
		cudaFree(d_pmlxVuy);
		cudaFree(d_pmlxVuz);
		cudaFree(d_pmlySxy);
		cudaFree(d_pmlySyy);
		cudaFree(d_pmlySyz);
		cudaFree(d_pmlyVux);
		cudaFree(d_pmlyVuy);
		cudaFree(d_pmlyVuz);
		cudaFree(d_pmlzSxz);
		cudaFree(d_pmlzSyz);
		cudaFree(d_pmlzSzz);
		cudaFree(d_pmlzVux);
		cudaFree(d_pmlzVuy);
		cudaFree(d_pmlzVuz);
		cudaFree(d_txx);
		cudaFree(d_tyy);
		cudaFree(d_tzz);
		cudaFree(d_txy);
		cudaFree(d_txz);
		cudaFree(d_tyz);
		cudaFree(d_vux);
		cudaFree(d_vuy);
		cudaFree(d_vuz);
	}
	free_space3d(elastic_pre_all, NZ, NY);
	free_space3d(elastic_pre, NZZ, NY);
	printf("\Press any key to exit program...");
	return 0;
}

// 申请二维动态数组
float** space2d(int nr, int nc)
{
	float** a;
	int i;
	a = (float**)calloc(nr, sizeof(float*));
	for (i = 0; i < nr; i++)
		a[i] = (float*)calloc(nc, sizeof(float));

	return a;
}

// 释放二维动态数组
void free_space2d(float** a, int nr)
{
	int i;
	for (i = 0; i < nr; i++)
		free(a[i]);
	free(a);
}

// 将二进制数据写入文件
void wfile(char filename[], float** data, int nr, int nc)
{
	int i, j;
	FILE* fp = fopen(filename, "wt");
	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < nc; j++)
		{
			fprintf(fp, "%e ", data[i][j]);
			if ((j + 1) % nc == 0)
				fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

// 申请三维动态数组
float*** space3d(int nr, int ny, int nc)
{
	float*** a;
	int i, j, k;
	a = (float***)malloc(sizeof(float**) * nr);
	for (i = 0; i < nr; i++)
	{
		a[i] = (float**)malloc(sizeof(float*) * ny);
	}
	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < ny; j++)
		{
			a[i][j] = (float*)malloc(sizeof(float) * nc); // sizeof(float) nc改为sizeof(float)*nc
		}
	}
	for (i = 0; i < nr; i++)
		for (j = 0; j < ny; j++)
			for (k = 0; k < nc; k++)
			{
				a[i][j][k] = 0.0f;
			}
	return a;
}

// 释放三维动态数组
void free_space3d(float*** a, int nr, int ny)
{
	int i, j;
	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < ny; j++)
		{
			free(a[i][j]);
		}
	}
	for (i = 0; i < nr; i++)
	{
		free(a[i]);
	}
	free(a);
}

// 将二进制数据写入文件—三维
void wfile3d(char filename[], float*** data, int nr, int ny, int nc)
{
	int i, j, k;
	FILE* fp = fopen(filename, "wt");
	/*
	for (int i = 0; i<nr; i++)
	{
	fwrite(data[i], sizeof(float), nc, fp);
	}
	*/
	{
		for (i = 0; i < nr; i++)
		{
			for (k = 0; k < nc; k++)
			{
				j = ny / 2; // y轴切片
				fprintf(fp, "%e ", data[i][j][k]);
				if ((k + 1) % nc == 0)
					fprintf(fp, "\n");
			}
			fprintf(fp, "\n");
		}
	}
	//          fwrite(&data[i][j],1,sizeof(float),fp);
	fclose(fp);
}

// 将二进制数据写入文件—三维
void wfile1d(char filename[], float* data, int nr, int ny, int nc)
{
	int i, j, k;
	FILE* fp = fopen(filename, "wt");
	/*
	for (int i = 0; i<nr; i++)
	{
	fwrite(data[i], sizeof(float), nc, fp);
	}
	*/
	{
		for (i = 0; i < nr; i++)
		{
			for (k = 0; k < nc; k++)
			{
				j = ny / 2; // y轴切片
				fprintf(fp, "%e ", data[i * nc * ny + j * nc + k]);
				if ((k + 1) % nc == 0)
					fprintf(fp, "\n");
			}
			fprintf(fp, "\n");
		}
	}
	//          fwrite(&data[i][j],1,sizeof(float),fp);
	fclose(fp);
}

void create_model(float*** vp, float*** vs, float* rho, float*** vf, float*** rhof, float* lamda, float* lamda2u, float* mu, int nr, int ny, int nc, float pressure, float*** elastic_pre)
{
	int ix, iy, iz;
	//定义温度
	float Temp = 20;
	//定义各个矿物的速度参数
	float Vs_mineral[9];
	float Vp_mineral[9];
	float rho_mineral[9];

	//// 孔隙 油or气
	//Vp_mineral[7] = 629.7f;
	//Vs_mineral[7] = 0.0f;
	//rho_mineral[7] = 139.8f;
	//// 平均部分
	//Vp_mineral[8] = 3000.0f;
	//Vs_mineral[8] = 1700.0f;
	//rho_mineral[8] = 2292.7f;
	// 粘土
	Vp_mineral[2] = 3431.9f;
	Vs_mineral[2] = 1639.0f;
	rho_mineral[2] = 2550.0f;
	// 钾长石
	Vp_mineral[3] = 1000 * (4.2832 + 1.855 * 0.0001 * pressure - 0.673 * exp(-2.214 * 0.01 * pressure));
	Vs_mineral[3] = 0.744 * Vp_mineral[3] - 0.620;
	rho_mineral[3] = 2540.0f;
	// 斜长石
	Vp_mineral[4] = 1000 * (4.1886 + 1.95 * 0.0001 * pressure - 0.604 * exp(-3.364 * 0.01 * pressure));
	Vs_mineral[4] = 0.744 * Vp_mineral[4] - 0.620;
	rho_mineral[4] = 2650.0f;
	// 石英
	Vp_mineral[5] = 1000 * (5.5606 + 2.5 * 0.0001 * pressure - 0.479 * exp(-3.007 * 0.01 * pressure));
	Vs_mineral[5] = 0.744 * Vp_mineral[5] - 0.620;
	rho_mineral[5] = 2630.0f;
	// 碳酸盐
	Vp_mineral[6] = 1000 * (5.5344 + 2.417 * 0.0001 * pressure - 0.723 * exp(-2.453 * 0.01 * pressure));
	Vs_mineral[6] = 0.744 * Vp_mineral[6] - 0.620;
	rho_mineral[6] = 2710.0f;
	// 孔隙 水
	Vp_mineral[1] = 1500.0f;
	Vs_mineral[1] = 0.0f;
	rho_mineral[1] = 1000.0f;
	// 平均部分
	Vp_mineral[7] = 3000.0f;
	Vs_mineral[7] = 1700.0f;
	rho_mineral[7] = 2292.7f;

	//对rho，vs，vp赋值,计算弹性系数
	for (iz = 0; iz < nr; iz++)
	{
		for (iy = 0; iy < ny; iy++)
		{
			for (ix = 0; ix < nc; ix++)
			{

				vs[iz][iy][ix] = Vs_mineral[(int)elastic_pre[iz][iy][ix]];
				vp[iz][iy][ix] = Vp_mineral[(int)elastic_pre[iz][iy][ix]];
				rho[iz * nc * ny + iy * nc + ix] = rho_mineral[(int)elastic_pre[iz][iy][ix]];
				mu[iz * nc * ny + iy * nc + ix] = rho[iz * nc * ny + iy * nc + ix] * vs[iz][iy][ix] * vs[iz][iy][ix];
				lamda2u[iz * nc * ny + iy * nc + ix] = rho[iz * nc * ny + iy * nc + ix] * vp[iz][iy][ix] * vp[iz][iy][ix];
				lamda[iz * nc * ny + iy * nc + ix] = lamda2u[iz * nc * ny + iy * nc + ix] - 2.0 * mu[iz * nc * ny + iy * nc + ix];
			}
		}
	}

}
