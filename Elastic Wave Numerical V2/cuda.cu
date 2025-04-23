
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#pragma argsused
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <conio.h>
//#include <cutil.h>
#define PI 3.1415926535
#define e 2.718281828
//#define NZ 400	
//#define NY 400   //此处是y轴的网格点数量
//#define NX 400
//#define NP 16
//#define BLOCK_SIZE_X 16
//#define BLOCK_SIZE_Y 8
//#define BLOCK_SIZE_Z 1
//#define NZ 300	
//#define NY 120   //此处是y轴的网格点数量
//#define NX 120
//#define NP 120
#define NZ 240	
#define NZZ 200	
#define NY 200   //此处是y轴的网格点数量
#define NX 200
#define NP 30
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 1

void exportArrayToFile(float*** a, int xSize, int ySize, int zSize, const char* filename);
float** space2d(int nr, int nc);
float*** space3d(int nr, int ny, int nc);
void free_space2d(float** a, int nr);
void free_space3d(float*** b, int nr, int ny);
void wfile(char filename[], float** data, int nr, int nc);
void wfile3d(char filename[], float*** data, int nr, int ny, int nc);
void create_model(float*** vp, float*** vs, float*** rho, float*** vf, float*** rhof, float*** lamda, float*** lamda2u, float*** mu, int nr, int ny, int nc, float pressure, float*** elastic_pre);
void fillCube(float*** space, int x, int y, int z, int length, float value);

int divUp(int a, int b) { return (a + b - 1) / b; }
//-------------------------------------------------------------------------------------------------------------------------------
//计算震源
__global__ void Source(float* txx, float* tyy, float* tzz, float I_sou, int sn, int NX_ext, int NY_ext, int NZ_ext)
{
	//加震源
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	if (offset == sn)
	{
		txx[offset] = txx[offset] + I_sou;
		tyy[offset] = tyy[offset] + I_sou;
		tzz[offset] = tzz[offset] + I_sou;

	}
}
//------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------
//计算速度
__global__ void FD_V(float* vux, float* vuy, float* vuz, float* rho_tempx, float* rho_tempy, float* rho_tempz,
	float* txx, float* tyy, float* tzz, float* txz, float* txy, float* tyz,
	float* pmlxSxx, float* pmlySxy, float* pmlzSxz, float* pmlxSxy, float* pmlySyy, float* pmlzSyz, float* pmlxSxz, float* pmlySyz, float* pmlzSzz,
	float* e_dxi2, float* dxi2, float* e_dyj, float* dyj, float* e_dzk, float* dzk, float* e_dxi, float* dxi, float* e_dyj2, float* dyj2, float* dzk2, float* e_dzk2,
	float* SXxx, float* SXxy, float* SXxz, float* SYxy, float* SYyy, float* SYyz, float* SZxz, float* SZyz, float* SZzz, float DT, float H,
	// 新增参数 —— 体力源
	float  I_sou,     // 当前时间步的源振幅
	int    sn,        // 源点线性索引
	int     waveType
)
{
	float x1, x2, x3;
	float y1, y2, y3;
	float z1, z2, z3;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int NX_ext = NX + 2 * NP;    //加上pml之后x方向的总网格数
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_b = ix + iy * NX_ext + (iz - 1) * NX_ext * NY_ext;//上
	int offset_r = ix + 1 + iy * NX_ext + iz * NX_ext * NY_ext;//右
	int offset_h = ix + (iy - 1) * NX_ext + iz * NX_ext * NY_ext;//后
	int offset_l = ix - 1 + iy * NX_ext + iz * NX_ext * NY_ext;//左
	int offset_q = ix + (iy + 1) * NX_ext + iz * NX_ext * NY_ext;//前
	int offset_u = ix + iy * NX_ext + (1 + iz) * NX_ext * NY_ext;//下
	if (ix > 0 && iy > 0 && iz > 0 && ix < (NX_ext - 1) && iy < (NY_ext - 1) && iz < (NZ_ext - 1))
	{
		x1 = (txx[offset_r] - txx[offset]) / H;
		x2 = (txy[offset] - txy[offset_h]) / H;
		x3 = (txz[offset] - txz[offset_b]) / H;

		y1 = (tyy[offset_q] - tyy[offset]) / H;
		y2 = (txy[offset] - txy[offset_l]) / H;
		y3 = (tyz[offset] - tyz[offset_b]) / H;

		z1 = (tzz[offset_u] - tzz[offset]) / H;
		z2 = (txz[offset] - txz[offset_l]) / H;
		z3 = (tyz[offset] - tyz[offset_h]) / H;

		pmlxSxx[offset] = pmlxSxx[offset] * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5) * (e_dxi2[offset] * SXxx[offset] + x1);
		pmlySxy[offset] = pmlySxy[offset] * e_dyj[offset] + (-DT * dyj[offset] * 0.5) * (e_dyj[offset] * SXxy[offset] + x2);
		pmlzSxz[offset] = pmlzSxz[offset] * e_dzk[offset] + (-DT * dzk[offset] * 0.5) * (e_dzk[offset] * SXxz[offset] + x3);
		SXxx[offset] = x1; SXxy[offset] = x2; SXxz[offset] = x3;
		x1 = x1 + pmlxSxx[offset];
		x2 = x2 + pmlySxy[offset];
		x3 = x3 + pmlzSxz[offset];

		pmlxSxy[offset] = pmlxSxy[offset] * e_dxi[offset] + (-DT * dxi[offset] * 0.5) * (e_dxi[offset] * SYxy[offset] + y2);
		pmlySyy[offset] = pmlySyy[offset] * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5) * (e_dyj2[offset] * SYyy[offset] + y1);
		pmlzSyz[offset] = pmlzSyz[offset] * e_dzk[offset] + (-DT * dzk[offset] * 0.5) * (e_dzk[offset] * SYyz[offset] + y3);
		SYxy[offset] = y2; SYyy[offset] = y1; SYyz[offset] = y3;
		y2 = y2 + pmlxSxy[offset];
		y1 = y1 + pmlySyy[offset];
		y3 = y3 + pmlzSyz[offset];

		pmlxSxz[offset] = pmlxSxz[offset] * e_dxi[offset] + (-DT * dxi[offset] * 0.5) * (e_dxi[offset] * SZxz[offset] + z2);
		pmlySyz[offset] = pmlySyz[offset] * e_dyj[offset] + (-DT * dyj[offset] * 0.5) * (e_dyj[offset] * SZyz[offset] + z3);
		pmlzSzz[offset] = pmlzSzz[offset] * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5) * (e_dzk2[offset] * SZzz[offset] + z1);
		SZxz[offset] = z2; SZyz[offset] = z3; SZzz[offset] = z1;
		z2 = z2 + pmlxSxz[offset];
		z3 = z3 + pmlySyz[offset];
		z1 = z1 + pmlzSzz[offset];

		vux[offset] = vux[offset] + DT * rho_tempx[offset] * (x1 + x2 + x3);
		vuy[offset] = vuy[offset] + DT * rho_tempy[offset] * (y1 + y2 + y3);
		vuz[offset] = vuz[offset] + DT * rho_tempz[offset] * (z1 + z2 + z3);
		// —— 在速度上直接注入体力源 —— 
		 // 注入体力源
		if (offset == sn) {
			if (waveType == 0) {
				// P 波
				vuz[offset] += I_sou * rho_tempz[offset] * DT;
			}
			else if (waveType == 1) {
				// SV 波
				vux[offset] += I_sou * rho_tempx[offset] * DT;
			}
			else if (waveType == 2) {
				// SH 波
				vuy[offset] += I_sou * rho_tempy[offset] * DT;
			}
		}
	}
}

//-------------------------------------------------------------------------------------------------------------------
//计算应力时的中间参数计算
//计算应力
__global__ void FD_T(float* vux, float* vuy, float* vuz, float* muxy, float* muxz, float* muyz,
	float* txx, float* tzz, float* tyy, float* txz, float* txy, float* tyz, float* lamda2u_ext, float* lamda_ext,
	float* pmlxVux, float* pmlyVuy, float* pmlzVuz, float* pmlxVuy, float* pmlyVux, float* pmlyVuz, float* pmlzVuy, float* pmlzVux, float* pmlxVuz,
	float* dxi, float* e_dxi, float* dyj, float* e_dyj, float* dzk, float* e_dzk, float* dyj2, float* e_dyj2, float* dxi2, float* e_dxi2, float* dzk2, float* e_dzk2,
	float* Vuxx, float* Vuyy, float* Vuzz, float* Vuxy, float* Vuyx, float* Vuxz, float* Vuzx, float* Vuyz, float* Vuzy, float DT, float H, float A3oeCs, float B3oeCs, float C3oeCs, float Pressure, float* mu_ext)
{

	float uxx, uyy, uzz;
	float uxy, uxz, uyx, uyz, uzx, uzy;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int NX_ext = NX + 2 * NP;    //加上pml之后x方向的总网格数
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;

	int offset = ix + iy * NX_ext + iz * NX_ext * NY_ext;
	int offset_b = ix + iy * NX_ext + (iz - 1) * NX_ext * NY_ext;//上
	int offset_u = ix + iy * NX_ext + (1 + iz) * NX_ext * NY_ext;//下
	int offset_r = ix + 1 + iy * NX_ext + iz * NX_ext * NY_ext;//右
	int offset_l = ix - 1 + iy * NX_ext + iz * NX_ext * NY_ext;//左
	int offset_q = ix + (iy + 1) * NX_ext + iz * NX_ext * NY_ext;//前
	int offset_h = ix + (iy - 1) * NX_ext + iz * NX_ext * NY_ext;//后

	if (ix > 0 && iy > 0 && iz > 0 && ix < (NX_ext - 1) && iy < (NY_ext - 1) && iz < (NZ_ext - 1))
	{
		////应变
		//float exx = -Pressure / (3 * lamda_ext[offset] + 2 * mu_ext[offset]);
		//float eyy = -Pressure / (3 * lamda_ext[offset] + 2 * mu_ext[offset]);
		//float ezz = -Pressure / (3 * lamda_ext[offset] + 2 * mu_ext[offset]);
		////3oeCs
		//float A11 = lamda2u_ext[offset] + (3 * lamda2u_ext[offset] + 2 * A3oeCs + 6 * B3oeCs + 2 * C3oeCs) * exx + (2 * B3oeCs + 2 * C3oeCs - lamda2u_ext[offset]) * (eyy + ezz);
		//float A12 = lamda_ext[offset] + (lamda_ext[offset] + 2 * B3oeCs + 2 * C3oeCs) * (exx + eyy) + (-lamda_ext[offset] + 2 * C3oeCs) * ezz;
		//float A13 = lamda_ext[offset] + (lamda_ext[offset] + 2 * B3oeCs + 2 * C3oeCs) * (exx + ezz) + (-lamda_ext[offset] + 2 * C3oeCs) * eyy;
		//float A22 = lamda2u_ext[offset] + (2 * B3oeCs + 2 * C3oeCs - lamda2u_ext[offset]) * (exx + ezz) + (2 * A3oeCs + 6 * B3oeCs + 2 * C3oeCs + 3 * lamda2u_ext[offset]) * eyy;
		//float A23 = lamda_ext[offset] + (2 * C3oeCs - lamda_ext[offset]) * exx + (2 * B3oeCs + 2 * C3oeCs + lamda_ext[offset]) * (eyy + ezz);
		//float A33 = lamda2u_ext[offset] + (2 * B3oeCs + 2 * C3oeCs - lamda2u_ext[offset]) * (exx + eyy) + (2 * A3oeCs + 6 * B3oeCs + 2 * C3oeCs + 3 * lamda2u_ext[offset]) * ezz;
		//float A44 = muyz[offset] + (B3oeCs - muyz[offset]) * exx + (muyz[offset] + 0.5 * A3oeCs + B3oeCs) * (eyy + ezz);
		//float A55 = muxz[offset] + (B3oeCs - muxz[offset]) * eyy + (muxz[offset] + 0.5 * A3oeCs + B3oeCs) * (exx + ezz);
		//float A66 = muxy[offset] + (B3oeCs - muxy[offset]) * ezz + (muxy[offset] + 0.5 * A3oeCs + B3oeCs) * (exx + eyy);


		//if (ix == 50 && iy == 50 && iz ==50)
		//{
		//	//printf("A13 = %f   A23 = %f  A33 = %f\n", A13, A23, A33);
		//	//printf("lamda2u = %f   lamda = %f\n", lamda2u_ext[offset], lamda_ext[offset]);
		//	printf("A44 = %f   A55 = %f  A66 = %f\n", A44, A55, A66);
		//	printf("muyz = %f   muxz = %f  muxy = %f\n", muyz[offset], muxz[offset], muxy[offset]);
		//}

		//差分
		uxx = (vux[offset] - vux[offset_l]) / H;
		uyy = (vuy[offset] - vuy[offset_h]) / H;
		uzz = (vuz[offset] - vuz[offset_b]) / H;

		uxy = (vux[offset_q] - vux[offset]) / H;
		uyx = (vuy[offset_r] - vuy[offset]) / H;

		uxz = (vux[offset_u] - vux[offset]) / H;
		uzx = (vuz[offset_r] - vuz[offset]) / H;

		uyz = (vuy[offset_u] - vuy[offset]) / H;
		uzy = (vuz[offset_q] - vuz[offset]) / H;


		pmlyVuz[offset] = pmlyVuz[offset] * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5) * (e_dzk2[offset] * Vuyz[offset] + uyz);
		pmlzVuy[offset] = pmlzVuy[offset] * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5) * (e_dyj2[offset] * Vuzy[offset] + uzy);
		Vuyz[offset] = uyz; Vuzy[offset] = uzy;
		uyz = uyz + pmlyVuz[offset];
		uzy = uzy + pmlzVuy[offset];


		pmlxVuz[offset] = pmlxVuz[offset] * e_dzk2[offset] + (-DT * dzk2[offset] * 0.5) * (e_dzk2[offset] * Vuxz[offset] + uxz);
		pmlzVux[offset] = pmlzVux[offset] * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5) * (e_dxi2[offset] * Vuzx[offset] + uzx);
		Vuzx[offset] = uzx; Vuxz[offset] = uxz;
		uxz = uxz + pmlxVuz[offset];
		uzx = uzx + pmlzVux[offset];

		pmlxVuy[offset] = pmlxVuy[offset] * e_dyj2[offset] + (-DT * dyj2[offset] * 0.5) * (e_dyj2[offset] * Vuxy[offset] + uxy);
		pmlyVux[offset] = pmlyVux[offset] * e_dxi2[offset] + (-DT * dxi2[offset] * 0.5) * (e_dxi2[offset] * Vuyx[offset] + uyx);
		Vuxy[offset] = uxy; Vuyx[offset] = uyx;
		uxy = uxy + pmlxVuy[offset];
		uyx = uyx + pmlyVux[offset];


		pmlxVux[offset] = pmlxVux[offset] * e_dxi[offset] + (-DT * dxi[offset] * 0.5) * (e_dxi[offset] * Vuxx[offset] + uxx);
		pmlyVuy[offset] = pmlyVuy[offset] * e_dyj[offset] + (-DT * dyj[offset] * 0.5) * (e_dyj[offset] * Vuyy[offset] + uyy);
		pmlzVuz[offset] = pmlzVuz[offset] * e_dzk[offset] + (-DT * dzk[offset] * 0.5) * (e_dzk[offset] * Vuzz[offset] + uzz);
		Vuxx[offset] = uxx; Vuyy[offset] = uyy; Vuzz[offset] = uzz;
		uxx = uxx + pmlxVux[offset];
		uyy = uyy + pmlyVuy[offset];
		uzz = uzz + pmlzVuz[offset];

		//txx[offset] = txx[offset] + DT * (A12 * uyy + A13 * uzz + A11 * uxx);
		//tyy[offset] = tyy[offset] + DT * (A12 * uxx + A23 * uzz + A22 * uyy);
		//tzz[offset] = tzz[offset] + DT * (A13 * uxx + A23 * uyy + A33 * uzz);
		//txy[offset] = txy[offset] + DT * A66 * (uxy + uyx);
		//txz[offset] = txz[offset] + DT * A55 * (uxz + uzx);
		//tyz[offset] = tyz[offset] + DT * A44 * (uyz + uzy);

		txx[offset] = txx[offset] + DT * (lamda_ext[offset] * (uyy + uzz) + lamda2u_ext[offset] * uxx);
		tyy[offset] = tyy[offset] + DT * (lamda_ext[offset] * (uxx + uzz) + lamda2u_ext[offset] * uyy);
		tzz[offset] = tzz[offset] + DT * (lamda_ext[offset] * (uxx + uyy) + lamda2u_ext[offset] * uzz);
		txy[offset] = txy[offset] + muxy[offset] * DT * (uxy + uyx);
		txz[offset] = txz[offset] + muxz[offset] * DT * (uxz + uzx);
		tyz[offset] = tyz[offset] + muyz[offset] * DT * (uyz + uzy);


	}
}


//---------------------------------------------------------------------------------------------------------------
//主函数
int main()
{
	//给定参数
	int NX_ext = NX + 2 * NP;
	int NY_ext = NY + 2 * NP;
	int NZ_ext = NZ + 2 * NP;
	int sx = NX_ext / 2;           //震源坐标点号
	int sy = NY_ext / 2;
	int sz = NP + 5;
	int sn = sx + sy * NX_ext + sz * NX_ext * NY_ext;
	int NT = 3500;	        //时间层数
	int MM = 1;
	int waveType = 0; // 模拟 P 波  1模拟x激发s波 2模拟y激发s波
	float H = 1.0 * pow(10.0, -5);//空间步长
	float RC = 1.0 * pow(10.0, -6);
	float DT = 5.0 * pow(10.0, -10);	    //时间步长
	float DP = NP * H;
	float DT_H = DT / H;
	float F0 = 10 * pow(10.0, 6);		//震源主频,主频太高会造成频散
	float T0 = 1.2 / F0;
	float Vpmax = 5560.6;    //模型最大纵波速度,用于稳定性计算
	float Vsmin = 1639.0;
	float AoeCs = -2378 * pow(10.0, 9);
	float BoeCs = -35 * pow(10.0, 9);
	float CoeCs = 238 * pow(10.0, 9);
	dim3 Blocksize = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 Gridsize = dim3(divUp(NX_ext, BLOCK_SIZE_X), divUp(NY_ext, BLOCK_SIZE_Y), divUp(NZ_ext, BLOCK_SIZE_Z));
	size_t mem_size = NZ_ext * NY_ext * NX_ext * sizeof(float);     //内存大小
	//---------------------------------------------------------------------
	//pml吸收边界设置，开辟内存，赋值
	float*** dxi, *** dxi2;
	float*** dyj, *** dyj2;
	float*** dzk, *** dzk2;
	float*** e_dxi, *** e_dxi2;
	float*** e_dyj, *** e_dyj2;
	float*** e_dzk, *** e_dzk2;
	dxi = space3d(NZ_ext, NY_ext, NX_ext);
	dxi2 = space3d(NZ_ext, NY_ext, NX_ext);
	dyj = space3d(NZ_ext, NY_ext, NX_ext);
	dyj2 = space3d(NZ_ext, NY_ext, NX_ext);
	dzk = space3d(NZ_ext, NY_ext, NX_ext);
	dzk2 = space3d(NZ_ext, NY_ext, NX_ext);
	e_dxi = space3d(NZ_ext, NY_ext, NX_ext);
	e_dxi2 = space3d(NZ_ext, NY_ext, NX_ext);
	e_dyj = space3d(NZ_ext, NY_ext, NX_ext);
	e_dyj2 = space3d(NZ_ext, NY_ext, NX_ext);
	e_dzk = space3d(NZ_ext, NY_ext, NX_ext);
	e_dzk2 = space3d(NZ_ext, NY_ext, NX_ext);
	float tt, x, y, z, xoleft, xoright, yoleft, yoright, zoleft, zoright, d0, best_dt, v0;
	xoleft = DP;
	xoright = (NX_ext - 1) * H - DP;
	yoleft = DP;
	yoright = (NY_ext - 1) * H - DP;
	zoleft = DP;
	zoright = (NZ_ext - 1) * H - DP;
	//用于对vx_x[iz][ix]等的求解，加入吸收边界，使数值衰减
	for (int iz = 0; iz < NZ_ext; iz++)
	{
		for (int iy = 0; iy < NY_ext; iy++)
		{
			for (int ix = 0; ix < NX_ext; ix++)
			{
				x = ix * H; y = iy * H; z = iz * H;
				if ((x >= 0 && x < xoleft) && (y >= 0 && y < NY_ext * H) && (z >= 0 && z < NZ_ext * H))
				{
					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dxi[iz][iy][ix] = d0 * pow(((xoleft - x) / DP), 2);
					dxi2[iz][iy][ix] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
					e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
					e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
				}
				if ((x >= xoright && x < NX_ext * H) && (y >= 0 && y < NY_ext * H) && (z >= 0 && z < NZ_ext * H))
				{
					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dxi[iz][iy][ix] = d0 * pow(((x - xoright) / DP), 2);
					dxi2[iz][iy][ix] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
					e_dxi[iz][iy][ix] = exp(-(dxi[iz][iy][ix]) * DT);
					e_dxi2[iz][iy][ix] = exp(-(dxi2[iz][iy][ix]) * DT);
				}
				if ((x >= 0 && x < NX_ext * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < NZ_ext * H))
				{
					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dyj[iz][iy][ix] = d0 * pow(((yoleft - y) / DP), 2);
					dyj2[iz][iy][ix] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
					e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
					e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
				}
				if ((x >= 0 && x < NX_ext * H) && (y >= yoright && y < NY_ext * H) && (z >= 0 && z < NZ_ext * H))
				{
					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dyj[iz][iy][ix] = d0 * pow(((y - yoright) / DP), 2);
					dyj2[iz][iy][ix] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
					e_dyj[iz][iy][ix] = exp(-(dyj[iz][iy][ix]) * DT);
					e_dyj2[iz][iy][ix] = exp(-(dyj2[iz][iy][ix]) * DT);
				}
				if ((x >= 0 && x < NX_ext * H) && (y >= 0 && y < NY_ext * H) && (z >= 0 && z < zoleft))
				{

					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dzk[iz][iy][ix] = d0 * pow(((zoleft - z) / DP), 2);
					dzk2[iz][iy][ix] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
					e_dzk[iz][iy][ix] = exp(-(dzk[iz][iy][ix]) * DT);
					e_dzk2[iz][iy][ix] = exp(-(dzk2[iz][iy][ix]) * DT);
				}
				if ((x >= 0 && x < NX_ext * H) && (y >= 0 && y < NY_ext * H) && (z >= zoright && z < NZ_ext * H))
				{
					v0 = 4000.0f;
					d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
					dzk[iz][iy][ix] = d0 * pow(((z - zoright) / DP), 2);
					dzk2[iz][iy][ix] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
					e_dzk[iz][iy][ix] = exp((-dzk[iz][iy][ix]) * DT);
					e_dzk2[iz][iy][ix] = exp((-dzk2[iz][iy][ix]) * DT);
				}
			}
		}
	}
	//char dxiname[] = "dxi.dat";
	//wfile3d(dxiname, dxi, NZ_ext, NY_ext, NX_ext);
	//char dxi2name[] = "dxi2.dat";
	//wfile3d(dxi2name, dxi2, NZ_ext, NY_ext, NX_ext);
	//char dyjname[] = "dyj.dat";
	//wfile3d(dyjname, dyj, NZ_ext, NY_ext, NX_ext);
	//char dyj2name[] = "dyj2.dat";
	//wfile3d(dyj2name, dyj2, NZ_ext, NY_ext, NX_ext);
	//char dzkname[] = "dzk.dat";
	//wfile3d(dzkname, dzk, NZ_ext, NY_ext, NX_ext);
	//char dzk2name[] = "dzk2.dat";
	//wfile3d(dzk2name, dzk2, NZ_ext, NY_ext, NX_ext);
	float* h_dxi = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_dyj = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_dzk = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_dxi2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_dyj2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_dzk2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dxi = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dyj = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dzk = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dxi2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dyj2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* h_e_dzk2 = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
	float* d_dxi;
	float* d_dyj;
	float* d_dzk;
	float* d_dxi2;
	float* d_dyj2;
	float* d_dzk2;
	float* d_e_dxi;
	float* d_e_dyj;
	float* d_e_dzk;
	float* d_e_dxi2;
	float* d_e_dyj2;
	float* d_e_dzk2;
	cudaMalloc(&d_dxi, mem_size);
	cudaMalloc(&d_dyj, mem_size);
	cudaMalloc(&d_dzk, mem_size);
	cudaMalloc(&d_dxi2, mem_size);
	cudaMalloc(&d_dyj2, mem_size);
	cudaMalloc(&d_dzk2, mem_size);
	cudaMalloc(&d_e_dxi, mem_size);
	cudaMalloc(&d_e_dyj, mem_size);
	cudaMalloc(&d_e_dzk, mem_size);
	cudaMalloc(&d_e_dxi2, mem_size);
	cudaMalloc(&d_e_dyj2, mem_size);
	cudaMalloc(&d_e_dzk2, mem_size);
	for (int k = 0; k < NZ_ext; k++)
	{
		for (int j = 0; j < NY_ext; j++)
		{
			for (int i = 0; i < NX_ext; i++)
			{
				h_dxi[i + j * NX_ext + k * NX_ext * NY_ext] = dxi[k][j][i];
				h_dyj[i + j * NX_ext + k * NX_ext * NY_ext] = dyj[k][j][i];
				h_dzk[i + j * NX_ext + k * NX_ext * NY_ext] = dzk[k][j][i];
				h_dxi2[i + j * NX_ext + k * NX_ext * NY_ext] = dxi2[k][j][i];
				h_dyj2[i + j * NX_ext + k * NX_ext * NY_ext] = dyj2[k][j][i];
				h_dzk2[i + j * NX_ext + k * NX_ext * NY_ext] = dzk2[k][j][i];
				h_e_dxi[i + j * NX_ext + k * NX_ext * NY_ext] = e_dxi[k][j][i];
				h_e_dyj[i + j * NX_ext + k * NX_ext * NY_ext] = e_dyj[k][j][i];
				h_e_dzk[i + j * NX_ext + k * NX_ext * NY_ext] = e_dzk[k][j][i];
				h_e_dxi2[i + j * NX_ext + k * NX_ext * NY_ext] = e_dxi2[k][j][i];
				h_e_dyj2[i + j * NX_ext + k * NX_ext * NY_ext] = e_dyj2[k][j][i];
				h_e_dzk2[i + j * NX_ext + k * NX_ext * NY_ext] = e_dzk2[k][j][i];
			}
		}
	}
	cudaMemcpy(d_dxi, h_dxi, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dyj, h_dyj, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dzk, h_dzk, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dxi2, h_dxi2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dyj2, h_dyj2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dzk2, h_dzk2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dxi, h_e_dxi, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dyj, h_e_dyj, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dzk, h_e_dzk, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dxi2, h_e_dxi2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dyj2, h_e_dyj2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e_dzk2, h_e_dzk2, mem_size, cudaMemcpyHostToDevice);

	free(h_dxi);
	free(h_dxi2);
	free(h_dyj);
	free(h_dyj2);
	free(h_dzk);
	free(h_dzk2);
	free(h_e_dxi);
	free(h_e_dxi2);
	free(h_e_dyj);
	free(h_e_dyj2);
	free(h_e_dzk);
	free(h_e_dzk2);
	free_space3d(dxi, NZ_ext, NY_ext);
	free_space3d(e_dxi, NZ_ext, NY_ext);
	free_space3d(dxi2, NZ_ext, NY_ext);
	free_space3d(e_dxi2, NZ_ext, NY_ext);
	free_space3d(dyj, NZ_ext, NY_ext);
	free_space3d(e_dyj, NZ_ext, NY_ext);
	free_space3d(dyj2, NZ_ext, NY_ext);
	free_space3d(e_dyj2, NZ_ext, NY_ext);
	free_space3d(dzk, NZ_ext, NY_ext);
	free_space3d(e_dzk, NZ_ext, NY_ext);
	free_space3d(dzk2, NZ_ext, NY_ext);
	free_space3d(e_dzk2, NZ_ext, NY_ext);
	//---------------------------------------------------------------------------
	//参数开辟内存
	float*** elastic_pre;
	elastic_pre = space3d(NZZ, NY, NX);
	//定义围压
	float pressure[15] = { 0.1, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400 };
	printf("input begin");
	// 打开文件
	FILE* file11;
	file11 = fopen("H:/超算数字岩心/200/32.234112_1.700000.txt", "r");
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
	int averageType = 8;//平均介质的参数
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
	for (int prenumber = 0; prenumber < 15; prenumber++)
	{
		float pressure0 = pressure[prenumber];
		printf("presuure = %f\n", pressure0);
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
		float*** vs_ext;		//初始模型横波速度
		float*** vp_ext;		//初始模型纵波速度
		float*** vf_ext;		//孔隙流体速度
		float*** rho_ext;	   //初始模型密度
		float*** rhof_ext;   //孔隙流体密度
		float*** lamda_ext;
		float*** lamda2u_ext;
		float*** mu_ext;
		vs_ext = space3d(NZ_ext, NY_ext, NX_ext);
		vp_ext = space3d(NZ_ext, NY_ext, NX_ext);
		rhof_ext = space3d(NZ_ext, NY_ext, NX_ext);
		rho_ext = space3d(NZ_ext, NY_ext, NX_ext);
		vf_ext = space3d(NZ_ext, NY_ext, NX_ext);
		mu_ext = space3d(NZ_ext, NY_ext, NX_ext);
		lamda_ext = space3d(NZ_ext, NY_ext, NX_ext);
		lamda2u_ext = space3d(NZ_ext, NY_ext, NX_ext);
		//----------------------------------------------------------------------------------
		//创建模型赋值
		create_model(vp_ext, vs_ext, rho_ext, vf_ext, rhof_ext, lamda_ext, lamda2u_ext, mu_ext, NZ_ext, NY_ext, NX_ext, pressure0, elastic_pre_all);
		int namepressure1 = pressure[prenumber];

		/*char lamdaname[] = "lamda";
		char lamda50name[50];
		sprintf(lamda50name, "%s %d.dat", lamdaname, namepressure1);
		wfile3d(lamda50name, lamda_ext, NZ_ext, NY_ext, NX_ext);

		char rhoname[] = "rho";
		char rho50name[50];
		sprintf(rho50name, "%s %d.dat", rhoname, namepressure1);
		wfile3d(rho50name, rho_ext, NZ_ext, NY_ext, NX_ext);

		char muname[] = "mu";
		char mu50name[50];
		sprintf(mu50name, "%s %d.dat", muname, namepressure1);
		wfile3d(mu50name, mu_ext, NZ_ext, NY_ext, NX_ext);*/
		//----------------------------------------------------------------------------------
		//应力和速度分量内存开辟
		float*** vux;
		float*** vuy;
		float*** vuz;
		float*** txx;
		float*** tyy;
		float*** tzz;
		float*** txz;
		float*** txy;
		float*** tyz;
		vux = space3d(NZ_ext, NY_ext, NX_ext);
		vuy = space3d(NZ_ext, NY_ext, NX_ext);
		vuz = space3d(NZ_ext, NY_ext, NX_ext);
		txx = space3d(NZ_ext, NY_ext, NX_ext);
		tyy = space3d(NZ_ext, NY_ext, NX_ext);
		tzz = space3d(NZ_ext, NY_ext, NX_ext);
		txy = space3d(NZ_ext, NY_ext, NX_ext);
		txz = space3d(NZ_ext, NY_ext, NX_ext);
		tyz = space3d(NZ_ext, NY_ext, NX_ext);
		float*** txx50 = space3d(NZ_ext, NY_ext, NX_ext);
		float*** txx100 = space3d(NZ_ext, NY_ext, NX_ext);
		float*** txx150 = space3d(NZ_ext, NY_ext, NX_ext);
		float*** txx200 = space3d(NZ_ext, NY_ext, NX_ext);
		float*** txx250 = space3d(NZ_ext, NY_ext, NX_ext);
		float*** txx300 = space3d(NZ_ext, NY_ext, NX_ext);

		float*** vuz50 = space3d(NZ_ext, NY_ext, NX_ext);
		float*** vuz100 = space3d(NZ_ext, NY_ext, NX_ext);
		float*** vuz150 = space3d(NZ_ext, NY_ext, NX_ext);
		float*** vuz200 = space3d(NZ_ext, NY_ext, NX_ext);
		float*** vuz250 = space3d(NZ_ext, NY_ext, NX_ext);
		float*** vuz300 = space3d(NZ_ext, NY_ext, NX_ext);
		//---------------------------------------------------------------------------------------------
		//------------------------------------------------------------------------------------------------
		//定义密度，粘度系数分量
		float*** rho_tempx;
		float*** rho_tempy;
		float*** rho_tempz;
		float*** muxy, *** muxz, *** muyz;
		muxy = space3d(NZ_ext, NY_ext, NX_ext);
		muxz = space3d(NZ_ext, NY_ext, NX_ext);
		muyz = space3d(NZ_ext, NY_ext, NX_ext);
		rho_tempx = space3d(NZ_ext, NY_ext, NX_ext);
		rho_tempy = space3d(NZ_ext, NY_ext, NX_ext);
		rho_tempz = space3d(NZ_ext, NY_ext, NX_ext);
		int ix, iy, iz, it;
		for (iz = 1; iz < NZ_ext - 1; iz++)
		{
			for (iy = 1; iy < NY_ext - 1; iy++)
			{
				for (ix = 1; ix < NX_ext - 1; ix++)
				{
					rho_tempx[iz][iy][ix] = 2 / (rho_ext[iz][iy][ix] + rho_ext[iz][iy][ix + 1]);
					rho_tempy[iz][iy][ix] = 2 / (rho_ext[iz][iy][ix] + rho_ext[iz][iy + 1][ix]);
					rho_tempz[iz][iy][ix] = 2 / (rho_ext[iz][iy][ix] + rho_ext[iz + 1][iy][ix]);

					if (mu_ext[iz][iy][ix] == 0 || mu_ext[iz][iy][ix + 1] == 0 || mu_ext[iz + 1][iy][ix] == 0 || mu_ext[iz + 1][iy][ix + 1] == 0)
						muxz[iz][iy][ix] = 0.0;
					else
						muxz[iz][iy][ix] = 1 / (0.25 * (1 / mu_ext[iz][iy][ix] + 1 / mu_ext[iz][iy][ix + 1] + 1 / mu_ext[iz + 1][iy][ix] + 1 / mu_ext[iz + 1][iy][ix + 1]));
					if (mu_ext[iz][iy][ix] == 0 || mu_ext[iz][iy + 1][ix] == 0 || mu_ext[iz + 1][iy][ix] == 0 || mu_ext[iz + 1][iy + 1][ix] == 0)
						muyz[iz][iy][ix] = 0.0;
					else
						muyz[iz][iy][ix] = 1 / (0.25 * (1 / mu_ext[iz][iy][ix] + 1 / mu_ext[iz][iy + 1][ix] + 1 / mu_ext[iz + 1][iy][ix] + 1 / mu_ext[iz + 1][iy + 1][ix]));
					if (mu_ext[iz][iy][ix] == 0 || mu_ext[iz][iy][ix + 1] == 0 || mu_ext[iz][iy + 1][ix] == 0 || mu_ext[iz][iy + 1][ix + 1] == 0)
						muxy[iz][iy][ix] = 0.0;
					else
						muxy[iz][iy][ix] = 1 / (0.25 * (1 / mu_ext[iz][iy][ix] + 1 / mu_ext[iz][iy][ix + 1] + 1 / mu_ext[iz][iy + 1][ix] + 1 / mu_ext[iz][iy + 1][ix + 1]));

				}
			}
		}
		/*char muxyname[] = "muxy";
		char muxy50name[50];
		sprintf(muxy50name, "%s %d.dat", muxyname, namepressure1);
		wfile3d(muxy50name, muxy, NZ_ext, NY_ext, NX_ext);

		char muxzname[] = "muxz";
		char muxz50name[50];
		sprintf(muxz50name, "%s %d.dat", muxzname, namepressure1);
		wfile3d(muxz50name, muxz, NZ_ext, NY_ext, NX_ext);

		char muyzname[] = "muyz";
		char muyz50name[50];
		sprintf(muyz50name, "%s %d.dat", muyzname, namepressure1);
		wfile3d(muyz50name, muyz, NZ_ext, NY_ext, NX_ext);*/

		//-----------------------------------------------并行计算-------------------------------------------------------------------------
		//-------------------------------------------------------------------------------------------------
		//在主机端CPU定义参数，分配内存
		float* h_mu_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_lamda_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_lamda2u_ext = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_rho_tempx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_rho_tempy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_rho_tempz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_muxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_muyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_muxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_vux = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_vuy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_vuz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_txx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_tyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_tzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_txz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_txy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_tyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlxSxx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlySxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlzSxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlxSxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlySyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlzSyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlxSxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlySyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlzSzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlxVux = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlyVuy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlzVuz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlxVuy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlyVux = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlxVuz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlzVux = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlyVuz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_pmlzVuy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_SXxx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_SXxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_SXxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_SYxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_SYyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_SYyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_SZxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_SZyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_SZzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_Vuxx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_Vuyy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_Vuzz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_Vuxy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_Vuyx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_Vuyz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_Vuzy = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_Vuxz = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		float* h_Vuzx = (float*)calloc(NZ_ext * NY_ext * NX_ext, sizeof(float));
		//---------------------------------------------------------------------------------
		//在主机端将三维参数转化为一维参数
		for (int k = 0; k < NZ_ext; k++)
		{
			for (int j = 0; j < NY_ext; j++)
			{
				for (int i = 0; i < NX_ext; i++)
				{
					h_mu_ext[i + j * NX_ext + k * NX_ext * NY_ext] = mu_ext[k][j][i];

					h_lamda_ext[i + j * NX_ext + k * NX_ext * NY_ext] = lamda_ext[k][j][i];
					h_lamda2u_ext[i + j * NX_ext + k * NX_ext * NY_ext] = lamda2u_ext[k][j][i];
					h_rho_tempx[i + j * NX_ext + k * NX_ext * NY_ext] = rho_tempx[k][j][i];
					h_rho_tempy[i + j * NX_ext + k * NX_ext * NY_ext] = rho_tempy[k][j][i];
					h_rho_tempz[i + j * NX_ext + k * NX_ext * NY_ext] = rho_tempz[k][j][i];

					h_muxz[i + j * NX_ext + k * NX_ext * NY_ext] = muxz[k][j][i];
					h_muxy[i + j * NX_ext + k * NX_ext * NY_ext] = muxy[k][j][i];
					h_muyz[i + j * NX_ext + k * NX_ext * NY_ext] = muyz[k][j][i];
				}
			}
		}
		//------------------------------------------------------------------------------------------------------------
		//在设备端GPU定义参数，分配内存
		float* d_lamda_ext, * d_lamda2u_ext, * d_rho_tempx, * d_rho_tempy, * d_rho_tempz, * d_muxz, * d_muxy, * d_muyz, * d_mu_ext;	   //初始模型密度
		float* d_vux, * d_vuy, * d_vuz;
		float* d_txx, * d_tyy, * d_tzz, * d_txy, * d_tyz, * d_txz, * d_ss;
		float* d_SXxx, * d_SXxy, * d_SXxz, * d_SYxy, * d_SYyy, * d_SYyz, * d_SZxz, * d_SZyz, * d_SZzz;
		float* d_Vuxx, * d_Vuyy, * d_Vuzz, * d_Vuxy, * d_Vuyx, * d_Vuxz, * d_Vuzx, * d_Vuyz, * d_Vuzy;
		float* d_pmlxSxx, * d_pmlySxy, * d_pmlzSxz, * d_pmlxss, * d_pmlxSxy, * d_pmlySyy, * d_pmlzSyz, * d_pmlyss, * d_pmlxSxz, * d_pmlySyz, * d_pmlzSzz, * d_pmlzss;
		float* d_pmlxVux, * d_pmlyVuy, * d_pmlzVuz, * d_pmlxVuy, * d_pmlyVux, * d_pmlxVuz, * d_pmlzVux, * d_pmlyVuz, * d_pmlzVuy;
		cudaMalloc(&d_mu_ext, mem_size);
		cudaMalloc(&d_lamda_ext, mem_size);
		cudaMalloc(&d_lamda2u_ext, mem_size);
		cudaMalloc(&d_rho_tempx, mem_size);
		cudaMalloc(&d_rho_tempy, mem_size);
		cudaMalloc(&d_rho_tempz, mem_size);
		cudaMalloc(&d_muxz, mem_size);
		cudaMalloc(&d_muyz, mem_size);
		cudaMalloc(&d_muxy, mem_size);
		cudaMalloc(&d_vux, mem_size);
		cudaMalloc(&d_vuy, mem_size);
		cudaMalloc(&d_vuz, mem_size);
		cudaMalloc(&d_txx, mem_size);
		cudaMalloc(&d_tyy, mem_size);
		cudaMalloc(&d_tzz, mem_size);
		cudaMalloc(&d_txz, mem_size);
		cudaMalloc(&d_txy, mem_size);
		cudaMalloc(&d_tyz, mem_size);
		cudaMalloc(&d_pmlxSxx, mem_size);
		cudaMalloc(&d_pmlySxy, mem_size);
		cudaMalloc(&d_pmlzSxz, mem_size);
		cudaMalloc(&d_pmlxSxy, mem_size);
		cudaMalloc(&d_pmlySyy, mem_size);
		cudaMalloc(&d_pmlzSyz, mem_size);
		cudaMalloc(&d_pmlxSxz, mem_size);
		cudaMalloc(&d_pmlySyz, mem_size);
		cudaMalloc(&d_pmlzSzz, mem_size);
		cudaMalloc(&d_pmlxVux, mem_size);
		cudaMalloc(&d_pmlyVuy, mem_size);
		cudaMalloc(&d_pmlzVuz, mem_size);
		cudaMalloc(&d_pmlxVuy, mem_size);
		cudaMalloc(&d_pmlyVux, mem_size);
		cudaMalloc(&d_pmlxVuz, mem_size);
		cudaMalloc(&d_pmlzVux, mem_size);
		cudaMalloc(&d_pmlyVuz, mem_size);
		cudaMalloc(&d_pmlzVuy, mem_size);
		cudaMalloc(&d_SXxx, mem_size);
		cudaMalloc(&d_SXxy, mem_size);
		cudaMalloc(&d_SXxz, mem_size);
		cudaMalloc(&d_SYxy, mem_size);
		cudaMalloc(&d_SYyy, mem_size);
		cudaMalloc(&d_SYyz, mem_size);
		cudaMalloc(&d_SZxz, mem_size);
		cudaMalloc(&d_SZyz, mem_size);
		cudaMalloc(&d_SZzz, mem_size);
		cudaMalloc(&d_Vuxx, mem_size);
		cudaMalloc(&d_Vuyy, mem_size);
		cudaMalloc(&d_Vuzz, mem_size);
		cudaMalloc(&d_Vuxy, mem_size);
		cudaMalloc(&d_Vuyx, mem_size);
		cudaMalloc(&d_Vuxz, mem_size);
		cudaMalloc(&d_Vuzx, mem_size);
		cudaMalloc(&d_Vuyz, mem_size);
		cudaMalloc(&d_Vuzy, mem_size);

		//---------------------------------------------------------------------------------------------
		//开始时间递推
		//这里的代码应该修正一下，应该从数据里面提取最大最小速度
		//检查稳定性条件
		best_dt = 6.0 * H / (7.0 * sqrt(2.0) * Vpmax);
		if (DT >= best_dt)
			printf("时间步长过大，应该小于 %.10f\n", best_dt);
		//控制网格频散
		if (Vsmin / (F0 * H) < 15)
			printf("空间步长太大,可能引起明显的网格频散\n");
		clock_t start, finish;
		start = clock();
		//--------------------------------------------------------------------------------------------
		//进行参数传递，将主机端CPU参数传递到设备端GPU
		cudaMemcpy(d_mu_ext, h_mu_ext, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_lamda_ext, h_lamda_ext, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_lamda2u_ext, h_lamda2u_ext, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rho_tempx, h_rho_tempx, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rho_tempy, h_rho_tempy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rho_tempz, h_rho_tempz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_muxz, h_muxz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_muxy, h_muxy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_muyz, h_muyz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vux, h_vux, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vuy, h_vuy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vuz, h_vuz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_txx, h_txx, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tyy, h_tyy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tzz, h_tzz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_txy, h_txy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_txz, h_txz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tyz, h_tyz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlxSxx, h_pmlxSxx, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlySxy, h_pmlySxy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlzSxz, h_pmlzSxz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlxSxy, h_pmlxSxy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlySyy, h_pmlySyy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlzSyz, h_pmlzSyz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlxSxz, h_pmlxSxz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlySyz, h_pmlySyz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlzSzz, h_pmlzSzz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlxVux, h_pmlxVux, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlyVuy, h_pmlyVuy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlzVuz, h_pmlzVuz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlxVuy, h_pmlxVuy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlyVux, h_pmlyVux, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlxVuz, h_pmlxVuz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlzVux, h_pmlzVux, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlyVuz, h_pmlyVuz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmlzVuy, h_pmlzVuy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_SXxx, h_SXxx, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_SXxy, h_SXxy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_SXxz, h_SXxz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_SYxy, h_SYxy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_SYyy, h_SYyy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_SYyz, h_SYyz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_SZxz, h_SZxz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_SZyz, h_SZyz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_SZzz, h_SZzz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Vuxx, h_Vuxx, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Vuyy, h_Vuyy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Vuzz, h_Vuzz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Vuxy, h_Vuxy, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Vuyx, h_Vuyx, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Vuxz, h_Vuxz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Vuzx, h_Vuzx, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Vuyz, h_Vuyz, mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Vuzy, h_Vuzy, mem_size, cudaMemcpyHostToDevice);

		//---------------------------------------------------------------------------------------------------------------------
		//在时间上进行迭代
		for (it = 0; it < NT; it++)
		{
			float tt = it * DT;
			if (it % 100 == 0)
			{
				printf("it=%d\n", it);
			}
			
			//加震源
			float I_sou;
			I_sou = -1 * (1 - 2 * PI * PI * F0 * F0 * (tt - T0) * (tt - T0)) * exp(-PI * PI * F0 * F0 * (tt - T0) * (tt - T0));
			//----------------------------------------------------------------------------------------------------
			//计算速度
			FD_V << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_rho_tempx, d_rho_tempy, d_rho_tempz, d_txx, d_tyy, d_tzz, d_txz, d_txy, d_tyz, d_pmlxSxx, d_pmlySxy, d_pmlzSxz, d_pmlxSxy, d_pmlySyy, d_pmlzSyz, d_pmlxSxz, d_pmlySyz, d_pmlzSzz, d_e_dxi2, d_dxi2, d_e_dyj, d_dyj, d_e_dzk, d_dzk, d_e_dxi, d_dxi, d_e_dyj2, d_dyj2, d_dzk2, d_e_dzk2, d_SXxx, d_SXxy, d_SXxz, d_SYxy, d_SYyy, d_SYyz, d_SZxz, d_SZyz, d_SZzz, DT, H, I_sou, sn, waveType);
			cudaDeviceSynchronize();
			//----------------------------------------------------------------------------------------------------------
			//计算应力
			FD_T << <Gridsize, Blocksize >> > (d_vux, d_vuy, d_vuz, d_muxy, d_muxz, d_muyz, d_txx, d_tzz, d_tyy, d_txz, d_txy, d_tyz, d_lamda2u_ext, d_lamda_ext, d_pmlxVux, d_pmlyVuy, d_pmlzVuz, d_pmlxVuy, d_pmlyVux, d_pmlyVuz, d_pmlzVuy, d_pmlzVux, d_pmlxVuz, d_dxi, d_e_dxi, d_dyj, d_e_dyj, d_dzk, d_e_dzk, d_dyj2, d_e_dyj2, d_dxi2, d_e_dxi2, d_dzk2, d_e_dzk2, d_Vuxx, d_Vuyy, d_Vuzz, d_Vuxy, d_Vuyx, d_Vuxz, d_Vuzx, d_Vuyz, d_Vuzy, DT, H, AoeCs, BoeCs, CoeCs, pressure0, d_mu_ext);
			cudaDeviceSynchronize();
			//----------------------------------------------------------------------------------------------------------
			//将速度，应力从设备端拷贝到主机
			cudaMemcpy(h_txx, d_txx, mem_size, cudaMemcpyDeviceToHost);
			//cudaMemcpy(h_tzz, d_tzz, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_vux, d_vux, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_vuy, d_vuy, mem_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_vuz, d_vuz, mem_size, cudaMemcpyDeviceToHost);
			//记录地震记录
			for (iz = 0; iz < NZ_ext; iz++)
			{
				sis_x[iz][it] = h_txx[iz * NX_ext * NY_ext + sx + sy * NX_ext];
				//sis_z[iz][it] = h_tzz[iz * NX_ext * NY_ext + sx + sy * NX_ext];
				sis_vx[iz][it] = h_vux[iz * NX_ext * NY_ext + sx + sy * NX_ext];
				sis_vy[iz][it] = h_vuy[iz * NX_ext * NY_ext + sx + sy * NX_ext];
				sis_vz[iz][it] = h_vuz[iz * NX_ext * NY_ext + sx + sy * NX_ext];
			}
			if (it == 500)
				for (int k = 0; k < NZ_ext; k++)
				{
					for (int j = 0; j < NY_ext; j++)
					{
						for (int i = 0; i < NX_ext; i++)
						{
							txx50[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
							vuz50[k][j][i] = h_vuz[i + j * NX_ext + k * NX_ext * NY_ext];
						}
					}
				}
			if (it == 1000)
				for (int k = 0; k < NZ_ext; k++)
				{
					for (int j = 0; j < NY_ext; j++)
					{
						for (int i = 0; i < NX_ext; i++)
						{
							txx100[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
							vuz100[k][j][i] = h_vuz[i + j * NX_ext + k * NX_ext * NY_ext];
						}
					}
				}
			if (it == 1500)
				for (int k = 0; k < NZ_ext; k++)
				{
					for (int j = 0; j < NY_ext; j++)
					{
						for (int i = 0; i < NX_ext; i++)
						{
							txx150[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
							vuz150[k][j][i] = h_vuz[i + j * NX_ext + k * NX_ext * NY_ext];
						}
					}
				}
			if (it == 2000)
				for (int k = 0; k < NZ_ext; k++)
				{
					for (int j = 0; j < NY_ext; j++)
					{
						for (int i = 0; i < NX_ext; i++)
						{
							txx200[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
							vuz200[k][j][i] = h_vuz[i + j * NX_ext + k * NX_ext * NY_ext];
						}
					}
				}
			if (it == 2500)
				for (int k = 0; k < NZ_ext; k++)
				{
					for (int j = 0; j < NY_ext; j++)
					{
						for (int i = 0; i < NX_ext; i++)
						{
							txx250[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
							vuz250[k][j][i] = h_vuz[i + j * NX_ext + k * NX_ext * NY_ext];
						}
					}
				}
			if (it == 3000)
				for (int k = 0; k < NZ_ext; k++)
				{
					for (int j = 0; j < NY_ext; j++)
					{
						for (int i = 0; i < NX_ext; i++)
						{
							txx300[k][j][i] = h_txx[i + j * NX_ext + k * NX_ext * NY_ext];
							vuz300[k][j][i] = h_vuz[i + j * NX_ext + k * NX_ext * NY_ext];
						}
					}
				}
		}
		finish = clock();
		printf("%f seconds\n", (float)(finish - start) / CLOCKS_PER_SEC);
		//--------------------------------------------------------------------------------------------------------------
		//输出波场快照
		printf("NZ_ext = %d NY_ext = %d NX_ext = %d\n", NZ_ext, NY_ext, NX_ext);
		//定义输出文件名
		int namepressure = pressure[prenumber];
		//输出地震记录
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

		//输出应力
		char txx50name[] = "txx500";
		char txxx50name[50];
		sprintf(txxx50name, "%s %d.dat", txx50name, namepressure);
		wfile3d(txxx50name, txx50, NZ_ext, NY_ext, NX_ext);

		char txx100name[] = "txx1000";
		char txxx100name[50];
		sprintf(txxx100name, "%s %d.dat", txx100name, namepressure);
		wfile3d(txxx100name, txx100, NZ_ext, NY_ext, NX_ext);

		char txx150name[] = "txx1500";
		char txxx150name[50];
		sprintf(txxx150name, "%s %d.dat", txx150name, namepressure);
		wfile3d(txxx150name, txx150, NZ_ext, NY_ext, NX_ext);

		char txx200name[] = "txx2000";
		char txxx200name[50];
		sprintf(txxx200name, "%s %d.dat", txx200name, namepressure);
		wfile3d(txxx200name, txx200, NZ_ext, NY_ext, NX_ext);

		char txx250name[] = "txx2500";
		char txxx250name[50];
		sprintf(txxx250name, "%s %d.dat", txx250name, namepressure);
		wfile3d(txxx250name, txx250, NZ_ext, NY_ext, NX_ext);

		char txx300name[] = "txx3000";
		char txxx300name[50];
		sprintf(txxx300name, "%s %d.dat", txx300name, namepressure);
		wfile3d(txxx300name, txx300, NZ_ext, NY_ext, NX_ext);

		//输出速度
		char vux50name[] = "vuz500";
		char vuxx50name[50];
		sprintf(vuxx50name, "%s %d.dat", vux50name, namepressure);
		wfile3d(vuxx50name, vuz50, NZ_ext, NY_ext, NX_ext);

		char vux100name[] = "vuz1000";
		char vuxx100name[50];
		sprintf(vuxx100name, "%s %d.dat", vux100name, namepressure);
		wfile3d(vuxx100name, vuz100, NZ_ext, NY_ext, NX_ext);

		char vux150name[] = "vuz1500";
		char vuxx150name[50];
		sprintf(vuxx150name, "%s %d.dat", vux150name, namepressure);
		wfile3d(vuxx150name, vuz150, NZ_ext, NY_ext, NX_ext);

		char vux200name[] = "vuz2000";
		char vuxx200name[50];
		sprintf(vuxx200name, "%s %d.dat", vux200name, namepressure);
		wfile3d(vuxx200name, vuz200, NZ_ext, NY_ext, NX_ext);

		char vux250name[] = "vuz2500";
		char vuxx250name[50];
		sprintf(vuxx250name, "%s %d.dat", vux250name, namepressure);
		wfile3d(vuxx250name, vuz250, NZ_ext, NY_ext, NX_ext);

		char vux300name[] = "vuz3000";
		char vuxx300name[50];
		sprintf(vuxx300name, "%s %d.dat", vux300name, namepressure);
		wfile3d(vuxx300name, vuz300, NZ_ext, NY_ext, NX_ext);
		//---------------------------------------------------------------------------------------------------------------
		//主机端释放内存
		free(h_mu_ext);
		free(h_muxy);
		free(h_muxz);
		free(h_muyz);
		free(h_rho_tempx);
		free(h_rho_tempy);
		free(h_rho_tempz);
		free(h_pmlxSxx);
		free(h_pmlxSxy);
		free(h_pmlxSxz);
		free(h_pmlxVux);
		free(h_pmlxVuy);
		free(h_pmlxVuz);
		free(h_pmlySxy);
		free(h_pmlySyy);
		free(h_pmlySyz);
		free(h_pmlyVux);
		free(h_pmlyVuy);
		free(h_pmlyVuz);
		free(h_pmlzSxz);
		free(h_pmlzSyz);
		free(h_pmlzSzz);
		free(h_pmlzVux);
		free(h_pmlzVuy);
		free(h_pmlzVuz);
		free(h_SXxx);
		free(h_SXxy);
		free(h_SXxz);
		free(h_SYxy);
		free(h_SYyy);
		free(h_SYyz);
		free(h_SZxz);
		free(h_SZyz);
		free(h_SZzz);
		free(h_Vuxx);
		free(h_Vuxy);
		free(h_Vuxz);
		free(h_Vuyx);
		free(h_Vuyy);
		free(h_Vuyz);
		free(h_Vuzx);
		free(h_Vuzy);
		free(h_Vuzz);

		free_space3d(rho_tempx, NZ_ext, NY_ext);
		free_space3d(rho_tempy, NZ_ext, NY_ext);
		free_space3d(rho_tempz, NZ_ext, NY_ext);
		free_space3d(muxy, NZ_ext, NY_ext);
		free_space3d(muxz, NZ_ext, NY_ext);
		free_space3d(muyz, NZ_ext, NY_ext);
		free_space3d(lamda_ext, NZ_ext, NY_ext);
		free_space3d(lamda2u_ext, NZ_ext, NY_ext);

		free_space3d(rho_ext, NZ_ext, NY_ext);
		free_space3d(mu_ext, NZ_ext, NY_ext);
		free_space3d(rhof_ext, NZ_ext, NY_ext);

		free_space3d(txx50, NZ_ext, NY_ext);
		free_space3d(txx100, NZ_ext, NY_ext);
		free_space3d(txx150, NZ_ext, NY_ext);
		free_space3d(txx200, NZ_ext, NY_ext);
		free_space3d(txx250, NZ_ext, NY_ext);
		free_space3d(txx300, NZ_ext, NY_ext);
		free_space3d(vuz50, NZ_ext, NY_ext);
		free_space3d(vuz100, NZ_ext, NY_ext);
		free_space3d(vuz150, NZ_ext, NY_ext);
		free_space3d(vuz200, NZ_ext, NY_ext);
		free_space3d(vuz250, NZ_ext, NY_ext);
		free_space3d(vuz300, NZ_ext, NY_ext);

		free(h_txx);
		free(h_tyy);
		free(h_tzz);
		free(h_txy);
		free(h_txz);
		free(h_tyz);
		free(h_vux);
		free(h_vuy);
		free(h_vuz);

		free_space2d(sis_x, NZ_ext);
		free_space2d(sis_y, NZ_ext);
		free_space2d(sis_z, NZ_ext);
		free_space2d(sis_vx, NZ_ext);
		free_space2d(sis_vy, NZ_ext);
		free_space2d(sis_vz, NZ_ext);
		free_space3d(tzz, NZ_ext, NY_ext);
		free_space3d(tyy, NZ_ext, NY_ext);
		free_space3d(txx, NZ_ext, NY_ext);
		free_space3d(txy, NZ_ext, NY_ext);
		free_space3d(txz, NZ_ext, NY_ext);
		free_space3d(tyz, NZ_ext, NY_ext);
		free_space3d(vux, NZ_ext, NY_ext);
		free_space3d(vuy, NZ_ext, NY_ext);
		free_space3d(vuz, NZ_ext, NY_ext);


		//设备端释放内存
		cudaFree(d_mu_ext);
		cudaFree(d_lamda2u_ext);
		cudaFree(d_lamda_ext);
		cudaFree(d_muxy);
		cudaFree(d_muxz);
		cudaFree(d_muyz);
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
		cudaFree(d_rho_tempx);
		cudaFree(d_rho_tempy);
		cudaFree(d_rho_tempz);
		cudaFree(d_SXxx);
		cudaFree(d_SXxy);
		cudaFree(d_SXxz);
		cudaFree(d_SYxy);
		cudaFree(d_SYyy);
		cudaFree(d_SYyz);
		cudaFree(d_SZxz);
		cudaFree(d_SZyz);
		cudaFree(d_SZzz);
		cudaFree(d_txx);
		cudaFree(d_tyy);
		cudaFree(d_tzz);
		cudaFree(d_txy);
		cudaFree(d_txz);
		cudaFree(d_tyz);
		cudaFree(d_vux);
		cudaFree(d_vuy);
		cudaFree(d_vuz);
		cudaFree(d_Vuxx);
		cudaFree(d_Vuxy);
		cudaFree(d_Vuxz);
		cudaFree(d_Vuyx);
		cudaFree(d_Vuyy);
		cudaFree(d_Vuyz);
		cudaFree(d_Vuzx);
		cudaFree(d_Vuzy);
		cudaFree(d_Vuzz);

	}
	free_space3d(elastic_pre_all, NZ, NY);
	free_space3d(elastic_pre, NZZ, NY);
	cudaFree(d_dxi);
	cudaFree(d_dxi2);
	cudaFree(d_dyj);
	cudaFree(d_dyj2);
	cudaFree(d_dzk);
	cudaFree(d_dzk2);
	cudaFree(d_e_dxi);
	cudaFree(d_e_dxi2);
	cudaFree(d_e_dyj);
	cudaFree(d_e_dyj2);
	cudaFree(d_e_dzk);
	cudaFree(d_e_dzk2);
	printf("\Press any key to exit program...");
	getch();
	return 0;
}

//申请二维动态数组
float** space2d(int nr, int nc)
{
	float** a;
	int i;
	a = (float**)calloc(nr, sizeof(float*));
	for (i = 0; i < nr; i++)
		a[i] = (float*)calloc(nc, sizeof(float));

	return a;
}

//释放二维动态数组
void free_space2d(float** a, int nr)
{
	int i;
	for (i = 0; i < nr; i++)
		free(a[i]);
	free(a);
}

//将二进制数据写入文件
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

//申请三维动态数组
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
			a[i][j] = (float*)malloc(sizeof(float) * nc);//sizeof(float) nc改为sizeof(float)*nc
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

//释放三维动态数组
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

//将二进制数据写入文件—三维
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
				j = ny / 2;//y轴切片
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

// 导出三维数组到文件
void exportArrayToFile(float*** a, int xSize, int ySize, int zSize, const char* filename) {
	FILE* file = fopen(filename, "w");
	if (file == NULL) {
		printf("Error opening file!\n");
		return;
	}

	for (int i = 0; i < xSize; i++) {
		for (int j = 0; j < ySize; j++) {
			for (int k = 0; k < zSize; k++) {
				fprintf(file, "%d %d %d %f\n", i, j, k, a[i][j][k]);
			}
		}
	}

	fclose(file);
}

// 填充函数实现
void fillCube(float*** space, int z, int y, int x, int length, float value) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < length; j++) {
			for (int k = 0; k < length; k++) {
				space[z + i][y + j][x + k] = value;
			}
		}
	}
}

void create_model(float*** vp, float*** vs, float*** rho, float*** vf, float*** rhof, float*** lamda, float*** lamda2u, float*** mu, int nr, int ny, int nc, float pressure, float*** elastic_pre)
{
	int ix, iy, iz;
	//定义温度
	float Temp = 20;
	//定义各个矿物的速度参数
	float Vs_mineral[9];
	float Vp_mineral[9];
	float rho_mineral[9];

	// 孔隙 油or气
	Vp_mineral[1] = 629.7f;
	Vs_mineral[1] = 0.0f;
	rho_mineral[1] = 139.8f;
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
	Vp_mineral[7] = 1500.0f;
	Vs_mineral[7] = 0.0f;
	rho_mineral[7] = 1000.0f;
	// 平均部分
	Vp_mineral[8] = 3000.0f;
	Vs_mineral[8] = 1700.0f;
	rho_mineral[8] = 2292.7f;




	//对rho，vs，vp赋值,计算弹性系数
	for (iz = 0; iz < nr; iz++)
	{
		for (iy = 0; iy < ny; iy++)
		{
			for (ix = 0; ix < nc; ix++)
			{

				vs[iz][iy][ix] = Vs_mineral[(int)elastic_pre[iz][iy][ix]];
				vp[iz][iy][ix] = Vp_mineral[(int)elastic_pre[iz][iy][ix]];
				rho[iz][iy][ix] = rho_mineral[(int)elastic_pre[iz][iy][ix]];
				mu[iz][iy][ix] = rho[iz][iy][ix] * vs[iz][iy][ix] * vs[iz][iy][ix];
				lamda2u[iz][iy][ix] = rho[iz][iy][ix] * vp[iz][iy][ix] * vp[iz][iy][ix];
				lamda[iz][iy][ix] = lamda2u[iz][iy][ix] - 2.0 * mu[iz][iy][ix];
			}
		}
	}

}

