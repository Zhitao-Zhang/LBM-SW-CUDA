#pragma hdrstop
//---------------------------------------------------------------------------
// #pragma argsused
// #define SW9
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NZ 50
#define NY 50
#define NX 50
#define npop 19
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8

typedef struct {
	int z;
	int y;
	int x;
} Index3D;

float*** space3d(int nr, int ny, int nc);
float**** space4d(int n1, int n2, int n3, int n4);
void free_space3d(float*** b, int nr, int ny);
void free_space4d(float**** a, int nr, int ny, int nc);
Index3D* find_nonzero_elements(float*** array, int* Boundray);
void zero3dArray(float*** array, int nr, int ny, int nc);
int countElements(float*** rock, int lx, int ly, int lz, float t);
void wfile1d(char filename[], float* data, int nz, int ny, int nx);
void wfile3d(char filename[], float*** data, int nr, int ny, int nc);
void write_data_to_file(float*** data, int nz, int ny, int nx);
void write_4d_data_to_file(float**** data, int nh, int nz, int ny, int nx);
int divUp(int a, int b) { return (a + b - 1) / b; }


__global__ void CalTempt(float* tempt, float* ff, float Boundary, int* nonzero_indices_x, int* nonzero_indices_y, int* nonzero_indices_z)
{
	int bod = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < npop; i++) {
		int index = i * NX * NY * NZ + nonzero_indices_z[bod] * NX * NY + nonzero_indices_y[bod] * NX + nonzero_indices_x[bod];
		tempt[index] = ff[index];
	}
}

__global__ void CalBGK(float* u_x, float* u_y, float* u_z, float* ux, float* uy, float* uz,
	float* F1sx, float* F1sy, float* F1sz, float* Fmx1, float* Fmy1, float* Fmz1, float* Fmx2, float* Fmy2, float* Fmz2, float* rho, float* cu, float* feq, float* ff, float tau)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int cx1[npop] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 0, 0, 0, 0 };
	int cy1[npop] = { 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1 };
	int cz1[npop] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
	float weA1 = 1.0 / 3.0;
	float weA2 = 1.0 / 18.0;
	float weA3 = 1.0 / 36.0;
	float weight[npop] = { weA1, weA2, weA2, weA2, weA2, weA2, weA2, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3 };

	if (ix < NX && iy < NY && iz < NZ)
	{
		int offset = ix + iy * NX + iz * NX * NY;

		u_x[offset] = ux[offset] + tau * (F1sx[offset] + Fmx1[offset] + Fmx2[offset]) / rho[offset];
		u_y[offset] = uy[offset] + tau * (F1sy[offset] + Fmy1[offset] + Fmy2[offset]) / rho[offset];
		u_z[offset] = uz[offset] + tau * (F1sz[offset] + Fmz1[offset] + Fmz2[offset]) / rho[offset];
		for (int i = 0; i < npop; i++)
		{

			int offset4D = i * NX * NY * NZ + ix + iy * NX + iz * NX * NY;
			cu[offset] = 3.0 * (cx1[i] * u_x[offset] + cy1[i] * u_y[offset] + cz1[i] * u_z[offset]);

			feq[offset4D] = rho[offset] * weight[i] * (1.0 + cu[offset] + 1.0 / 2.0 * (cu[offset] * cu[offset]) - 3.0 / 2.0 * (u_x[offset] * u_x[offset] + u_y[offset] * u_y[offset] + u_z[offset] * u_z[offset]));


			ff[offset4D] = ff[offset4D] - 1.0 / tau * (ff[offset4D] - feq[offset4D]);


		}

	}
}

__global__ void UpdateBoundary(float* ff, float* tempt, float Boundary, int* nonzero_indices_x, int* nonzero_indices_y, int* nonzero_indices_z)
{
	int opp[npop] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17 };
	int bod = blockIdx.x * blockDim.x + threadIdx.x;
	if (bod < Boundary)
	{
		for (int i = 0; i < npop; i++)
		{
			int index = i * NX * NY * NZ + nonzero_indices_z[bod] * NX * NY + nonzero_indices_y[bod] * NX + nonzero_indices_x[bod];
			int index1 = opp[i] * NX * NY * NZ + nonzero_indices_z[bod] * NX * NY + nonzero_indices_y[bod] * NX + nonzero_indices_x[bod];

			ff[index] = tempt[index1];

		}
	}
}

__global__ void UpdateMidff(float* fftemp, float* ff, int i)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int cx[npop] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 0, 0, 0, 0 };
	int cy[npop] = { 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1 };
	int cz[npop] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
	if (ix < NX && iy < NY && iz < NZ)
	{
		int offset4D = i * NX * NY * NZ + ix + iy * NX + iz * NX * NY;
		int newZ = (iz + cz[i] + NZ) % NZ;
		int newY = (iy + cy[i] + NY) % NY;
		int newX = (ix + cx[i] + NX) % NX;
		int offset = newZ * NX * NY + newY * NX + newX;
		fftemp[offset] = ff[offset4D];
	}
}

__global__ void Updateff(float* fftemp, float* ff, int i)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	if (ix < NX && iy < NY && iz < NZ)
	{
		int index = ix + iy * NX + iz * NX * NY;

		int offset4D = i * NX * NY * NZ + ix + iy * NX + iz * NX * NY;
		ff[offset4D] = fftemp[index];
	}
}

__global__ void Updaterho(float* rho, float* ff)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	if (ix < NX && iy < NY && iz < NZ)
	{
		int index = ix + iy * NX + iz * NX * NY;

		rho[index] = 0.0;

		for (int i = 0; i < npop; i++)
		{
			int offset4D = i * NX * NY * NZ + ix + iy * NX + iz * NX * NY;
			rho[index] += ff[offset4D];
		}
	}
}

__global__ void UpdateUPot(float* ux, float* uy, float* uz, float* ff, float* pp, float* rho, float G1, float* psx)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int cx1[npop] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 0, 0, 0, 0 };
	int cy1[npop] = { 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1 };
	int cz1[npop] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };

	if (ix < NX && iy < NY && iz < NZ)
	{
		int index = ix + iy * NX + iz * NX * NY;


		ux[index] = 0.0;
		uy[index] = 0.0;
		uz[index] = 0.0;

		for (int i = 0; i < npop; i++)
		{
			int offset4D = i * NX * NY * NZ + ix + iy * NX + iz * NX * NY;
			ux[index] = ux[index] + ff[offset4D] * cx1[i];
			uy[index] = uy[index] + ff[offset4D] * cy1[i];
			uz[index] = uz[index] + ff[offset4D] * cz1[i];

		}
		ux[index] = ux[index] / rho[index];
		uy[index] = uy[index] / rho[index];
		uz[index] = uz[index] / rho[index];

		pp[index] = rho[index] / 3.0 + 3.0 * G1 * (1.0 - exp(-rho[index])) * (1.0 - exp(-rho[index]));
		psx[index] = sqrt(fabs((pp[index] - rho[index] / 3.0) / G1) * 6.0);
	}
}

__global__ void PotBoundary(float* psx, float Boundary, int* nonzero_indices_x, int* nonzero_indices_y, int* nonzero_indices_z)
{
	int bod = blockIdx.x * blockDim.x + threadIdx.x;
	if (bod < Boundary)
	{
		int index = nonzero_indices_z[bod] * NX * NY + nonzero_indices_y[bod] * NX + nonzero_indices_x[bod];
		
		psx[index] = 0.0;
		
	}
}

__global__ void CalP(float* p, float* rho, float G1, float* psx)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	if (ix < NX && iy < NY && iz < NZ)
	{
		int index = ix + iy * NX + iz * NX * NY;

		p[index] = rho[index] / 3.0 + G1 / 6.0 * psx[index] * psx[index];
	}
}

__global__ void UpdateMidPot(float* psxpretemp, float* psx, float* spretemp, float* obst, int i)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int cx[npop] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 0, 0, 0, 0 };
	int cy[npop] = { 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1 };
	int cz[npop] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };

	if (ix < NX && iy < NY && iz < NZ)
	{
		int offset3D = ix + iy * NX + iz * NX * NY;
		int newZ = (iz - cz[i] + NZ) % NZ;
		int newY = (iy - cy[i] + NY) % NY;
		int newX = (ix - cx[i] + NX) % NX;
		int offset = newZ * NX * NY + newY * NX + newX;

		psxpretemp[offset] = psx[offset3D];
		spretemp[offset] = obst[offset3D];
	}
}

__global__ void UpdatePot(float* psxpre, float* psxpretemp, float* spre, float* spretemp, int i)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	if (ix < NX && iy < NY && iz < NZ)
	{
		int index4D = i * NX * NY * NZ + iz * NX * NY + iy * NX + ix;
		int index = iz * NX * NY + iy * NX + ix;

		psxpre[index4D] = psxpretemp[index];
		spre[index4D] = spretemp[index];
	}
}

__global__ void CalF(float* smx_pre1, float* smy_pre1, float* smz_pre1, float* Fmx1, float* Fmy1, float* Fmz1, float* Fmx2, float* Fmy2, float* Fmz2, float* spre, float* psxpre)
{
	float weA3 = 1.0 / 36.0;
	float weA4 = 1.0 / 6.0;
	float weA5 = 1.0 / 12.0;
	int cx1[npop] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 0, 0, 0, 0 };
	int cy1[npop] = { 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1 };
	int cz1[npop] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };

	float weight1[npop] = { 0.0, weA4, weA4, weA4, weA4, weA4, weA4, weA5, weA5, weA5, weA5, weA5, weA5, weA5, weA5, weA5, weA5, weA5, weA5 };
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	if (ix < NX && iy < NY && iz < NZ)
	{
		int index = iz * NX * NY + iy * NX + ix;

		smx_pre1[index] = 0.0;
		smy_pre1[index] = 0.0;
		smz_pre1[index] = 0.0;

		Fmx1[index] = 0.0;
		Fmy1[index] = 0.0;
		Fmz1[index] = 0.0;
		Fmx2[index] = 0.0;
		Fmy2[index] = 0.0;
		Fmz2[index] = 0.0;

		for (int i = 0; i < npop; i++)
		{
			int index4D = i * NX * NY * NZ + iz * NX * NY + iy * NX + ix;
			smx_pre1[index] = smx_pre1[index] + weight1[i] * spre[index4D] * cx1[i];
			smy_pre1[index] = smy_pre1[index] + weight1[i] * spre[index4D] * cy1[i];
			smz_pre1[index] = smz_pre1[index] + weight1[i] * spre[index4D] * cz1[i];
			Fmx1[index] = Fmx1[index] + weight1[i] * psxpre[index4D] * cx1[i];
			Fmy1[index] = Fmy1[index] + weight1[i] * psxpre[index4D] * cy1[i];
			Fmz1[index] = Fmz1[index] + weight1[i] * psxpre[index4D] * cz1[i];
			Fmx2[index] = Fmx2[index] + weight1[i] * psxpre[index4D] * cx1[i] * psxpre[index4D];
			Fmy2[index] = Fmy2[index] + weight1[i] * psxpre[index4D] * cy1[i] * psxpre[index4D];
			Fmz2[index] = Fmz2[index] + weight1[i] * psxpre[index4D] * cz1[i] * psxpre[index4D];
		}
	}
}

__global__ void BoundaryF(float* Fmx1, float* Fmy1, float* Fmz1, float* Fmx2, float* Fmy2, float* Fmz2, float* smx_pre1, float* smy_pre1, float* smz_pre1, int* nonzero_indices_x, int* nonzero_indices_y, int* nonzero_indices_z, float Boundary)
{
	int bod = blockIdx.x * blockDim.x + threadIdx.x;
	if (bod < Boundary)
	{
		int index = nonzero_indices_z[bod] * NX * NY + nonzero_indices_y[bod] * NX + nonzero_indices_x[bod];

		Fmx1[index] = 0.0;
		Fmy1[index] = 0.0;
		Fmz1[index] = 0.0;
		Fmx2[index] = 0.0;
		Fmy2[index] = 0.0;
		Fmz2[index] = 0.0;
		smx_pre1[index] = 0.0;
		smy_pre1[index] = 0.0;
		smz_pre1[index] = 0.0;
	}
}

__global__ void UpdateF(float* F1sx, float* F1sy, float* F1sz, float* Fmx1, float* Fmy1, float* Fmz1, float* Fmx2, float* Fmy2, float* Fmz2, float* psx, float* smx_pre1, float* smy_pre1, float* smz_pre1, float G1, float G2, float B)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	if (ix < NX && iy < NY && iz < NZ)
	{
		int index = iz * NX * NY + iy * NX + ix;

		F1sx[index] = -psx[index] * psx[index] * smx_pre1[index] * G2; // 流固作用力
		F1sy[index] = -psx[index] * psx[index] * smy_pre1[index] * G2;
		F1sz[index] = -psx[index] * psx[index] * smz_pre1[index] * G2;

		Fmx1[index] = -psx[index] * Fmx1[index] * G1 / 3.0 * B; // 流体内部作用力
		Fmy1[index] = -psx[index] * Fmy1[index] * G1 / 3.0 * B;
		Fmz1[index] = -psx[index] * Fmz1[index] * G1 / 3.0 * B;
		Fmx2[index] = -Fmx2[index] * G1 / 3.0 * (1.0 - B) / 2.0;
		Fmy2[index] = -Fmy2[index] * G1 / 3.0 * (1.0 - B) / 2.0;
		Fmz2[index] = -Fmz2[index] * G1 / 3.0 * (1.0 - B) / 2.0;
	}
}



int main()
{
	//三维数组
	dim3 Blocksize = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 Gridsize = dim3(divUp(NX, BLOCK_SIZE_X), divUp(NY, BLOCK_SIZE_Y), divUp(NZ, BLOCK_SIZE_Z));
	size_t mem_size = NZ * NY * NX * sizeof(float);
	//四维数组
	dim3 Blocksize4d = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 Gridsize4d = dim3(divUp(NX, BLOCK_SIZE_X), divUp(NY, BLOCK_SIZE_Y), divUp(NZ * npop, BLOCK_SIZE_Z));
	size_t mem_size4d = npop * NZ * NY * NX * sizeof(float);

	int Nall = NX * NY * NZ;
	float*** data11;
	float*** obst;
	float*** data;
	//读取文件
	data11 = space3d(NZ, NY, NX);
	obst = space3d(NZ, NY, NX);
	data = space3d(NZ, NY, NX);
	printf("input begin");
	// 打开文件
	FILE* file11;
	//file11 = fopen("C:/Users/41480/Desktop/LBM/LBM/200x200x200.dat", "r");
	file11 = fopen("D:/LBM/LBM/程序50x50x50/200x200x200.dat", "r");
	if (file11 == NULL) {
		fprintf(stderr, "无法打开文件\n");
		return 1;
	}
	// 读取文件内容并存储到数组中
	int INPUTnz, INPUTny, INPUTnx;
	float INPUTvalue;
	while (fscanf(file11, "%d %d %d %f", &INPUTnz, &INPUTny, &INPUTnx, &INPUTvalue) != EOF) {
		if (INPUTnz >= 0 && INPUTnz < NZ && INPUTny >= 0 && INPUTny < NY && INPUTnx >= 0 && INPUTnx < NX) {
			data11[INPUTnz][INPUTny][INPUTnx] = INPUTvalue;
		}
		else {
			fprintf(stderr, "索引超出范围: nz=%d, ny=%d, nx=%d\n", INPUTnz, INPUTny, INPUTnx);
		}
	}
	// 关闭文件
	fclose(file11);



	for (int z = 0; z < NZ; z++)
		for (int y = 0; y < NY; y++)
			for (int x = 0; x < NX; x++)
			{
				data[z][y][x] = data11[z][y][x];

				if (data11[z][y][x] == 1)
					data11[z][y][x] = 0;
				else
					data11[z][y][x] = 1;

				obst[z][y][x] = data11[z][y][x];
			}
	float*** ux;
	float*** uy;
	float*** uz;
	float*** p;
	float*** rho;
	float*** water;
	float*** oil;
	float*** pp;
	float*** u_x;
	float*** u_y;
	float*** u_z;
	ux = space3d(NZ, NY, NX);
	uy = space3d(NZ, NY, NX);
	uz = space3d(NZ, NY, NX);
	p = space3d(NZ, NY, NX);
	rho = space3d(NZ, NY, NX);
	water = space3d(NZ, NY, NX);
	oil = space3d(NZ, NY, NX);
	pp = space3d(NZ, NY, NX);
	u_x = space3d(NZ, NY, NX);
	u_y = space3d(NZ, NY, NX);
	u_z = space3d(NZ, NY, NX);

	//****************
	//测试用数组
	float*** test;
	float**** test4d;
	test = space3d(NZ, NY, NX);
	test4d = space4d(npop, NZ, NY, NX);
	//****************
	float**** ff;
	float**** feq;
	ff = space4d(npop, NZ, NY, NX);
	feq = space4d(npop, NZ, NY, NX);
	float weA1 = 1.0 / 3.0;
	float weA2 = 1.0 / 18.0;
	float weA3 = 1.0 / 36.0;
	float weight[npop] = { weA1, weA2, weA2, weA2, weA2, weA2, weA2, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3 };

	//定义参数
	float cc = 1.0;
	float csqu = (cc * cc) / 3.0;
	float rho_h = 1.9;
	float rho_l = 0.14;
	float rho_ratio = 1.5;//初始密度比，变饱和度调整这里，0.14-1.9之间
	float RR = 10.0;		// RR droplet radius
	float rho_w = 0.03;	// rho_w, density of wall
	float tau = 1.0f;		// Relaxtion parameter
	float t_max = 1000;	// Maximum time step
	float Nwri = 500.0;	// output every Nwri time steps
	float G1 = -0.2777778;
	float G2 = -0.12; // 调整接触角的系数， - 0.12是亲水
	float B = 0.886;
	int Boundary;
	//find边界
	Index3D* nonzero_indices = find_nonzero_elements(obst, &Boundary);
	printf("boundary = %d\n", Boundary);
	//一维数组
	int threadsPerBlock = 128;
	int blocksPerGrid = (Boundary + threadsPerBlock - 1) / threadsPerBlock;

	int* h_nonzero_indicesX = (int*)calloc(Boundary, sizeof(int));
	int* h_nonzero_indicesY = (int*)calloc(Boundary, sizeof(int));
	int* h_nonzero_indicesZ = (int*)calloc(Boundary, sizeof(int));
	for (int i = 0; i < Boundary; i++)
	{
		h_nonzero_indicesX[i] = (int)nonzero_indices[i].x;
		h_nonzero_indicesY[i] = (int)nonzero_indices[i].y;
		h_nonzero_indicesZ[i] = (int)nonzero_indices[i].z;
	}
	/*for (int i = 0; i < Boundary; i++)
	{
		printf("boundray index = [%d][%d][%d]        boundray index = [%d][%d][%d]\n", nonzero_indices[i].z, nonzero_indices[i].y, nonzero_indices[i].x, h_nonzero_indicesZ[i], h_nonzero_indicesY[i], h_nonzero_indicesX[i]);

	}*/


	for (int z = 0; z < NZ; z++)
		for (int y = 0; y < NY; y++)
			for (int x = 0; x < NX; x++)
			{
				if (obst[z][y][x] == 0)
					rho[z][y][x] = rho_ratio;
				else
					rho[z][y][x] = rho_l;

				for (int i = 0; i < npop; i++)
				{
					ff[i][z][y][x] = rho[z][y][x] * weight[i];
					feq[i][z][y][x] = ff[i][z][y][x];
				}
			}
	float* h_data11 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_obst = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_data = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_ux = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_uy = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_uz = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_p = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_rho = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_water = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_oil = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_pp = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_u_x = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_u_y = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_u_z = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_ff = (float*)calloc(npop * NZ * NY * NX, sizeof(float));
	float* h_feq = (float*)calloc(npop * NZ * NY * NX, sizeof(float));
	float* h_tempt = (float*)calloc(npop * NZ * NY * NX, sizeof(float));
	float* h_Fx = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fy = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fz = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fx1 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fy1 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fz1 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fx2 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fy2 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fz2 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Sx = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Sy = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Sz = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_psx = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fmx1 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fmy1 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fmz1 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fmx2 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fmy2 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_Fmz2 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_F1sx = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_F1sy = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_F1sz = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_smx_pre1 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_smy_pre1 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_smz_pre1 = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_psxpre = (float*)calloc(npop * NZ * NY * NX, sizeof(float));
	float* h_spre = (float*)calloc(npop * NZ * NY * NX, sizeof(float));
	float* h_cu = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_fftemp = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_psxpretemp = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_spretemp = (float*)calloc(NZ * NY * NX, sizeof(float));
	float* h_rhol = (float*)calloc(NZ * NY * NX, sizeof(float));

	for (int z = 0; z < NZ; z++)
		for (int y = 0; y < NY; y++)
			for (int x = 0; x < NX; x++)
			{
				h_obst[z * NX * NY + y * NX + x] = obst[z][y][x];
				h_rho[z * NX * NY + y * NX + x] = rho[z][y][x];
				h_data11[z * NX * NY + y * NX + x] = data11[z][y][x];
				h_data[z * NX * NY + y * NX + x] = data[z][y][x];

				for (int i = 0; i < npop; i++)
				{
					h_ff[i * NX * NY * NZ + z * NX * NY + y * NX + x] = ff[i][z][y][x];
					h_feq[i * NX * NY * NZ + z * NX * NY + y * NX + x] = feq[i][z][y][x];

				}
			}



	float* d_data11;
	float* d_obst;
	float* d_data;
	float* d_ux;
	float* d_uy;
	float* d_uz;
	float* d_p;
	float* d_rho;
	float* d_water;
	float* d_oil;
	float* d_pp;
	float* d_u_x;
	float* d_u_y;
	float* d_u_z;
	float* d_Fx;
	float* d_Fy;
	float* d_Fz;
	float* d_Fx1;
	float* d_Fy1;
	float* d_Fz1;
	float* d_Fx2;
	float* d_Fy2;
	float* d_Fz2;
	float* d_Sx;
	float* d_Sy;
	float* d_Sz;
	float* d_psx;
	float* d_Fmx1;
	float* d_Fmy1;
	float* d_Fmz1;
	float* d_Fmx2;
	float* d_Fmy2;
	float* d_Fmz2;
	float* d_F1sx;
	float* d_F1sy;
	float* d_F1sz;
	float* d_cu;
	float* d_fftemp;
	float* d_psxpretemp;
	float* d_spretemp;
	float* d_smx_pre1;
	float* d_smy_pre1;
	float* d_smz_pre1;
	float* d_rhol;

	float* d_ff;
	float* d_feq;
	float* d_tempt;
	float* d_psxpre;
	float* d_spre;

	int* d_nonzero_indices_x;
	int* d_nonzero_indices_y;
	int* d_nonzero_indices_z;

	cudaMalloc(&d_data11, mem_size);
	cudaMalloc(&d_obst, mem_size);
	cudaMalloc(&d_data, mem_size);
	cudaMalloc(&d_ux, mem_size);
	cudaMalloc(&d_uy, mem_size);
	cudaMalloc(&d_uz, mem_size);
	cudaMalloc(&d_p, mem_size);
	cudaMalloc(&d_rho, mem_size);
	cudaMalloc(&d_water, mem_size);
	cudaMalloc(&d_oil, mem_size);
	cudaMalloc(&d_pp, mem_size);
	cudaMalloc(&d_u_x, mem_size);
	cudaMalloc(&d_u_y, mem_size);
	cudaMalloc(&d_u_z, mem_size);
	cudaMalloc(&d_Fx, mem_size);
	cudaMalloc(&d_Fy, mem_size);
	cudaMalloc(&d_Fz, mem_size);
	cudaMalloc(&d_Fx1, mem_size);
	cudaMalloc(&d_Fy1, mem_size);
	cudaMalloc(&d_Fz1, mem_size);
	cudaMalloc(&d_Fx2, mem_size);
	cudaMalloc(&d_Fy2, mem_size);
	cudaMalloc(&d_Fz2, mem_size);
	cudaMalloc(&d_Sx, mem_size);
	cudaMalloc(&d_Sy, mem_size);
	cudaMalloc(&d_Sz, mem_size);
	cudaMalloc(&d_psx, mem_size);
	cudaMalloc(&d_Fmx1, mem_size);
	cudaMalloc(&d_Fmy1, mem_size);
	cudaMalloc(&d_Fmz1, mem_size);
	cudaMalloc(&d_Fmx2, mem_size);
	cudaMalloc(&d_Fmy2, mem_size);
	cudaMalloc(&d_Fmz2, mem_size);
	cudaMalloc(&d_F1sx, mem_size);
	cudaMalloc(&d_F1sy, mem_size);
	cudaMalloc(&d_F1sz, mem_size);
	cudaMalloc(&d_cu, mem_size);
	cudaMalloc(&d_fftemp, mem_size);
	cudaMalloc(&d_psxpretemp, mem_size);
	cudaMalloc(&d_spretemp, mem_size);
	cudaMalloc(&d_smx_pre1, mem_size);
	cudaMalloc(&d_smy_pre1, mem_size);
	cudaMalloc(&d_smz_pre1, mem_size);
	cudaMalloc(&d_rhol, mem_size);

	cudaMalloc(&d_ff, mem_size4d);
	cudaMalloc(&d_feq, mem_size4d);
	cudaMalloc(&d_tempt, mem_size4d);
	cudaMalloc(&d_psxpre, mem_size4d);
	cudaMalloc(&d_spre, mem_size4d);

	size_t mem_sizeindex = Boundary * sizeof(int);
	cudaMalloc(&d_nonzero_indices_x, mem_sizeindex);
	cudaMalloc(&d_nonzero_indices_y, mem_sizeindex);
	cudaMalloc(&d_nonzero_indices_z, mem_sizeindex);




	cudaMemcpy(d_data11, h_data11, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_obst, h_obst, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_data, h_data, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ux, h_ux, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uy, h_uy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uz, h_uz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, h_p, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho, h_rho, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_water, h_water, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_oil, h_oil, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pp, h_pp, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_x, h_u_x, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_y, h_u_y, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_z, h_u_z, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fx, h_Fx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fy, h_Fy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fz, h_Fz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fx1, h_Fx1, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fy1, h_Fy1, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fz1, h_Fz1, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fx2, h_Fx2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fy2, h_Fy2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fz2, h_Fz2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Sx, h_Sx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Sy, h_Sy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Sz, h_Sz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_psx, h_psx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fmx1, h_Fmx1, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fmy1, h_Fmy1, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fmz1, h_Fmz1, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fmx2, h_Fmx2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fmy2, h_Fmy2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fmz2, h_Fmz2, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F1sx, h_F1sx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F1sy, h_F1sy, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F1sz, h_F1sz, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cu, h_cu, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fftemp, h_fftemp, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_psxpretemp, h_psxpretemp, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_spretemp, h_spretemp, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_smx_pre1, h_smx_pre1, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_smy_pre1, h_smy_pre1, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_smz_pre1, h_smz_pre1, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rhol, h_rhol, mem_size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_ff, h_ff, mem_size4d, cudaMemcpyHostToDevice);
	cudaMemcpy(d_feq, h_feq, mem_size4d, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tempt, h_tempt, mem_size4d, cudaMemcpyHostToDevice);
	cudaMemcpy(d_psxpre, h_psxpre, mem_size4d, cudaMemcpyHostToDevice);
	cudaMemcpy(d_spre, h_spre, mem_size4d, cudaMemcpyHostToDevice);

	cudaMemcpy(d_nonzero_indices_x, h_nonzero_indicesX, mem_sizeindex, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nonzero_indices_y, h_nonzero_indicesY, mem_sizeindex, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nonzero_indices_z, h_nonzero_indicesZ, mem_sizeindex, cudaMemcpyHostToDevice);

	free_space3d(ux, NZ, NY);
	free_space3d(uy, NZ, NY);
	free_space3d(uz, NZ, NY);
	free_space3d(p, NZ, NY);
	free_space3d(rho, NZ, NY);
	free_space3d(water, NZ, NY);
	free_space3d(oil, NZ, NY);
	free_space3d(pp, NZ, NY);
	free_space3d(u_x, NZ, NY);
	free_space3d(u_y, NZ, NY);
	free_space3d(u_z, NZ, NY);

	free_space4d(ff, npop, NZ, NY);
	free_space4d(feq, npop, NZ, NY);

	free(h_ff);
	free(h_data11);
	free(h_obst);
	free(h_data);
	free(h_ux);
	free(h_uy);
	free(h_uz);
	free(h_p);
	free(h_water);
	free(h_oil);
	free(h_pp);
	free(h_u_x);
	free(h_u_y);
	free(h_u_z);
	free(h_feq);
	free(h_tempt);
	free(h_Fx);
	free(h_Fy);
	free(h_Fz);
	free(h_Fx1);
	free(h_Fy1);
	free(h_Fz1);
	free(h_Fx2);
	free(h_Fy2);
	free(h_Fz2);
	free(h_Sx);
	free(h_Sy);
	free(h_Sz);
	free(h_psx);
	free(h_Fmx1);
	free(h_Fmy1);
	free(h_Fmz1);
	free(h_Fmx2);
	free(h_Fmy2);
	free(h_Fmz2);
	free(h_F1sx);
	free(h_F1sy);
	free(h_F1sz);
	free(h_smx_pre1);
	free(h_smy_pre1);
	free(h_smz_pre1);
	free(h_psxpre);
	free(h_spre);
	free(h_cu);
	free(h_fftemp);
	free(h_psxpretemp);
	free(h_spretemp);
	free(h_rhol);

	free(h_nonzero_indicesX);
	free(h_nonzero_indicesY);
	free(h_nonzero_indicesZ);


	//开始迭代
	clock_t start, finish;
	start = clock();
	for (int l = 0; l < t_max; l++)
	{
		if (l % 10 == 0) {
			printf("time = %d\n", l);
		}

		CalTempt << < blocksPerGrid, threadsPerBlock >> > (d_tempt, d_ff, Boundary, d_nonzero_indices_x, d_nonzero_indices_y, d_nonzero_indices_z);
		cudaDeviceSynchronize();

		CalBGK << < Blocksize, Gridsize >> > (d_u_x, d_u_y, d_u_z, d_ux, d_uy, d_uz, d_F1sx, d_F1sy, d_F1sz, d_Fmx1, d_Fmy1, d_Fmz1, d_Fmx2, d_Fmy2, d_Fmz2, d_rho, d_cu, d_feq, d_ff, tau);
		cudaDeviceSynchronize();


		UpdateBoundary << < blocksPerGrid, threadsPerBlock >> > (d_ff, d_tempt, Boundary, d_nonzero_indices_x, d_nonzero_indices_y, d_nonzero_indices_z);
		cudaDeviceSynchronize();


		for (int i = 0; i < npop; i++)
		{
			UpdateMidff << < Blocksize, Gridsize >> > (d_fftemp, d_ff, i);
			cudaDeviceSynchronize();

			Updateff << < Blocksize, Gridsize >> > (d_fftemp, d_ff, i);
			cudaDeviceSynchronize();

		}
		//测试
		/*if (l == 1)
		{
			cudaMemcpy(h_ff, d_ff, mem_size4d, cudaMemcpyDeviceToHost);
			for (int z = 0; z < NZ; z++)
			{
				for (int y = 0; y < NY; y++)
				{
					for (int x = 0; x < NX; x++)
					{
						for (int i = 0; i < npop; i++)
						{
							test4d[i][z][y][x] = h_ff[i * NX * NY * NZ + z * NX * NY + y * NX + x];
						}
					}
				}
			}
			write_4d_data_to_file(test4d, npop, NZ, NY, NX);

			cudaMemcpy(h_rho, d_rho, mem_size, cudaMemcpyDeviceToHost);
			for (int z = 0; z < NZ; z++)
			{
				for (int y = 0; y < NY; y++)
				{
					for (int x = 0; x < NX; x++)
					{

						test[z][y][x] = h_rho[z * NX * NY + y * NX + x];

					}
				}
			}
			write_data_to_file(test, NZ, NY, NX);

		}*/

		Updaterho << < Blocksize, Gridsize >> > (d_rho, d_ff);
		cudaDeviceSynchronize();

		UpdateUPot << < Blocksize, Gridsize >> > (d_ux, d_uy, d_uz, d_ff, d_pp, d_rho, G1, d_psx);
		cudaDeviceSynchronize();


		PotBoundary << < blocksPerGrid, threadsPerBlock >> > (d_psx, Boundary, d_nonzero_indices_x, d_nonzero_indices_y, d_nonzero_indices_z);
		cudaDeviceSynchronize();

		CalP << < Blocksize, Gridsize >> > (d_p, d_rho, G1, d_psx);
		cudaDeviceSynchronize();


		for (int i = 0; i < npop; i++)
		{
			UpdateMidPot << < Blocksize, Gridsize >> > (d_psxpretemp, d_psx, d_spretemp, d_obst, i);
			cudaDeviceSynchronize();

			UpdatePot << < Blocksize, Gridsize >> > (d_psxpre, d_psxpretemp, d_spre, d_spretemp, i);
			cudaDeviceSynchronize();
		}

		CalF << < Blocksize, Gridsize >> > (d_smx_pre1, d_smy_pre1, d_smz_pre1, d_Fmx1, d_Fmy1, d_Fmz1, d_Fmx2, d_Fmy2, d_Fmz2, d_spre, d_psxpre);
		cudaDeviceSynchronize();

		BoundaryF << < blocksPerGrid, threadsPerBlock >> > (d_Fmx1, d_Fmy1, d_Fmz1, d_Fmx2, d_Fmy2, d_Fmz2, d_smx_pre1, d_smy_pre1, d_smz_pre1, d_nonzero_indices_x, d_nonzero_indices_y, d_nonzero_indices_z, Boundary);
		cudaDeviceSynchronize();

		UpdateF << < Blocksize, Gridsize >> > (d_F1sx, d_F1sy, d_F1sz, d_Fmx1, d_Fmy1, d_Fmz1, d_Fmx2, d_Fmy2, d_Fmz2, d_psx, d_smx_pre1, d_smy_pre1, d_smz_pre1, G1, G2, B);
		cudaDeviceSynchronize();

		


		//时间迭代结束
	}
	cudaMemcpy(h_rho, d_rho, mem_size, cudaMemcpyDeviceToHost);
	// 计算 rho 的总和
	double q = 0.0;
	for (int k = 0; k < NZ; k++) {
		for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				q += h_rho[k * NX * NY + j * NX + i];
			}
		}
	}
	printf("Iteration %f: Total rho = %f\n", t_max, q);

	float*** rhol;
	rhol = space3d(NZ, NY, NX);
	for (int k = 0; k < NZ; k++)
	{
		for (int j = 0; j < NY; j++)
		{
			for (int i = 0; i < NX; i++)
			{
				rhol[k][j][i] = h_rho[k * NX * NY + j * NX + i];
			}
		}
	}

	for (int bod = 0; bod < Boundary; bod++)
	{
		rhol[nonzero_indices[bod].z][nonzero_indices[bod].y][nonzero_indices[bod].x] = 0.01;

	}

	int s = 0;

	for (int k = 0; k < NZ; k++)
	{
		for (int j = 0; j < NY; j++)
		{
			for (int i = 0; i < NX; i++)
			{
				if (rhol[k][j][i] < ((rho_h + rho_l) / 2.0) && obst[k][j][i] != 1)
				{
					s++;
					data11[k][j][i] = 2;
					printf("1\n");
				}
				else
				{
					data11[k][j][i] = 3;

				}
			}
		}
	}
	float sr = s / (NX * NY * NZ - Boundary); // 当前饱和度值




	finish = clock();
	printf("%f seconds\n", (float)(finish - start) / CLOCKS_PER_SEC);
	float*** rock;
	rock = space3d(NZ, NY, NX);

	for (int iz = 0; iz < NZ; iz++)
	{
		for (int iy = 0; iy < NY; iy++)
		{
			for (int ix = 0; ix < NX; ix++)
			{
				rock[iz][iy][ix] = data11[iz][iy][ix];

				if (rock[iz][iy][ix] == 2)
					data[iz][iy][ix] = 7; //7是水  1是油

			}
		}
	}


	write_data_to_file(data, NZ, NY, NX);



	for (int i = 1; i <= 7; i++) {
		int t = i;
		int num = countElements(rock, NZ, NY, NX, t);
		int totalElements = NZ * NY * NX;
		float P = (float)num / totalElements;
		printf("组分 %d ---- %f\n", i, P);
	}

	int numw = countElements(rock, NZ, NY, NX, 2);
	int numo = countElements(rock, NZ, NY, NX, 3);
	printf("numw = %d, numo = %d\n", numw, numo);
	float sw = 100 * numw / (numw + numo);
	printf("含水饱和度 ---- %f\n", sw);



	//free
	free(h_rho);	
	free_space3d(obst, NZ, NY);
	free_space3d(rhol, NZ, NY);
	free_space3d(data11, NZ, NY);
	free_space3d(rock, NZ, NY);
	free_space3d(data, NZ, NY);


	cudaFree(d_data11);
	cudaFree(d_obst);
	cudaFree(d_data);
	cudaFree(d_ux);
	cudaFree(d_uy);
	cudaFree(d_uz);
	cudaFree(d_p);
	cudaFree(d_rho);
	cudaFree(d_water);
	cudaFree(d_oil);
	cudaFree(d_pp);
	cudaFree(d_u_x);
	cudaFree(d_u_y);
	cudaFree(d_u_z);
	cudaFree(d_Fx);
	cudaFree(d_Fy);
	cudaFree(d_Fz);
	cudaFree(d_Fx1);
	cudaFree(d_Fy1);
	cudaFree(d_Fz1);
	cudaFree(d_Fx2);
	cudaFree(d_Fy2);
	cudaFree(d_Fz2);
	cudaFree(d_Sx);
	cudaFree(d_Sy);
	cudaFree(d_Sz);
	cudaFree(d_psx);
	cudaFree(d_Fmx1);
	cudaFree(d_Fmy1);
	cudaFree(d_Fmz1);
	cudaFree(d_Fmx2);
	cudaFree(d_Fmy2);
	cudaFree(d_Fmz2);
	cudaFree(d_F1sx);
	cudaFree(d_F1sy);
	cudaFree(d_F1sz);
	cudaFree(d_cu);
	cudaFree(d_fftemp);
	cudaFree(d_psxpretemp);
	cudaFree(d_spretemp);
	cudaFree(d_smx_pre1);
	cudaFree(d_smy_pre1);
	cudaFree(d_smz_pre1);
	cudaFree(d_rhol);

	cudaFree(d_ff);
	cudaFree(d_feq);
	cudaFree(d_tempt);
	cudaFree(d_psxpre);
	cudaFree(d_spre);

	cudaFree(d_nonzero_indices_x);
	cudaFree(d_nonzero_indices_y);
	cudaFree(d_nonzero_indices_z);
	//****************
	//测试用数组
	free_space3d(test, NZ, NY);
	free_space4d(test4d, npop, NZ, NY);
	//****************

	printf("\Press any key to exit program...");
	return 0;
}

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

float**** space4d(int n1, int n2, int n3, int n4)
{
	float**** a;
	int i, j, k;

	a = (float****)malloc(sizeof(float***) * n1);
	for (i = 0; i < n1; i++)
	{
		a[i] = (float***)malloc(sizeof(float**) * n2);
		for (j = 0; j < n2; j++)
		{
			a[i][j] = (float**)malloc(sizeof(float*) * n3);
			for (k = 0; k < n3; k++)
			{
				a[i][j][k] = (float*)malloc(sizeof(float) * n4);
			}
		}
	}

	for (i = 0; i < n1; i++)
		for (j = 0; j < n2; j++)
			for (k = 0; k < n3; k++)
				for (int l = 0; l < n4; l++)
				{
					a[i][j][k][l] = 0.0f;
				}

	return a;
}

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

void free_space4d(float**** a, int nr, int ny, int nc)
{
	int i, j, k;
	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < ny; j++)
		{
			for (k = 0; k < nc; k++)
			{
				free(a[i][j][k]);
			}
			free(a[i][j]);
		}
		free(a[i]);
	}
	free(a);
}

Index3D* find_nonzero_elements(float*** array, int* Boundray)
{
	int max_elements = NZ * NY * NX;
	Index3D* indices = (Index3D*)malloc(max_elements * sizeof(Index3D));
	int index = 0;
	for (int z = 0; z < NZ; z++)
		for (int y = 0; y < NY; y++)
			for (int x = 0; x < NX; x++)
			{
				if (array[z][y][x] != 0.0f)
				{
					indices[index].z = z;
					indices[index].y = y;
					indices[index].x = x;
					index++;
				}
			}
	indices = (Index3D*)realloc(indices, index * sizeof(Index3D));
	*Boundray = index;
	return indices;
}

void zero3dArray(float*** array, int nr, int ny, int nc)
{
	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < ny; j++) {
			memset(array[i][j], 0, nc * sizeof(float));
		}
	}
}

// 计算 rock 中等于 t 的元素数量
int countElements(float*** rock, int lz, int ly, int lx, float t)
{
	int count = 0;
	for (int i = 0; i < lz; i++) {
		for (int j = 0; j < ly; j++) {
			for (int k = 0; k < lx; k++) {
				if (rock[i][j][k] == t) {
					count++;
				}
			}
		}
	}
	return count;
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

void wfile1d(char filename[], float* data, int nz, int ny, int nx)
{
	int i;
	FILE* fp = fopen(filename, "wt");
	if (fp == NULL)
	{
		perror("Error opening file");
		return;
	}

	for (i = 0; i < nz * ny * nx; i++)
	{
		fprintf(fp, "%e ", data[i]);
		if ((i + 1) % nx == 0)
		{
			fprintf(fp, "\n");
		}
	}

	fclose(fp);
}

// 将数据写入文件
void write_data_to_file(float*** data, int nz, int ny, int nx)
{
	FILE* file = fopen("data.txt", "w");
	if (file == NULL) {
		printf("Error opening file!\n");
		return;
	}

	for (int i = 0; i < nz; i++) {
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nx; k++) {
				fprintf(file, "%d %d %d %f\n", i, j, k, data[i][j][k]);
			}
		}
	}

	fclose(file);
}

// 将四维数据写入文件
void write_4d_data_to_file(float**** data, int nh, int nz, int ny, int nx)
{
	FILE* file = fopen("data4D.txt", "w");
	if (file == NULL) {
		printf("Error opening file!\n");
		return;
	}

	for (int h = 0; h < nh; h++) {
		for (int i = 0; i < nz; i++) {
			for (int j = 0; j < ny; j++) {
				for (int k = 0; k < nx; k++) {
					fprintf(file, "%d %d %d %d %f\n", h, i, j, k, data[h][i][j][k]);
				}
			}
		}
	}

	fclose(file);
}
