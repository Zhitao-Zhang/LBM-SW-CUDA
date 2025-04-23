//----------------------------------------------------------------------------------------
// 测试并行
//---------------------------------------------------------------------------
// 运行时间3000有频散
#pragma hdrstop
//---------------------------------------------------------------------------
// #pragma argsused
// #define SW9
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <athread.h>
#include <stdio.h>
#include <mpi.h>
// #define LWPF_UNITS U(test)
//  定义xyz方向的进程数
#define PX 7
#define PY 7
#define PZ 6
#define tran 1 // MPI过渡带宽度
#define PI 3.1415926535
#define e 2.718281828

// lwpf
// #define LWPF_UNITS U(test)
// #include "lwpf3/lwpf.h"
// #define sizeSlavez 25
float *rho_ext, *txx, *tyy, *tzz, *txy, *txz, *tyz, *dxi, *dxi2, *dyj, *dyj2, *dzk, *dzk2, *vx, *vy, *vz, *mu_ext, *lamda2u_ext, *lamda_ext, *muxy, *muxz, *muyz, *edxi, *edxi2, *edyj, *edyj2, *edzk, *edzk2;
float *pmlxSxx, *pmlySxy, *pmlzSxz, *pmlxSxy, *pmlySyy, *pmlzSyz, *pmlxSxz, *pmlySyz, *pmlzSzz;
float *DxSxxPre, *DySxyPre, *DzSxzPre, *DxSxyPre, *DySyyPre, *DzSyzPre, *DxSxzPre, *DySyzPre, *DzSzzPre;
float *pmlxVx, *pmlyVy, *pmlzVz, *pmlzVx, *pmlxVz, *pmlyVx, *pmlxVy, *pmlzVy, *pmlyVz;
float *DxVxPre, *DyVyPre, *DzVzPre, *DyVxPre, *DxVyPre, *DzVxPre, *DxVzPre, *DzVyPre, *DyVzPre;
float H, DT;
int nxt, nyt, nzt, it;
int myid;
// 每个进程的前后左右上下
int back, forward, left, right;
extern SLAVE_FUN(funVelocity)();
extern SLAVE_FUN(funStress)();
float **space2d(int nr, int nc);
float ***space3d(int nr, int ny, int nc);
void free_space2d(float **a, int nr);
void free_space3d(float ***b, int nr, int ny);
void wfile(char filename[], float **data, int nr, int nc);
void wfile3d(char filename[], float ***data, int nr, int ny, int nc);
void wfile3dy(char filename[], float ***data, int nr, int ny, int nc);
void readCompressedBinary(float *aa, int totalSize, const char *filename);
void wfile1d(char filename[], float *data, int nr);
void create_model(float *vp, float *vs, float *rho, float *vf, float *rhof, float *lamda, float *lamda2u, float *mu, int nr, int ny, int nc, int myid, int coord[], float *elastic_pro, float pressure);
int main(int argc, char **argv)
{
    // 给定参数
    int nx = 500; // x方向网格点数%%%%%
    int ny = 500; // y方向网格点数%%%%%
    int nzz = 500;
    int nz = 540; // z方向网格点数%%%%
    int NP = 30;  // pml层网格点数
    int i = 0;
    int sx = nx / 2 + NP; // 震源坐标点号
    int sy = ny / 2 + NP;
    int sz = 5 + NP;
    int NX_ext = nx + 2 * NP;
    int NY_ext = ny + 2 * NP;
    int NZ_ext = nz + 2 * NP;
    int NT = 4100;           // 空间步长5%%%
    H = 1.0 * pow(10.0, -5); // z轴步长
    int waveType = 0;// 0模拟 P 波  1模拟x激发s波 2模拟y激发s波
    float RC = 1.0 * pow(10.0, -6);
    float DP = NP * H;
    DT = 5.0 * pow(10.0, -10);     // 时间步长%%%%
    double F0 = 10 * pow(10.0, 6); // 震源主频,主频太高会造成频散
    float T0 = 1.2 / F0;
    double Vpmax = 4270.0; // 模型最大纵波速度,用于稳定性计算
    double Vpmin = 1500.0; // 模型最小纵波速度,用于控制数值频散
    double Vsmax = 2650.0;
    double Vsmin = 1500.0;
    float *vs;      // 初始模型横波速度
    float *vp;      // 初始模型纵波速度
    float *vf;      // 孔隙流体速度
    float *rhof;    // 孔隙流体密度
    float **sis_vx; // 记录vz
    float **sis_vy; // 记录vz
    float **sis_vz; // 记录vz
    float ***txx3d; // 记录txx
    // 定义围压
    // float pressure[12] = {0.1, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400};
    float pressure[1] = {250};
    float tt, x, y, z, xoleft, xoright, yoleft, yoright, zoleft, zoright, d0 = 0.0f;
    float v0, y5, y6, y7, y8, y9, y10, y11, y12, y13;
    float xx1, xy1, xz1, yx2, yy2, yz2, zx3, zy3, zz3, best_dt;
    int ix, iy, iz;
    // input参数
    //  输入模型
    // float ***elastic_pre;
    // elastic_pre = space3d(nzz, ny, nx);
    printf("input begin");
    // // 打开文件
    // FILE *file11;
    // file11 = fopen("./63.060177_1.100000.txt", "r");
    // if (file11 == NULL)
    // {
    //     fprintf(stderr, "无法打开文件\n");
    //     return 1;
    // }
    // // 读取文件内容并存储到数组中
    // int INPUTnz, INPUTny, INPUTnx;
    // float INPUTvalue;
    // while (fscanf(file11, "%d %d %d %f", &INPUTnz, &INPUTny, &INPUTnx, &INPUTvalue) != EOF)
    // {
    //     if (INPUTnz >= 0 && INPUTnz < nzz && INPUTny >= 0 && INPUTny < ny && INPUTnx >= 0 && INPUTnx < nx)
    //     {
    //         elastic_pre[INPUTnz][INPUTny][INPUTnx] = INPUTvalue;
    //     }
    //     else
    //     {
    //         fprintf(stderr, "索引超出范围: nz=%d, ny=%d, nx=%d\n", INPUTnz, INPUTny, INPUTnx);
    //     }
    // }
    // // 关闭文件
    // fclose(file11);
    // 读取二进制文件
    int totalSize = nx * ny * nzz;
    float *elastic_pre_1D;
    elastic_pre_1D = (float *)malloc(totalSize * sizeof(float));
    memset(elastic_pre_1D, 0, totalSize * sizeof(float));
    char *filename00 = "./63.060177_1.100000BIN.bin";
    FILE *file00 = fopen(filename00, "rb");
    if (file00 == NULL)
    {
        fprintf(stderr, "no open file0\n");
        return 1;
    }
    // 调用函数
    readCompressedBinary(elastic_pre_1D, totalSize, filename00);
    // 打印数组的某些值以验证
    printf("elastic_pre[30][75][65] = %f\n", elastic_pre_1D[30 * nx * ny + 75 * nx + 65]);
    printf("elastic_pre[50][50][50] = %f\n", elastic_pre_1D[50 * nx * ny + 50 * nx + 50]);
    printf("elastic_pre[20][50][50] = %f\n", elastic_pre_1D[20 * nx * ny + 50 * nx + 50]);
    // 扩展input
    float *elastic_pre_all;
    elastic_pre_all = (float *)malloc(NZ_ext * NY_ext * NX_ext * sizeof(float));
    memset(elastic_pre_all, 0, NZ_ext * NY_ext * NX_ext * sizeof(float));
    int Zdif = (NZ_ext - nzz) * 0.5;
    int Ydif = (NY_ext - ny) * 0.5;
    int Xdif = (NX_ext - nx) * 0.5;
    int averageType = 8;
    for (int k = 0; k < NZ_ext; k++)
        for (int j = 0; j < NY_ext; j++)
            for (int i = 0; i < NX_ext; i++)
            {
                if (k >= Zdif && k < (nzz + Zdif) && j >= Ydif && j < (ny + Ydif) && i >= Xdif && i < (nx + Xdif))
                {
                    elastic_pre_all[k * NX_ext * NY_ext + j * NX_ext + i] = elastic_pre_1D[(k - Zdif) * nx * ny + (j - Ydif) * nx + (i - Xdif)];
                }
                else
                {
                    elastic_pre_all[k * NX_ext * NY_ext + j * NX_ext + i] = averageType;
                }
            }
    printf("elastic_pre_all[30][75][65] = %f\n", elastic_pre_all[30 * NX_ext * NY_ext + 75 * NX_ext + 65]);
    printf("elastic_pre_all[70][50][50] = %f\n", elastic_pre_all[70 * NX_ext * NY_ext + 50 * NX_ext + 50]);
    printf("elastic_pre_all[20][50][50] = %f\n", elastic_pre_all[20 * NX_ext * NY_ext + 50 * NX_ext + 50]);
    // free_space3d(elastic_pre, nzz, ny);
    free(elastic_pre_1D);
    // input结束
    printf("input finish\n");
    //********************************************************************************************************
    // mpi初始化
    MPI_Comm MCW, MC1;
    int numersize;
    MPI_Init(&argc, &argv);
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numersize);
    MPI_Comm_dup(MPI_COMM_WORLD, &MCW);
    MPI_Request request[4];
    // 定义维度进程 周期 坐标 排序 错误信息
    int back, forward, left, right, up, down;
    int dim[3], period[3], coord[3], reorder, err;
    dim[0] = PX;
    dim[1] = PY;
    dim[2] = PZ;
    reorder = 1;
    // x y z 都是非周期环面
    period[0] = 0;
    period[1] = 0;
    period[2] = 0;
    // xyz方向每个进程中有多少个网格
    nxt = NX_ext / PX;
    nyt = NY_ext / PY;
    nzt = NZ_ext / PZ;
    printf("nxt = %d  nyt = %d   nzt = %d\n", nxt, nyt, nzt);
    // 计算点源所在的进程 向上取整数
    int x0pro, y0pro, z0pro;
    x0pro = ((sx + nxt - 1) / nxt) - 1; // - 1是因为索引从0开始
    y0pro = ((sy + nyt - 1) / nyt) - 1;
    z0pro = ((sz + nzt - 1) / nzt) - 1;
    // 计算点源所在进程的索引
    int sxnew, synew, sznew;
    sxnew = (sx % nxt);
    synew = (sy % nyt);
    sznew = (sz % nzt);
    // 创建笛卡尔拓扑
    err = MPI_Cart_create(MCW, 3, dim, period, reorder, &MC1);
    // 得到拓扑结构的标识 之后能通过标识来吧对应的数据放入对应的进程中  eg. 之后进行数据通信的时候，对本进程想要往左边进程传递数据 只需要调用left
    err = MPI_Cart_shift(MC1, 0, 1, &left, &right);
    err = MPI_Cart_shift(MC1, 1, 1, &back, &forward);
    err = MPI_Cart_shift(MC1, 2, 1, &down, &up);
    err = MPI_Cart_coords(MC1, myid, 3, coord);
    printf("rank = %d\t  coord[0] = %d\t  coord[1] = %d\t   coord[2] = %d\t   left = %d\t  right = %d\t  back = %d\t  forward = %d\t  up = %d\t down = %d\n", myid, coord[0], coord[1], coord[2], left, right, back, forward, up, down);
    vs = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    vp = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    rho_ext = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    vf = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    rhof = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    mu_ext = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    lamda_ext = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    lamda2u_ext = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    int coordx = coord[0];
    int coordy = coord[1];
    int coordz = coord[2];
    // 数据扩充，加上吸收边界
    dxi = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    dxi2 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    dyj = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    dyj2 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    dzk = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    dzk2 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(dxi, 0, nzt * nyt * nxt * sizeof(float));
    memset(dxi2, 0, nzt * nyt * nxt * sizeof(float));
    memset(dyj, 0, nzt * nyt * nxt * sizeof(float));
    memset(dyj2, 0, nzt * nyt * nxt * sizeof(float));
    memset(dzk, 0, nzt * nyt * nxt * sizeof(float));
    memset(dzk2, 0, nzt * nyt * nxt * sizeof(float));
    muxy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    muxz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    muyz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(muxy, 0, nzt * nyt * nxt * sizeof(float));
    memset(muxz, 0, nzt * nyt * nxt * sizeof(float));
    memset(muyz, 0, nzt * nyt * nxt * sizeof(float));
    edxi = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    edxi2 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    edyj = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    edyj2 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    edzk = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    edzk2 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(edxi, 0, nzt * nyt * nxt * sizeof(float));
    memset(edxi2, 0, nzt * nyt * nxt * sizeof(float));
    memset(edyj, 0, nzt * nyt * nxt * sizeof(float));
    memset(edyj2, 0, nzt * nyt * nxt * sizeof(float));
    memset(edzk, 0, nzt * nyt * nxt * sizeof(float));
    memset(edzk2, 0, nzt * nyt * nxt * sizeof(float));
    // 开辟空间
    txx3d = space3d(nzt, nyt, nxt);
    sis_vx = space2d(nzt, NT);
    sis_vy = space2d(nzt, NT);
    sis_vz = space2d(nzt, NT);
    //(xoleft,xoright)模型的物理边界—不算PML
    xoleft = DP;
    xoright = (nxt - 1) * H - DP;
    yoleft = DP;
    yoright = (nyt - 1) * H - DP;
    zoleft = DP;
    zoright = (nzt - 1) * H - DP;
    // 检查左边界
    if ((left == -1) && (right >= 0) && (back >= 0) && (forward >= 0) && (up >= 0) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    if (x >= 0 && x < xoleft)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((xoleft - x) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
                    }
                }
    }
    if ((left == -1) && (right >= 0) && (back >= 0) && (forward >= 0) && (up == -1) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= 0 && x < xoleft)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((xoleft - x) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dzk[pmlindex] = d0 * pow(((z - zoright) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
                    }
                }
    }
    if ((left == -1) && (right >= 0) && (back >= 0) && (forward >= 0) && (up >= 0) && (down == -1))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= 0 && x < xoleft)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((xoleft - x) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dzk[pmlindex] = d0 * pow(((zoleft - z) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
                    }
                }
    }
    //======================================================
    if ((left == -1) && (right >= 0) && (back == -1) && (forward >= 0) && (up >= 0) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= 0 && x < xoleft)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((xoleft - x) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dyj[pmlindex] = d0 * pow(((yoleft - y) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
                    }
                }
    }
    if ((left == -1) && (right >= 0) && (back == -1) && (forward >= 0) && (up == -1) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= 0 && x < xoleft)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((xoleft - x) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dyj[pmlindex] = d0 * pow(((yoleft - y) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dzk[pmlindex] = d0 * pow(((z - zoright) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
                    }
                }
    }
    if ((left == -1) && (right >= 0) && (back == -1) && (forward >= 0) && (up >= 0) && (down == -1))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= 0 && x < xoleft)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((xoleft - x) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dyj[pmlindex] = d0 * pow(((yoleft - y) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dzk[pmlindex] = d0 * pow(((zoleft - z) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
                    }
                }
    }
    //==================================================================
    if ((left == -1) && (right >= 0) && (back >= 0) && (forward == -1) && (up >= 0) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= 0 && x < xoleft)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((xoleft - x) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dyj[pmlindex] = d0 * pow(((y - yoright) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
                    }
                }
    }
    if ((left == -1) && (right >= 0) && (back >= 0) && (forward == -1) && (up == -1) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= 0 && x < xoleft)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((xoleft - x) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dyj[pmlindex] = d0 * pow(((y - yoright) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dzk[pmlindex] = d0 * pow(((z - zoright) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
                    }
                }
    }
    if ((left == -1) && (right >= 0) && (back >= 0) && (forward == -1) && (up >= 0) && (down == -1))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= 0 && x < xoleft)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((xoleft - x) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((xoleft - x - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dyj[pmlindex] = d0 * pow(((y - yoright) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dzk[pmlindex] = d0 * pow(((zoleft - z) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
                    }
                }
    }
    // 检查右边界
    if ((left >= 0) && (right == -1) && (back >= 0) && (forward >= 0) && (up >= 0) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= xoright && x < nxt * H)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((x - xoright) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
                    }
                }
    }
    if ((left >= 0) && (right == -1) && (back >= 0) && (forward >= 0) && (up == -1) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= xoright && x < nxt * H)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((x - xoright) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dzk[pmlindex] = d0 * pow(((z - zoright) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
                    }
                }
    }
    if ((left >= 0) && (right == -1) && (back >= 0) && (forward >= 0) && (up >= 0) && (down == -1))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= xoright && x < nxt * H)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((x - xoright) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dzk[pmlindex] = d0 * pow(((zoleft - z) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
                    }
                }
    }
    //======================================================
    if ((left >= 0) && (right == -1) && (back == -1) && (forward >= 0) && (up >= 0) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= xoright && x < nxt * H)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((x - xoright) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dyj[pmlindex] = d0 * pow(((yoleft - y) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
                    }
                }
    }
    if ((left >= 0) && (right == -1) && (back == -1) && (forward >= 0) && (up == -1) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= xoright && x < nxt * H)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((x - xoright) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dyj[pmlindex] = d0 * pow(((yoleft - y) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dzk[pmlindex] = d0 * pow(((z - zoright) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
                    }
                }
    }
    if ((left >= 0) && (right == -1) && (back == -1) && (forward >= 0) && (up >= 0) && (down == -1))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= xoright && x < nxt * H)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((x - xoright) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dyj[pmlindex] = d0 * pow(((yoleft - y) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dzk[pmlindex] = d0 * pow(((zoleft - z) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
                    }
                }
    }
    //===================================================
    if ((left >= 0) && (right == -1) && (back >= 0) && (forward == -1) && (up >= 0) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= xoright && x < nxt * H)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((x - xoright) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dyj[pmlindex] = d0 * pow(((y - yoright) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
                    }
                }
    }
    if ((left >= 0) && (right == -1) && (back >= 0) && (forward == -1) && (up == -1) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= xoright && x < nxt * H)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((x - xoright) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dyj[pmlindex] = d0 * pow(((y - yoright) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dzk[pmlindex] = d0 * pow(((z - zoright) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
                    }
                }
    }
    if ((left >= 0) && (right == -1) && (back >= 0) && (forward == -1) && (up >= 0) && (down == -1))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if (x >= xoright && x < nxt * H)
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dxi[pmlindex] = d0 * pow(((x - xoright) / DP), 2);
                        dxi2[pmlindex] = d0 * pow(((x + 0.5 * H - xoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dyj[pmlindex] = d0 * pow(((y - yoright) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dzk[pmlindex] = d0 * pow(((zoleft - z) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
                    }
                }
    }
    // 后
    if ((left >= 0) && (right >= 0) && (back == -1) && (forward >= 0) && (up >= 0) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dyj[pmlindex] = d0 * pow(((yoleft - y) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
                    }
                }
    }
    if ((left >= 0) && (right >= 0) && (back == -1) && (forward >= 0) && (up == -1) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dyj[pmlindex] = d0 * pow(((yoleft - y) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dzk[pmlindex] = d0 * pow(((z - zoright) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
                    }
                }
    }
    if ((left >= 0) && (right >= 0) && (back == -1) && (forward >= 0) && (up >= 0) && (down == -1))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < yoleft) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dyj[pmlindex] = d0 * pow(((yoleft - y) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((yoleft - y - 0.5 * H) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dzk[pmlindex] = d0 * pow(((zoleft - z) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
                    }
                }
    }
    // 前
    if ((left >= 0) && (right >= 0) && (back >= 0) && (forward == -1) && (up >= 0) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dyj[pmlindex] = d0 * pow(((y - yoright) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
                    }
                }
    }
    if ((left >= 0) && (right >= 0) && (back >= 0) && (forward == -1) && (up == -1) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dyj[pmlindex] = d0 * pow(((y - yoright) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dzk[pmlindex] = d0 * pow(((z - zoright) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
                    }
                }
    }
    if ((left >= 0) && (right >= 0) && (back >= 0) && (forward == -1) && (up >= 0) && (down == -1))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if ((x >= 0 && x < nxt * H) && (y >= yoright && y < nyt * H) && (z >= 0 && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dyj[pmlindex] = d0 * pow(((y - yoright) / DP), 2);
                        dyj2[pmlindex] = d0 * pow(((y + 0.5 * H - yoright) / DP), 2);
                    }
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dzk[pmlindex] = d0 * pow(((zoleft - z) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
                    }
                }
    }
    // 上
    if ((left >= 0) && (right >= 0) && (back >= 0) && (forward >= 0) && (up == -1) && (down >= 0))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= zoright && z < nzt * H))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        dzk[pmlindex] = d0 * pow(((z - zoright) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((z + 0.5 * H - zoright) / DP), 2);
                    }
                }
    }
    // 下
    if ((left >= 0) && (right >= 0) && (back >= 0) && (forward >= 0) && (up >= 0) && (down == -1))
    {
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    x = ix * H;
                    y = iy * H;
                    z = iz * H;
                    int pmlindex = iz * nxt * nyt + iy * nxt + ix;
                    if ((x >= 0 && x < nxt * H) && (y >= 0 && y < nyt * H) && (z >= 0 && z < zoleft))
                    {
                        v0 = 4000.0f;
                        d0 = 3.0 * v0 * log(1.0 / RC) / (2.0 * DP);
                        // d0=3.0*v0*(8.0/15.0-3.0/100.0*NP+NP*NP/1500.0)/H;
                        dzk[pmlindex] = d0 * pow(((zoleft - z) / DP), 2);
                        dzk2[pmlindex] = d0 * pow(((zoleft - z - 0.5 * H) / DP), 2);
                    }
                }
    }

    // mu lamde参数计算
    for (iz = 0; iz < nzt; iz++)
        for (iy = 0; iy < nyt; iy++)
            for (ix = 0; ix < nxt; ix++)
            {
                edxi2[iz * nxt * nyt + iy * nxt + ix] = exp(-(dxi2[iz * nxt * nyt + iy * nxt + ix]) * DT);
                edxi[iz * nxt * nyt + iy * nxt + ix] = exp(-(dxi[iz * nxt * nyt + iy * nxt + ix]) * DT);
                edyj2[iz * nxt * nyt + iy * nxt + ix] = exp(-(dyj2[iz * nxt * nyt + iy * nxt + ix]) * DT);
                edyj[iz * nxt * nyt + iy * nxt + ix] = exp(-(dyj[iz * nxt * nyt + iy * nxt + ix]) * DT);
                edzk2[iz * nxt * nyt + iy * nxt + ix] = exp(-(dzk2[iz * nxt * nyt + iy * nxt + ix]) * DT);
                edzk[iz * nxt * nyt + iy * nxt + ix] = exp(-(dzk[iz * nxt * nyt + iy * nxt + ix]) * DT);
            }
    // 计算的参数
    vx = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    vy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    vz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    txx = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    tyy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    tzz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    txy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    txz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    tyz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(vx, 0, nzt * nyt * nxt * sizeof(float));
    memset(vy, 0, nzt * nyt * nxt * sizeof(float));
    memset(vz, 0, nzt * nyt * nxt * sizeof(float));
    memset(txx, 0, nzt * nyt * nxt * sizeof(float));
    memset(tyy, 0, nzt * nyt * nxt * sizeof(float));
    memset(tzz, 0, nzt * nyt * nxt * sizeof(float));
    memset(txy, 0, nzt * nyt * nxt * sizeof(float));
    memset(txz, 0, nzt * nyt * nxt * sizeof(float));
    memset(tyz, 0, nzt * nyt * nxt * sizeof(float));
    pmlxSxx = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlySxy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlzSxz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlxSxy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlySyy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlzSyz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlxSxz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlySyz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlzSzz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(pmlxSxx, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlySxy, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlzSxz, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlxSxy, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlySyy, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlzSyz, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlxSxz, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlySyz, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlzSzz, 0, nzt * nyt * nxt * sizeof(float));
    DxSxxPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DySxyPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DzSxzPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DxSxyPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DySyyPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DzSyzPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DxSxzPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DySyzPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DzSzzPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(DxSxxPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DySxyPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DzSxzPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DxSxyPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DySyyPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DzSyzPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DxSxzPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DySyzPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DzSzzPre, 0, nzt * nyt * nxt * sizeof(float));
    DxVxPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DyVyPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DzVzPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DyVxPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DxVyPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DzVxPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DxVzPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DzVyPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    DyVzPre = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(DxVxPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DyVyPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DzVzPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DyVxPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DxVyPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DzVxPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DxVzPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DzVyPre, 0, nzt * nyt * nxt * sizeof(float));
    memset(DyVzPre, 0, nzt * nyt * nxt * sizeof(float));
    pmlxVx = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlyVy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlzVz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlzVx = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlxVz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlyVx = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlxVy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlzVy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    pmlyVz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(pmlxVx, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlyVy, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlzVz, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlzVx, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlxVz, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlyVx, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlxVy, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlzVy, 0, nzt * nyt * nxt * sizeof(float));
    memset(pmlyVz, 0, nzt * nyt * nxt * sizeof(float));
    // 存放打包的数据
    float *vxSave2X, *vySave2Y, *vzSave2Z;
    vxSave2X = (float *)malloc(sizeof(float *) * nyt * nzt);
    vySave2Y = (float *)malloc(sizeof(float *) * nxt * nzt);
    vzSave2Z = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *vySaveX, *vzSaveX, *vxSaveY, *vzSaveY, *vxSaveZ, *vySaveZ;
    vySaveX = (float *)malloc(sizeof(float *) * nyt * nzt);
    vzSaveX = (float *)malloc(sizeof(float *) * nyt * nzt);
    vxSaveY = (float *)malloc(sizeof(float *) * nxt * nzt);
    vzSaveY = (float *)malloc(sizeof(float *) * nxt * nzt);
    vxSaveZ = (float *)malloc(sizeof(float *) * nxt * nyt);
    vySaveZ = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *txxSaveX = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *txySave2X = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *txzSave2X = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *tyySaveY = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *tyzSave2Y = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *txySave2Y = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *tzzSaveZ = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *txzSave2Z = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *tyzSave2Z = (float *)malloc(sizeof(float *) * nxt * nyt);
    memset(vxSave2X, 0, nyt * nzt * sizeof(float));
    memset(vySave2Y, 0, nxt * nzt * sizeof(float));
    memset(vzSave2Z, 0, nxt * nyt * sizeof(float));
    memset(vySaveX, 0, nyt * nzt * sizeof(float));
    memset(vzSaveX, 0, nyt * nzt * sizeof(float));
    memset(vxSaveY, 0, nxt * nzt * sizeof(float));
    memset(vzSaveY, 0, nxt * nzt * sizeof(float));
    memset(vxSaveZ, 0, nxt * nyt * sizeof(float));
    memset(vySaveZ, 0, nxt * nyt * sizeof(float));
    memset(txxSaveX, 0, nyt * nzt * sizeof(float));
    memset(txySave2X, 0, nyt * nzt * sizeof(float));
    memset(txzSave2X, 0, nyt * nzt * sizeof(float));
    memset(tyySaveY, 0, nxt * nzt * sizeof(float));
    memset(tyzSave2Y, 0, nxt * nzt * sizeof(float));
    memset(txySave2Y, 0, nxt * nzt * sizeof(float));
    memset(tzzSaveZ, 0, nxt * nyt * sizeof(float));
    memset(txzSave2Z, 0, nxt * nyt * sizeof(float));
    memset(tyzSave2Z, 0, nxt * nyt * sizeof(float));
    // 解包数据
    float *vxRecv2X = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *vyRecv2Y = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *vzRecv2Z = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *vyRecvX = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *vzRecvX = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *vxRecvY = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *vzRecvY = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *vxRecvZ = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *vyRecvZ = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *txxRecvX = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *txyRecv2X = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *txzRecv2X = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *tyyRecvY = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *tyzRecv2Y = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *txyRecv2Y = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *txzRecv2Z = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *tyzRecv2Z = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *tzzRecvZ = (float *)malloc(sizeof(float *) * nxt * nyt);
    memset(vxRecv2X, 0, nyt * nzt * sizeof(float));
    memset(vyRecv2Y, 0, nxt * nzt * sizeof(float));
    memset(vzRecv2Z, 0, nxt * nyt * sizeof(float));
    memset(vyRecvX, 0, nyt * nzt * sizeof(float));
    memset(vzRecvX, 0, nyt * nzt * sizeof(float));
    memset(vxRecvY, 0, nxt * nzt * sizeof(float));
    memset(vzRecvY, 0, nxt * nzt * sizeof(float));
    memset(vxRecvZ, 0, nxt * nyt * sizeof(float));
    memset(vyRecvZ, 0, nxt * nyt * sizeof(float));
    memset(txxRecvX, 0, nyt * nzt * sizeof(float));
    memset(txyRecv2X, 0, nyt * nzt * sizeof(float));
    memset(txzRecv2X, 0, nyt * nzt * sizeof(float));
    memset(tyyRecvY, 0, nxt * nzt * sizeof(float));
    memset(tyzRecv2Y, 0, nxt * nzt * sizeof(float));
    memset(txyRecv2Y, 0, nxt * nzt * sizeof(float));
    memset(txzRecv2Z, 0, nxt * nyt * sizeof(float));
    memset(tyzRecv2Z, 0, nxt * nyt * sizeof(float));
    memset(tzzRecvZ, 0, nxt * nyt * sizeof(float));
    // 控制网格频散
    if (Vsmin / (F0 * H) < 15)
        printf("空间步长太大, 可能引起明显的网格频散\n");
    // 从核初始化
    athread_init();
    // 检查稳定性条件
    best_dt = 6.0 * H / (7.0 * sqrt(2.0) * Vpmax);
    if (DT >= best_dt)
        printf("时间步长过大，应该小于 %f\n", best_dt);
    // 全局弹性模量分配到各进程中
    float *elastic_pro = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(elastic_pro, 0, nzt * nyt * nxt * sizeof(float));
    for (iz = 0; iz < nzt; iz++)
    {
        for (iy = 0; iy < nyt; iy++)
        {
            for (ix = 0; ix < nxt; ix++)
            {
                int findindex = (iz + nzt * coord[2]) * NX_ext * NY_ext + (iy + nyt * coord[1]) * NX_ext + (ix + nxt * coord[0]);
                elastic_pro[iz * nxt * nyt + iy * nxt + ix] = elastic_pre_all[findindex];
            }
        }
    }
    free(elastic_pre_all);
    //**********************************************************************************************************************************
    for (int prenumber = 0; prenumber < 1; prenumber++)
    {
        float pressure0 = pressure[prenumber];
        if (myid == 0)
            printf("presuure = %f\n", pressure0);
        // 计算弹性模量
        create_model(vp, vs, rho_ext, vf, rhof, lamda_ext, lamda2u_ext, mu_ext, nzt, nyt, nxt, myid, coord, elastic_pro, pressure0);
        // mu lamde参数计算
        for (iz = 0; iz < nzt; iz++)
            for (iy = 0; iy < nyt; iy++)
                for (ix = 0; ix < nxt; ix++)
                {
                    if (mu_ext[iz * nxt * nyt + iy * nxt + ix] == 0 || mu_ext[iz * nxt * nyt + iy * nxt + ix + 1] == 0 || mu_ext[(iz + 1) * nxt * nyt + iy * nxt + ix] == 0 || mu_ext[(iz + 1) * nxt * nyt + iy * nxt + ix + 1] == 0)
                        muxz[iz * nxt * nyt + iy * nxt + ix] = 0.0;
                    else
                        muxz[iz * nxt * nyt + iy * nxt + ix] = 1 / (0.25 * (1 / mu_ext[iz * nxt * nyt + iy * nxt + ix] + 1 / mu_ext[iz * nxt * nyt + iy * nxt + ix + 1] + 1 / mu_ext[(iz + 1) * nxt * nyt + iy * nxt + ix] + 1 / mu_ext[(iz + 1) * nxt * nyt + iy * nxt + ix + 1]));
                    if (mu_ext[iz * nxt * nyt + iy * nxt + ix] == 0 || mu_ext[iz * nxt * nyt + (iy + 1) * nxt + ix] == 0 || mu_ext[(iz + 1) * nxt * nyt + iy * nxt + ix] == 0 || mu_ext[(iz + 1) * nxt * nyt + (iy + 1) * nxt + ix] == 0)
                        muyz[iz * nxt * nyt + iy * nxt + ix] = 0.0;
                    else
                        muyz[iz * nxt * nyt + iy * nxt + ix] = 1 / (0.25 * (1 / mu_ext[iz * nxt * nyt + iy * nxt + ix] + 1 / mu_ext[iz * nxt * nyt + (iy + 1) * nxt + ix] + 1 / mu_ext[(iz + 1) * nxt * nyt + iy * nxt + ix] + 1 / mu_ext[(iz + 1) * nxt * nyt + (iy + 1) * nxt + ix]));
                    if (mu_ext[iz * nxt * nyt + iy * nxt + ix] == 0 || mu_ext[iz * nxt * nyt + iy * nxt + ix + 1] == 0 || mu_ext[iz * nxt * nyt + (iy + 1) * nxt + ix] == 0 || mu_ext[iz * nxt * nyt + (iy + 1) * nxt + ix + 1] == 0)
                        muxy[iz * nxt * nyt + iy * nxt + ix] = 0.0;
                    else
                        muxy[iz * nxt * nyt + iy * nxt + ix] = 1 / (0.25 * (1 / mu_ext[iz * nxt * nyt + iy * nxt + ix] + 1 / mu_ext[iz * nxt * nyt + iy * nxt + ix + 1] + 1 / mu_ext[iz * nxt * nyt + (iy + 1) * nxt + ix] + 1 / mu_ext[iz * nxt * nyt + (iy + 1) * nxt + ix + 1]));
                }

        clock_t start, finish;
        start = clock();
        int SourcePro;
        for (it = 0; it < NT; it++)
        {
            SourcePro = PX * PY * z0pro + PX * y0pro + x0pro;
            tt = it * DT;
            if (it % 50 == 0 && myid == 1)
                printf("it=%d\n", it);

            //  计算质点速度v
            // lwpf_init(NULL);
            athread_spawn(funVelocity, 0);
            athread_join();

            float I_sou;
            I_sou =  -1 * (1 - 2 * PI * PI * F0 * F0 * (tt - T0) * (tt - T0)) * exp(-PI * PI * F0 * F0 * (tt - T0) * (tt - T0));
            //注入体力源
            if (coordx == x0pro && coordy == y0pro && coordz == z0pro)
            {
                if(waveType == 0)
                {
                    //p wave
                    vz[sznew * nxt * nyt + synew * nxt + sxnew] = vz[sznew * nxt * nyt + synew * nxt + sxnew] + I_sou / rho_ext[sznew * nxt * nyt + synew * nxt + sxnew] * DT;
                }
                if(waveType == 1)
                {
                    vx[sznew * nxt * nyt + synew * nxt + sxnew] = vx[sznew * nxt * nyt + synew * nxt + sxnew] + I_sou / rho_ext[sznew * nxt * nyt + synew * nxt + sxnew] * DT;
                }
                if(waveType == 2)
                {
                    vy[sznew * nxt * nyt + synew * nxt + sxnew] = vy[sznew * nxt * nyt + synew * nxt + sxnew] + I_sou / rho_ext[sznew * nxt * nyt + synew * nxt + sxnew] * DT;
                }
            }

            // 打包不连续的数据
            int xx, yy, zz;
            for (zz = 0; zz < nzt; zz++)
            {
                // yoz
                for (yy = 0; yy < nyt; yy++)
                {
                    vxSave2X[zz * nyt + yy] = vx[zz * nxt * nyt + yy * nxt + nxt - 2]; // 右侧

                    vySaveX[zz * nyt + yy] = vy[zz * nxt * nyt + yy * nxt + tran]; // 左侧
                    vzSaveX[zz * nyt + yy] = vz[zz * nxt * nyt + yy * nxt + tran]; // 左侧
                }
                // xoz
                for (xx = 0; xx < nxt; xx++)
                {
                    vxSaveY[zz * nxt + xx] = vx[zz * nxt * nyt + tran * nxt + xx]; // 后侧
                    vzSaveY[zz * nxt + xx] = vz[zz * nxt * nyt + tran * nxt + xx]; // 后侧

                    vySave2Y[zz * nxt + xx] = vy[zz * nxt * nyt + (nyt - 2) * nxt + xx]; // 前
                }
            }
            // xoy
            for (yy = 0; yy < nyt; yy++)
            {
                for (xx = 0; xx < nxt; xx++)
                {
                    vzSave2Z[yy * nxt + xx] = vz[(nzt - 2) * nxt * nyt + yy * nxt + xx];

                    vxSaveZ[yy * nxt + xx] = vx[tran * nxt * nyt + yy * nxt + xx];
                    vySaveZ[yy * nxt + xx] = vy[tran * nxt * nyt + yy * nxt + xx];
                }
            }
            // 将消息进行传递 过渡层的传递
            // 往前传递并接收
            if (forward >= 0)
            {
                MPI_Send(vySave2Y, (nxt * nzt), MPI_FLOAT, forward, 9, MCW);
                MPI_Recv(vxRecvY, (nxt * nzt), MPI_FLOAT, forward, 6, MCW, &status);
                MPI_Recv(vzRecvY, (nxt * nzt), MPI_FLOAT, forward, 10, MCW, &status);
                for (zz = 0; zz < nzt; zz++)
                {
                    for (xx = 0; xx < nxt; xx++)
                    {
                        vx[zz * nxt * nyt + (nyt - 1) * nxt + xx] = vxRecvY[zz * nxt + xx];
                        vz[zz * nxt * nyt + (nyt - 1) * nxt + xx] = vzRecvY[zz * nxt + xx];
                    }
                }
            }
            if (back >= 0)
            {
                MPI_Recv(vyRecv2Y, (nxt * nzt), MPI_FLOAT, back, 9, MCW, &status);
                MPI_Send(vxSaveY, (nxt * nzt), MPI_FLOAT, back, 6, MCW);
                MPI_Send(vzSaveY, (nxt * nzt), MPI_FLOAT, back, 10, MCW);
                // 解包赋值
                for (zz = 0; zz < nzt; zz++)
                {
                    for (xx = 0; xx < nxt; xx++)
                    {
                        vy[zz * nxt * nyt + (tran - 1) * nxt + xx] = vyRecv2Y[zz * nxt + xx];
                    }
                }
            }
            if (left >= 0)
            {
                MPI_Send(vySaveX, (nyt * nzt), MPI_FLOAT, left, 2, MCW);
                MPI_Send(vzSaveX, (nyt * nzt), MPI_FLOAT, left, 4, MCW);
                MPI_Recv(vxRecv2X, (nyt * nzt), MPI_FLOAT, left, 1, MCW, &status);

                for (zz = 0; zz < nzt; zz++)
                {
                    for (yy = 0; yy < nyt; yy++)
                    {
                        vx[zz * nxt * nyt + yy * nxt + tran - 1] = vxRecv2X[zz * nyt + yy];
                    }
                }
            }
            if (right >= 0)
            {
                MPI_Recv(vyRecvX, (nyt * nzt), MPI_FLOAT, right, 2, MCW, &status);
                MPI_Recv(vzRecvX, (nyt * nzt), MPI_FLOAT, right, 4, MCW, &status);
                MPI_Send(vxSave2X, (nyt * nzt), MPI_FLOAT, right, 1, MCW);

                for (zz = 0; zz < nzt; zz++)
                {
                    for (yy = 0; yy < nyt; yy++)
                    {
                        vy[zz * nxt * nyt + yy * nxt + nxt - 1] = vyRecvX[zz * nyt + yy];
                        vz[zz * nxt * nyt + yy * nxt + nxt - 1] = vzRecvX[zz * nyt + yy];
                    }
                }
            }
            if (down >= 0)
            {
                MPI_Send(vxSaveZ, (nxt * nyt), MPI_FLOAT, down, 3, MCW);
                MPI_Send(vySaveZ, (nxt * nyt), MPI_FLOAT, down, 5, MCW);
                MPI_Recv(vzRecv2Z, (nxt * nyt), MPI_FLOAT, down, 7, MCW, &status);
                for (yy = 0; yy < nyt; yy++)
                {
                    for (xx = 0; xx < nxt; xx++)
                    {
                        vz[(tran - 1) * nxt * nyt + yy * nxt + xx] = vzRecv2Z[yy * nxt + xx];
                    }
                }
            }
            if (up >= 0)
            {
                MPI_Recv(vxRecvZ, (nxt * nyt), MPI_FLOAT, up, 3, MCW, &status);
                MPI_Recv(vyRecvZ, (nxt * nyt), MPI_FLOAT, up, 5, MCW, &status);
                MPI_Send(vzSave2Z, (nxt * nyt), MPI_FLOAT, up, 7, MCW);
                for (yy = 0; yy < nyt; yy++)
                {
                    for (xx = 0; xx < nxt; xx++)
                    {
                        vx[(nzt - 1) * nxt * nyt + yy * nxt + xx] = vxRecvZ[yy * nxt + xx];
                        vy[(nzt - 1) * nxt * nyt + yy * nxt + xx] = vyRecvZ[yy * nxt + xx];
                    }
                }
            }
            MPI_Barrier(MCW);
            //*****************************************************************************************************
            // 计算应力
            //  lwpf_init(NULL);

            athread_spawn(funStress, 0);

            athread_join();

            // lwpf_report_summary(stdout);
            // lwpf_report_detail(stdout);

            // 打包不连续的数据
            for (zz = 0; zz < nzt; zz++)
            {
                // yoz
                for (yy = 0; yy < nyt; yy++)
                {
                    txxSaveX[zz * nyt + yy] = txx[zz * nxt * nyt + yy * nxt + tran]; // 左侧

                    txySave2X[zz * nyt + yy] = txy[zz * nxt * nyt + yy * nxt + nxt - 2]; // 右侧
                    txzSave2X[zz * nyt + yy] = txz[zz * nxt * nyt + yy * nxt + nxt - 2]; // 右侧
                }
                // xoz
                for (xx = 0; xx < nxt; xx++)
                {
                    tyySaveY[zz * nxt + xx] = tyy[zz * nxt * nyt + tran * nxt + xx];       // 后侧
                    tyzSave2Y[zz * nxt + xx] = tyz[zz * nxt * nyt + (nyt - 2) * nxt + xx]; // 前
                    txySave2Y[zz * nxt + xx] = txy[zz * nxt * nyt + (nyt - 2) * nxt + xx]; // 前
                }
            }
            // xoy
            for (yy = 0; yy < nyt; yy++)
            {
                for (xx = 0; xx < nxt; xx++)
                {
                    tzzSaveZ[yy * nxt + xx] = tzz[tran * nxt * nyt + yy * nxt + xx];
                    txzSave2Z[yy * nxt + xx] = txz[(nzt - 2) * nxt * nyt + yy * nxt + xx];
                    tyzSave2Z[yy * nxt + xx] = tyz[(nzt - 2) * nxt * nyt + yy * nxt + xx];
                }
            }
            // 将消息进行传递 过渡层的传递
            //*****************************************************************************************************
            if (forward >= 0)
            {
                MPI_Send(tyzSave2Y, (nxt * nzt), MPI_FLOAT, forward, 16, MCW);
                MPI_Send(txySave2Y, (nxt * nzt), MPI_FLOAT, forward, 17, MCW);
                MPI_Recv(tyyRecvY, (nxt * nzt), MPI_FLOAT, forward, 15, MCW, &status);
                for (zz = 0; zz < nzt; zz++)
                {
                    for (xx = 0; xx < nxt; xx++)
                    {
                        tyy[zz * nxt * nyt + (nyt - 1) * nxt + xx] = tyyRecvY[zz * nxt + xx];
                    }
                }
            }
            if (back >= 0)
            {
                MPI_Recv(tyzRecv2Y, (nxt * nzt), MPI_FLOAT, back, 16, MCW, &status);
                MPI_Recv(txyRecv2Y, (nxt * nzt), MPI_FLOAT, back, 17, MCW, &status);
                MPI_Send(tyySaveY, (nxt * nzt), MPI_FLOAT, back, 15, MCW);
                for (zz = 0; zz < nzt; zz++)
                {
                    for (xx = 0; xx < nxt; xx++)
                    {
                        tyz[zz * nxt * nyt + (tran - 1) * nxt + xx] = tyzRecv2Y[zz * nxt + xx];
                        txy[zz * nxt * nyt + (tran - 1) * nxt + xx] = txyRecv2Y[zz * nxt + xx];
                    }
                }
            }
            if (left >= 0)
            {
                MPI_Send(txxSaveX, (nyt * nzt), MPI_FLOAT, left, 14, MCW);
                MPI_Recv(txyRecv2X, (nyt * nzt), MPI_FLOAT, left, 12, MCW, &status);
                MPI_Recv(txzRecv2X, (nyt * nzt), MPI_FLOAT, left, 13, MCW, &status);
                for (zz = 0; zz < nzt; zz++)
                {
                    for (yy = 0; yy < nyt; yy++)
                    {
                        txy[zz * nxt * nyt + yy * nxt + tran - 1] = txyRecv2X[zz * nyt + yy];
                        txz[zz * nxt * nyt + yy * nxt + tran - 1] = txzRecv2X[zz * nyt + yy];
                    }
                }
            }
            if (right >= 0)
            {
                MPI_Recv(txxRecvX, (nyt * nzt), MPI_FLOAT, right, 14, MCW, &status);
                MPI_Send(txySave2X, (nyt * nzt), MPI_FLOAT, right, 12, MCW);
                MPI_Send(txzSave2X, (nyt * nzt), MPI_FLOAT, right, 13, MCW);
                for (zz = 0; zz < nzt; zz++)
                {
                    for (yy = 0; yy < nyt; yy++)
                    {
                        txx[zz * nxt * nyt + yy * nxt + nxt - 1] = txxRecvX[zz * nyt + yy];
                    }
                }
            }
            if (down >= 0)
            {
                MPI_Send(tzzSaveZ, (nxt * nyt), MPI_FLOAT, down, 18, MCW);
                MPI_Recv(txzRecv2Z, (nxt * nyt), MPI_FLOAT, down, 19, MCW, &status);
                MPI_Recv(tyzRecv2Z, (nxt * nyt), MPI_FLOAT, down, 20, MCW, &status);
                for (yy = 0; yy < nyt; yy++)
                {
                    for (xx = 0; xx < nxt; xx++)
                    {
                        txz[(tran - 1) * nxt * nyt + yy * nxt + xx] = txzRecv2Z[yy * nxt + xx];
                        tyz[(tran - 1) * nxt * nyt + yy * nxt + xx] = tyzRecv2Z[yy * nxt + xx];
                    }
                }
            }
            if (up >= 0)
            {
                MPI_Recv(tzzRecvZ, (nxt * nyt), MPI_FLOAT, up, 18, MCW, &status);
                MPI_Send(txzSave2Z, (nxt * nyt), MPI_FLOAT, up, 19, MCW);
                MPI_Send(tyzSave2Z, (nxt * nyt), MPI_FLOAT, up, 20, MCW);
                for (yy = 0; yy < nyt; yy++)
                {
                    for (xx = 0; xx < nxt; xx++)
                    {
                        tzz[(nzt - 1) * nxt * nyt + yy * nxt + xx] = tzzRecvZ[yy * nxt + xx];
                    }
                }
            }
            //***********************************************************************************************************************
            // 输出波场快照
            for (iz = 0; iz < nzt; iz++)
            {
                for (iy = 0; iy < nyt; iy++)
                {
                    for (ix = 0; ix < nxt; ix++)
                    {
                        txx3d[iz][iy][ix] = txx[iz * nxt * nyt + iy * nxt + ix];
                    }
                }
            }
            // 定义输出文件名
            int namepressure = pressure[prenumber];
            if (it == 2000)
            {
                for (int plot = 0; plot < PX; plot++)
                {
                    if (myid >= (PY * PZ * plot + y0pro * PZ) && myid < (PY * PZ * plot + y0pro * PZ + PZ))
                    {
                        int name1000 = myid;
                        char txxname1000[] = "txx1000";
                        char txx1000name[50];
                        sprintf(txx1000name, "%s Pro%d Pressure%d.dat", txxname1000, name1000, namepressure);
                        wfile3d(txx1000name, txx3d, nzt, nyt, nxt);
                    }
                }
            }
            if (it == 3000)
            {
                for (int plot = 0; plot < PX; plot++)
                {
                    if (myid >= (PY * PZ * plot + y0pro * PZ) && myid < (PY * PZ * plot + y0pro * PZ + PZ))
                    {
                        int name2000 = myid;
                        char txxname2000[] = "txx2000";
                        char txx2000name[50];
                        sprintf(txx2000name, "%s Pro%d Pressure%d.dat", txxname2000, name2000, namepressure);
                        wfile3d(txx2000name, txx3d, nzt, nyt, nxt);
                    }
                }
            }
            if (it == 4000)
            {
                for (int plot = 0; plot < PX; plot++)
                {
                    if (myid >= (PY * PZ * plot + y0pro * PZ) && myid < (PY * PZ * plot + y0pro * PZ + PZ))
                    {
                        int name3000 = myid;
                        char txxname3000[] = "txx3000";
                        char txx3000name[50];
                        sprintf(txx3000name, "%s Pro%d Pressure%d.dat", txxname3000, name3000, namepressure);
                        wfile3d(txx3000name, txx3d, nzt, nyt, nxt);
                    }
                }
            }
            // 记录点源处下方
            if (coordx == x0pro && coordy == y0pro)
            {
                if (it % 1 == 0)
                {
                    for (int sisz = 0; sisz < nzt; sisz++)
                    {
                        sis_vx[sisz][it] = vx[sisz * nxt * nyt + synew * nxt + sxnew];
                        sis_vy[sisz][it] = vy[sisz * nxt * nyt + synew * nxt + sxnew];
                        sis_vz[sisz][it] = vz[sisz * nxt * nyt + synew * nxt + sxnew];
                    }
                }
            }
        }
        // 结束从核
        athread_halt();
        finish = clock();
        printf("myid  = %d   %f seconds\n", myid, (double)(finish - start) / CLOCKS_PER_SEC);
        // 输出波场快照
        int namepressure = pressure[prenumber];
        if (coordx == x0pro && coordy == y0pro)
        {
            int namesisvz = myid;
            char sisvxname3000[] = "sisvx";
            char sisvx3000name[50];
            sprintf(sisvx3000name, "%s Pro%d Pressure%d.dat", sisvxname3000, namesisvz, namepressure);
            wfile(sisvx3000name, sis_vx, nzt, NT);

            char sisvyname3000[] = "sisvy";
            char sisvy3000name[50];
            sprintf(sisvy3000name, "%s Pro%d Pressure%d.dat", sisvyname3000, namesisvz, namepressure);
            wfile(sisvy3000name, sis_vy, nzt, NT);

            char sisvzname3000[] = "sisvz";
            char sisvz3000name[50];
            sprintf(sisvz3000name, "%s Pro%d Pressure%d.dat", sisvzname3000, namesisvz, namepressure);
            wfile(sisvz3000name, sis_vz, nzt, NT);
        }
    }
    // 释放动态空间
    free_space3d(txx3d, nzt, nyt);
    free_space2d(sis_vx, nzt);
    free_space2d(sis_vy, nzt);
    free_space2d(sis_vz, nzt);
    free(elastic_pro);
    // 测试参数
    free(rho_ext);
    free(vs);
    free(vp);
    free(vf);
    free(rhof);
    free(mu_ext);
    free(lamda_ext);
    free(lamda2u_ext);
    free(tzz);
    free(tyy);
    free(txx);
    free(txy);
    free(txz);
    free(tyz);
    free(vx);
    free(vy);
    free(vz);
    free(dxi);
    free(dxi2);
    free(dyj);
    free(dyj2);
    free(dzk);
    free(dzk2);
    free(edxi);
    free(edxi2);
    free(edyj);
    free(edyj2);
    free(edzk);
    free(edzk2);
    free(muxz);
    free(muxy);
    free(muyz);
    free(DxVxPre);
    free(DyVyPre);
    free(DzVzPre);
    free(DyVxPre);
    free(DxVyPre);
    free(DzVxPre);
    free(DxVzPre);
    free(DzVyPre);
    free(DyVzPre);
    free(pmlxVx);
    free(pmlyVy);
    free(pmlzVz);
    free(pmlzVx);
    free(pmlxVz);
    free(pmlyVx);
    free(pmlxVy);
    free(pmlzVy);
    free(pmlyVz);
    free(pmlxSxx);
    free(pmlySxy);
    free(pmlzSxz);
    free(pmlxSxy);
    free(pmlySyy);
    free(pmlzSyz);
    free(pmlxSxz);
    free(pmlySyz);
    free(pmlzSzz);
    free(DxSxxPre);
    free(DySxyPre);
    free(DzSxzPre);
    free(DxSxyPre);
    free(DySyyPre);
    free(DzSyzPre);
    free(DxSxzPre);
    free(DySyzPre);
    free(DzSzzPre);
    free(vxSave2X);
    free(vySave2Y);
    free(vzSave2Z);
    free(vxRecv2X);
    free(vyRecv2Y);
    free(vzRecv2Z);
    free(vySaveX);
    free(vzSaveX);
    free(vxSaveY);
    free(vzSaveY);
    free(vxSaveZ);
    free(vySaveZ);
    free(vyRecvX);
    free(vzRecvX);
    free(vxRecvY);
    free(vzRecvY);
    free(vxRecvZ);
    free(vyRecvZ);
    free(txxSaveX);
    free(txySave2X);
    free(txzSave2X);
    free(tyySaveY);
    free(tyzSave2Y);
    free(txySave2Y);
    free(txxRecvX);
    free(txyRecv2X);
    free(txzRecv2X);
    free(tyyRecvY);
    free(tyzRecv2Y);
    free(txyRecv2Y);
    free(tzzSaveZ);
    free(txzRecv2Z);
    free(tyzRecv2Z);
    free(tzzRecvZ);
    free(txzSave2Z);
    free(tyzSave2Z);
    printf("\\Press any key to exit program...");
    MPI_Finalize();
    return 0;
}
// 申请二维动态数组
float **space2d(int nr, int nc)
{
    float **a;
    int i;
    a = (float **)calloc(nr, sizeof(float *));
    for (i = 0; i < nr; i++)
    {
        a[i] = (float *)calloc(nc, sizeof(float));
    }

    return a;
}
// 申请三维动态数组
float ***space3d(int nr, int ny, int nc)
{
    float ***a;
    int i, j, k;
    a = (float ***)malloc(sizeof(float **) * nr);
    for (i = 0; i < nr; i++)
    {
        a[i] = (float **)malloc(sizeof(float *) * ny);
    }
    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < ny; j++)
        {
            a[i][j] = (float *)malloc(sizeof(float) * nc); // sizeof(float) nc改为sizeof(float)*nc
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
// 释放二维动态数组
void free_space2d(float **a, int nr)
{
    int i;
    for (i = 0; i < nr; i++)
        free(a[i]);
    free(a);
}
void free_space3d(float ***a, int nr, int ny)
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
void wfile1d(char filename[], float *data, int nr)
{
    int i, j;
    FILE *fp;
    fp = fopen(filename, "wt");

    for (i = 0; i < nr; i++)
    {

        fprintf(fp, "%e ", data[i]);

        fprintf(fp, "\n");
    }
    //          fwrite(&data[i][j],1,sizeof(float),fp);
    fclose(fp);
}
// 将二进制数据写入文件—二维
void wfile(char filename[], float **data, int nr, int nc)
{
    int i, j;
    FILE *fp;
    fp = fopen(filename, "wt");

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
    //          fwrite(&data[i][j],1,sizeof(float),fp);
    fclose(fp);
}
// 将二进制数据写入文件—三维
void wfile3d(char filename[], float ***data, int nr, int ny, int nc)
{
    int i, j, k;
    FILE *fp;
    fp = fopen(filename, "wt");
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
                j = 25;
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
// 将二进制数据写入文件—三维x0y direction
void wfile3dy(char filename[], float ***data, int nr, int ny, int nc)
{
    int i, j, k;
    FILE *fp;
    fp = fopen(filename, "wt");
    {
        for (i = 0; i < nr; i++)
        {
            for (k = 0; k < nc; k++)
            {
                j = 48;
                fprintf(fp, "%e ", data[j][i][k]);
                if ((k + 1) % nc == 0)
                    fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
}
// 二进制文件读入
void readCompressedBinary(float *aa, int totalSize, const char *filename)
{

    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("no open file");
        return;
    }

    int buffer = 0;
    int bufferIndex = 0;
    int aaIndex = 0;

    int *A = (int *)calloc(totalSize / 10, sizeof(int));
    int G = 0;
    fread(A, sizeof(int), totalSize / 10, file);

    while (aaIndex < totalSize)
    {
        buffer = A[G];
        G++;
        for (int i = 0; i < 10 && aaIndex < totalSize; i++)
        {

            int shiftAmount = 32 - (i + 1) * 3;
            int value = (buffer >> shiftAmount) & 0x7;

            aa[aaIndex] = (float)value;
            aaIndex++;
        }
    }

    fclose(file);
    free(A);
}
// 建立地质模型
void create_model(float *vp, float *vs, float *rho, float *vf, float *rhof, float *lamda, float *lamda2u, float *mu, int nr, int ny, int nc, int myid, int coord[], float *elastic_pre, float pressure)
{
    int ix, iy, iz;
    // 定义温度
    float Temp = 20;
    // 定义各个矿物的速度参数
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
    // Vp_mineral[8] = 4015.0f;
    // Vs_mineral[8] = 2655.2f;
    // rho_mineral[8] = 2292.7f;
    Vp_mineral[8] = 3000.0f;
    Vs_mineral[8] = 1700.0f;
    rho_mineral[8] = 2292.7f;

    // 对rho，vs，vp赋值,计算弹性系数
    for (iz = 0; iz < nr; iz++)
    {
        for (iy = 0; iy < ny; iy++)
        {
            for (ix = 0; ix < nc; ix++)
            {

                vs[iz * nc * ny + iy * nc + ix] = Vs_mineral[(int)elastic_pre[iz * nc * ny + iy * nc + ix]];
                vp[iz * nc * ny + iy * nc + ix] = Vp_mineral[(int)elastic_pre[iz * nc * ny + iy * nc + ix]];
                rho[iz * nc * ny + iy * nc + ix] = rho_mineral[(int)elastic_pre[iz * nc * ny + iy * nc + ix]];
                mu[iz * nc * ny + iy * nc + ix] = rho[iz * nc * ny + iy * nc + ix] * vs[iz * nc * ny + iy * nc + ix] * vs[iz * nc * ny + iy * nc + ix];
                lamda2u[iz * nc * ny + iy * nc + ix] = rho[iz * nc * ny + iy * nc + ix] * vp[iz * nc * ny + iy * nc + ix] * vp[iz * nc * ny + iy * nc + ix];
                lamda[iz * nc * ny + iy * nc + ix] = lamda2u[iz * nc * ny + iy * nc + ix] - 2.0 * mu[iz * nc * ny + iy * nc + ix];
            }
        }
    }
}
