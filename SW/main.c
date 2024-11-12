#pragma hdrstop
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <athread.h>
#include <stdio.h>
#include <mpi.h>

typedef struct
{
    int z;
    int y;
    int x;
} Index3D;

float ***space3d(int nr, int ny, int nc);
float ****space4d(int n1, int n2, int n3, int n4);
void free_space3d(float ***b, int nr, int ny);
void free_space4d(float ****a, int nr, int ny, int nc);
Index3D *find_nonzero_elements(float *array, int *Boundray, int NZ, int NY, int NX);
void zero3dArray(float ***array, int nr, int ny, int nc);
int countElements(float ***rock, int lx, int ly, int lz, float t);
int countElements1D(float *rock, int lz, int ly, int lx, float t);
void wfile1d(char filename[], float *data, int nz, int ny, int nx);
void wfile3d(char filename[], float ***data, int nr, int ny, int nc);
void write_data_to_file(float ***data, int nz, int ny, int nx, const char *filename);
void write_data_to_file2BIN(float *data, int totalSize, const char *filename);
void write_4d_data_to_file(float ****data, int nh, int nz, int ny, int nx);
void readCompressedBinary(float *aa, int totalSize, const char *filename);
void write_data_to_file2BIN111(float *data, int totalSize, const char *filename);
int divUp(int a, int b) { return (a + b - 1) / b; }

extern SLAVE_FUN(CalTempt)();
extern SLAVE_FUN(CalBGK)();
extern SLAVE_FUN(CalBGK1)();
extern SLAVE_FUN(UpdateBoundary)();
extern SLAVE_FUN(UpdateMidff)();
extern SLAVE_FUN(UpdateMidff1)();
extern SLAVE_FUN(Updateff)();
extern SLAVE_FUN(Updaterho)();
extern SLAVE_FUN(UpdateUPot)();
extern SLAVE_FUN(PotBoundaryCalP)();
extern SLAVE_FUN(UpdateMidPot)();
extern SLAVE_FUN(UpdateMidPot1)();
extern SLAVE_FUN(UpdatePot)();
extern SLAVE_FUN(CalF)();
extern SLAVE_FUN(BoundaryF)();
extern SLAVE_FUN(UpdateF)();
float G1, G2, B;
int nxt, nyt, nzt, npop;
int nxt2, nyt2, nzt2;
int Step4D;
float t_max;
int myid;
int back, forward, left, right;
float *data, *obst;
int Boundary;
int *nonzero_indicesX, *nonzero_indicesY, *nonzero_indicesZ;
float *rho, *rhoexp, *ff, *feq, *tempt;
float *ux, *uy, *uz;
float *u_x, *u_y, *u_z, *cu;
float *F1sx, *F1sy, *F1sz, *Fmx1, *Fmy1, *Fmz1, *Fmx2, *Fmy2, *Fmz2;
float cx1, cy1, cz1, tau;
float *fftemp;
float *fftemp2;
float *pp, *psx, *p;
float *psxpretemp, *spretemp;
float *psxpretemp2, *spretemp2;
float *psxpre, *spre;
float *smx_pre1, *smy_pre1, *smz_pre1;
float ***test;
float ****test4d;
//  定义xyz方向的进程数
#define PX 5
#define PY 5
#define PZ 5
#define npop 19
#define tran 1

int main(int argc, char **argv)
{
    //------------------
    int numm = 5;
    //------------------
    int xx, yy, zz;
    int NZ = 500;
    int NY = 500;
    int NX = 500;
    t_max = 10;
    G1 = -0.2777778;
    G2 = -0.12; // 调整接触角的系数， - 0.12是亲水
    B = 0.886;
    float rho_h = 1.9;
    float rho_l = 0.14;
    float rho_ratio = 1.9; // 初始密度比，变饱和度调整这里，0.14-1.9之间
    float weA1 = 1.0 / 3.0;
    float weA2 = 1.0 / 18.0;
    float weA3 = 1.0 / 36.0;
    float weight[npop] = {weA1, weA2, weA2, weA2, weA2, weA2, weA2, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3, weA3};

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
    // x y z 都是周期环面
    period[0] = 1;
    period[1] = 1;
    period[2] = 1;
    // xyz方向每个进程中有多少个网格
    nxt = NX / PX;
    nyt = NY / PY;
    nzt = NZ / PZ;
    nxt2 = nxt;
    nyt2 = nyt;
    nzt2 = nzt;
    printf("nxt = %d  nyt = %d   nzt = %d\n", nxt, nyt, nzt);
    // 创建笛卡尔拓扑
    err = MPI_Cart_create(MCW, 3, dim, period, reorder, &MC1);
    err = MPI_Cart_shift(MC1, 0, 1, &left, &right);
    err = MPI_Cart_shift(MC1, 1, 1, &back, &forward);
    err = MPI_Cart_shift(MC1, 2, 1, &down, &up);
    err = MPI_Cart_coords(MC1, myid, 3, coord);
    printf("rank = %d\t  coord[0] = %d\t  coord[1] = %d\t   coord[2] = %d\t   left = %d\t  right = %d\t  back = %d\t  forward = %d\t  up = %d\t down = %d\n", myid, coord[0], coord[1], coord[2], left, right, back, forward, up, down);
    int coordx = coord[0];
    int coordy = coord[1];
    int coordz = coord[2];
    //------------------------------------
    // 读取文件
    //------------------------------------
    float ***data11;
    float ***DATA;
    data11 = space3d(NZ, NY, NX);
    DATA = space3d(NZ, NY, NX);
    float *dataJudge = (float *)malloc(NZ * NX * NY * sizeof(float));
    memset(dataJudge, 0, NZ * NX * NY * sizeof(float));
    float *DATA1D = (float *)malloc(NZ * NY * NX * sizeof(float));
    memset(DATA1D, 0, NZ * NY * NX * sizeof(float));
    float ***DATA3D;
    DATA3D = space3d(NZ, NY, NX);

    if (myid == 0)
    {
        printf("input begin\n");
    }
    // 读取二进制文件
    // char *filename00 = "./0.092986_1.900000BIN.bin";
    // FILE *file00 = fopen(filename00, "rb");
    // if (file00 == NULL)
    // {
    //     fprintf(stderr, "no open file0\n");
    //     return 1;
    // }
    // // 调用函数
    // readCompressedBinary(testre, totalSize, filename00);

    //  打开文件
    FILE *file11;
    file11 = fopen("./outputsample1_500x500x500.dat", "r");
    // file11 = fopen("./outputsample1_400x400x400.dat", "r");
    if (file11 == NULL)
    {
        fprintf(stderr, "无法打开文件\n");
        return 1;
    }
    // 读取文件内容并存储到数组中
    int INPUTnz, INPUTny, INPUTnx;
    float INPUTvalue;
    while (fscanf(file11, "%d %d %d %f", &INPUTnz, &INPUTny, &INPUTnx, &INPUTvalue) != EOF)
    {
        if (INPUTnz >= 0 && INPUTnz < NZ && INPUTny >= 0 && INPUTny < NY && INPUTnx >= 0 && INPUTnx < NX)
        {
            data11[INPUTnz][INPUTny][INPUTnx] = INPUTvalue;
        }
        else
        {
            fprintf(stderr, "索引超出范围: nz=%d, ny=%d, nx=%d\n", INPUTnz, INPUTny, INPUTnx);
        }
    }
    // 关闭文件
    fclose(file11);

    for (int z = 0; z < NZ; z++)
        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++)
            {
                DATA1D[z * NY * NX + y * NX + x] = data11[z][y][x];

                if (data11[z][y][x] == 1)
                    data11[z][y][x] = 0;
                else
                    data11[z][y][x] = 1;

                DATA[z][y][x] = data11[z][y][x];
                dataJudge[z * NX * NY + y * NX + x] = data11[z][y][x];
            }

    obst = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(obst, 0, nzt * nyt * nxt * sizeof(float));
    for (int z = 0; z < nzt; z++)
    {
        for (int y = 0; y < nyt; y++)
        {
            for (int x = 0; x < nxt; x++)
            {
                obst[z * nxt * nyt + y * nxt + x] = data11[coordz * nzt + z][coordy * nyt + y][coordx * nxt + x];
            }
        }
    }
    // 动态分配内存
    rho = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(rho, 0, nzt * nyt * nxt * sizeof(float));
    rhoexp = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(rhoexp, 0, nzt * nyt * nxt * sizeof(float));
    ff = (float *)malloc(npop * nzt * nyt * nxt * sizeof(float));
    memset(ff, 0, npop * nzt * nyt * nxt * sizeof(float));
    feq = (float *)malloc(npop * nzt * nyt * nxt * sizeof(float));
    memset(feq, 0, npop * nzt * nyt * nxt * sizeof(float));
    tempt = (float *)malloc(npop * nzt * nyt * nxt * sizeof(float));
    memset(tempt, 0, npop * nzt * nyt * nxt * sizeof(float));
    ux = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    uy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    uz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(ux, 0, nzt * nyt * nxt * sizeof(float));
    memset(uy, 0, nzt * nyt * nxt * sizeof(float));
    memset(uz, 0, nzt * nyt * nxt * sizeof(float));
    u_x = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    u_y = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    u_z = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(u_x, 0, nzt * nyt * nxt * sizeof(float));
    memset(u_y, 0, nzt * nyt * nxt * sizeof(float));
    memset(u_z, 0, nzt * nyt * nxt * sizeof(float));

    F1sx = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    F1sy = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    F1sz = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    Fmx1 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    Fmy1 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    Fmz1 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    Fmx2 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    Fmy2 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    Fmz2 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(F1sx, 0, nzt * nyt * nxt * sizeof(float));
    memset(F1sy, 0, nzt * nyt * nxt * sizeof(float));
    memset(F1sz, 0, nzt * nyt * nxt * sizeof(float));
    memset(Fmx1, 0, nzt * nyt * nxt * sizeof(float));
    memset(Fmy1, 0, nzt * nyt * nxt * sizeof(float));
    memset(Fmz1, 0, nzt * nyt * nxt * sizeof(float));
    memset(Fmx2, 0, nzt * nyt * nxt * sizeof(float));
    memset(Fmy2, 0, nzt * nyt * nxt * sizeof(float));
    memset(Fmz2, 0, nzt * nyt * nxt * sizeof(float));
    fftemp2 = (float *)malloc(npop * nzt * nyt * nxt * sizeof(float));
    memset(fftemp2, 0, npop * nzt * nyt * nxt * sizeof(float));
    fftemp = (float *)malloc(npop * nzt2 * nyt2 * nxt2 * sizeof(float));
    memset(fftemp, 0, npop * nzt2 * nyt2 * nxt2 * sizeof(float));
    pp = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    psx = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(pp, 0, nzt * nyt * nxt * sizeof(float));
    memset(psx, 0, nzt * nyt * nxt * sizeof(float));
    p = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(p, 0, nzt * nyt * nxt * sizeof(float));

    psxpretemp = (float *)malloc(npop * nzt * nyt * nxt * sizeof(float));
    memset(psxpretemp, 0, npop * nzt * nyt * nxt * sizeof(float));
    spretemp = (float *)malloc(npop * nzt * nyt * nxt * sizeof(float));
    memset(spretemp, 0, npop * nzt * nyt * nxt * sizeof(float));
    psxpretemp2 = (float *)malloc(npop * nzt * nyt * nxt * sizeof(float));
    memset(psxpretemp2, 0, npop * nzt * nyt * nxt * sizeof(float));
    spretemp2 = (float *)malloc(npop * nzt * nyt * nxt * sizeof(float));
    memset(spretemp2, 0, npop * nzt * nyt * nxt * sizeof(float));

    psxpre = (float *)malloc(npop * nzt * nyt * nxt * sizeof(float));
    memset(psxpre, 0, npop * nzt * nyt * nxt * sizeof(float));
    spre = (float *)malloc(npop * nzt * nyt * nxt * sizeof(float));
    memset(spre, 0, npop * nzt * nyt * nxt * sizeof(float));
    smx_pre1 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    smy_pre1 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    smz_pre1 = (float *)malloc(nzt * nyt * nxt * sizeof(float));
    memset(smx_pre1, 0, nzt * nyt * nxt * sizeof(float));
    memset(smy_pre1, 0, nzt * nyt * nxt * sizeof(float));
    memset(smz_pre1, 0, nzt * nyt * nxt * sizeof(float));
    float *rhol = (float *)malloc(NZ * NY * NX * sizeof(float));
    memset(rhol, 0, NZ * NY * NX * sizeof(float));
    float *rhogoble = (float *)malloc(NZ * NY * NX * sizeof(float));
    memset(rhogoble, 0, NZ * NY * NX * sizeof(float));

    float *ffYOZ = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *ffYOZ2 = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *ffXOZ = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *ffXOZ2 = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *ffXOY = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *ffXOY2 = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *ffRecYOZ = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *ffRecYOZ2 = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *ffRecXOZ = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *ffRecXOZ2 = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *ffRecXOY = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *ffRecXOY2 = (float *)malloc(sizeof(float *) * nxt * nyt);
    memset(ffYOZ, 0, nzt * nyt * sizeof(float));
    memset(ffYOZ2, 0, nzt * nyt * sizeof(float));
    memset(ffXOZ, 0, nzt * nxt * sizeof(float));
    memset(ffXOZ2, 0, nzt * nxt * sizeof(float));
    memset(ffXOY, 0, nyt * nxt * sizeof(float));
    memset(ffXOY2, 0, nyt * nxt * sizeof(float));
    memset(ffRecYOZ, 0, nzt * nyt * sizeof(float));
    memset(ffRecYOZ2, 0, nzt * nyt * sizeof(float));
    memset(ffRecXOZ, 0, nzt * nxt * sizeof(float));
    memset(ffRecXOZ2, 0, nzt * nxt * sizeof(float));
    memset(ffRecXOY, 0, nyt * nxt * sizeof(float));
    memset(ffRecXOY2, 0, nyt * nxt * sizeof(float));

    float *fftempYOZ = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *fftempYOZ2 = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *fftempXOZ = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *fftempXOZ2 = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *fftempXOY = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *fftempXOY2 = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *fftempRecYOZ = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *fftempRecYOZ2 = (float *)malloc(sizeof(float *) * nyt * nzt);
    float *fftempRecXOZ = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *fftempRecXOZ2 = (float *)malloc(sizeof(float *) * nxt * nzt);
    float *fftempRecXOY = (float *)malloc(sizeof(float *) * nxt * nyt);
    float *fftempRecXOY2 = (float *)malloc(sizeof(float *) * nxt * nyt);
    memset(fftempYOZ, 0, nzt * nyt * sizeof(float));
    memset(fftempYOZ2, 0, nzt * nyt * sizeof(float));
    memset(fftempXOZ, 0, nzt * nxt * sizeof(float));
    memset(fftempXOZ2, 0, nzt * nxt * sizeof(float));
    memset(fftempXOY, 0, nyt * nxt * sizeof(float));
    memset(fftempXOY2, 0, nyt * nxt * sizeof(float));
    memset(fftempRecYOZ, 0, nzt * nyt * sizeof(float));
    memset(fftempRecYOZ2, 0, nzt * nyt * sizeof(float));
    memset(fftempRecXOZ, 0, nzt * nxt * sizeof(float));
    memset(fftempRecXOZ2, 0, nzt * nxt * sizeof(float));
    memset(fftempRecXOY, 0, nyt * nxt * sizeof(float));
    memset(fftempRecXOY2, 0, nyt * nxt * sizeof(float));

    for (int z = 0; z < nzt; z++)
    {
        for (int y = 0; y < nyt; y++)
        {
            for (int x = 0; x < nxt; x++)
            {
                if (obst[z * nxt * nyt + y * nxt + x] == 0)
                    rho[z * nxt * nyt + y * nxt + x] = rho_ratio;
                else
                    rho[z * nxt * nyt + y * nxt + x] = rho_l;

                for (int i = 0; i < npop; i++)
                {
                    ff[i * nzt * nyt * nxt + z * nxt * nyt + y * nxt + x] = rho[z * nxt * nyt + y * nxt + x] * weight[i];
                    feq[i * nzt * nyt * nxt + z * nxt * nyt + y * nxt + x] = ff[i * nzt * nyt * nxt + z * nxt * nyt + y * nxt + x];
                }
            }
        }
    }
    // if (myid == 0)
    // {
    //     // printf("rho = %f\n", rho[10 * nxt * nyt + 10 * nxt + 10]);
    //     // printf("ff = %f\n", ff[10 * nxt * nyt + 10 * nxt + 10]);
    //     // printf("feq = %f\n", feq[10 * nxt * nyt + 10 * nxt + 10]);
    //     // for (int ai = 0; ai < 19; ai++)
    //     // {
    //     //     printf("ff[%d] = %f\n", ai, ff[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
    //     //     printf("feq[%d] = %f\n", ai, feq[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
    //     // }
    //     test = space3d(nzt, nyt, nxt);
    //     test4d = space4d(npop, nzt, nyt, nxt);
    //     for (int z = 0; z < nzt; z++)
    //     {
    //         for (int y = 0; y < nyt; y++)
    //         {
    //             for (int x = 0; x < nxt; x++)
    //             {
    //                 for (int i = 0; i < npop; i++)
    //                 {
    //                     test4d[i][z][y][x] = tempt[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
    //                 }
    //                 test[z][y][x] = obst[z * nxt * nyt + y * nxt + x];
    //             }
    //         }
    //     }
    //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
    //     char filename[50];
    //     sprintf(filename, "test.txt");
    //     write_data_to_file(test, nzt, nyt, nxt, filename);
    //     free_space3d(test, nzt, nyt);
    //     free_space4d(test4d, npop, nzt, nyt);
    // }

    // 从核初始化
    athread_init();
    clock_t start, finish;
    start = clock();
    for (int ll = 0; ll < t_max; ll++)
    {
        if (ll % 10 == 0 && myid == 0)
        {
            printf("time = %d\n", ll);
        }

        for (Step4D = 0; Step4D < npop; Step4D++)
        {
            athread_spawn(CalTempt, 0);
            athread_join();
        }

        athread_spawn(CalBGK, 0);
        athread_join();

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 0 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = tempt[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = u_y[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     printf("tempt = %f\n", tempt[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("ff = %f\n", ff[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("feq = %f\n", feq[numm * nxt * nyt + numm * nxt + numm]);
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         printf("feq[%d] = %f\n", ai, feq[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //         printf("ff[%d] = %f\n", ai, ff[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //         printf("tempt[%d] = %f\n", ai, tempt[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        for (Step4D = 0; Step4D < npop; Step4D++)
        {
            athread_spawn(CalBGK1, 0);
            athread_join();
        }

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 4 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = feq[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = u_x[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     printf("tempt = %f\n", tempt[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("ff = %f\n", ff[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("feq = %f\n", feq[numm * nxt * nyt + numm * nxt + numm]);
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         printf("feq[%d] = %f\n", ai, feq[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //         printf("ff[%d] = %f\n", ai, ff[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //         printf("tempt[%d] = %f\n", ai, tempt[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        athread_spawn(UpdateBoundary, 0);
        athread_join();

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 0 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = ff[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = u_x[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         printf("ff[%d] = %f\n", ai, ff[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        for (Step4D = 0; Step4D < npop; Step4D++)
        {
            athread_spawn(UpdateMidff, 0);
            athread_join();

            int cx[npop] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 0};
            int cy[npop] = {0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1};
            int cz[npop] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            if (cx[Step4D] == 1)
            {
                // yoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        ffYOZ2[zz * nyt2 + yy] = ff[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1]; // 右侧
                    }
                }
                // MPI_Send(ffYOZ2, (nyt2 * nzt2), MPI_FLOAT, right, 1, MCW);
                // MPI_Recv(ffRecYOZ2, (nyt2 * nzt2), MPI_FLOAT, left, 1, MCW, &status);
                MPI_Sendrecv(ffYOZ2, nyt * nzt, MPI_FLOAT, right, 1,
                             ffRecYOZ2, nyt * nzt, MPI_FLOAT, left, 1, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        fftemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)] = ffRecYOZ2[zz * nyt2 + yy];
                    }
                }
            }
            if (cx[Step4D] == -1)
            {
                // yoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        ffYOZ[zz * nyt2 + yy] = ff[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)]; // 左侧
                    }
                }
                // MPI_Send(ffYOZ, (nyt2 * nzt2), MPI_FLOAT, left, 2, MCW);
                // MPI_Recv(ffRecYOZ, (nyt2 * nzt2), MPI_FLOAT, right, 2, MCW, &status);
                MPI_Sendrecv(ffYOZ, nyt * nzt, MPI_FLOAT, left, 2,
                             ffRecYOZ, nyt * nzt, MPI_FLOAT, right, 2, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        fftemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1] = ffRecYOZ[zz * nyt2 + yy];
                    }
                }
            }
            if (cy[Step4D] == 1)
            {
                // xoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOZ2[zz * nxt2 + xx] = ff[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx]; // 前
                    }
                }
                // MPI_Send(ffXOZ2, (nxt2 * nzt2), MPI_FLOAT, forward, 9, MCW);
                // MPI_Recv(ffRecXOZ2, (nxt2 * nzt2), MPI_FLOAT, back, 9, MCW, &status);
                MPI_Sendrecv(ffXOZ2, nxt * nzt, MPI_FLOAT, forward, 9,
                             ffRecXOZ2, nxt * nzt, MPI_FLOAT, back, 9, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        fftemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx] = ffRecXOZ2[zz * nxt2 + xx];
                    }
                }
            }
            if (cy[Step4D] == -1)
            {
                // xoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOZ[zz * nxt2 + xx] = ff[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx]; // 后侧
                    }
                }
                // MPI_Send(ffXOZ, (nxt2 * nzt2), MPI_FLOAT, back, 6, MCW);
                // MPI_Recv(ffRecXOZ, (nxt2 * nzt2), MPI_FLOAT, forward, 6, MCW, &status);
                MPI_Sendrecv(ffXOZ, nxt * nzt, MPI_FLOAT, back, 6,
                             ffRecXOZ, nxt * nzt, MPI_FLOAT, forward, 6, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        fftemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx] = ffRecXOZ[zz * nxt2 + xx];
                    }
                }
            }
            if (cz[Step4D] == 1)
            {
                // xoy
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOY2[yy * nxt2 + xx] = ff[Step4D * nxt2 * nyt2 * nzt2 + (nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                    }
                }
                // MPI_Send(ffXOY2, (nxt2 * nyt2), MPI_FLOAT, up, 7, MCW);
                // MPI_Recv(ffRecXOY2, (nxt2 * nyt2), MPI_FLOAT, down, 7, MCW, &status);
                MPI_Sendrecv(ffXOY2, nxt2 * nyt2, MPI_FLOAT, up, 7,
                             ffRecXOY2, nxt2 * nyt2, MPI_FLOAT, down, 7, MCW, &status);
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        fftemp[Step4D * nxt2 * nyt2 * nzt2 + (tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = ffRecXOY2[yy * nxt2 + xx];
                    }
                }
            }
            if (cz[Step4D] == -1)
            {
                // xoy
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOY[yy * nxt2 + xx] = ff[Step4D * nxt2 * nyt2 * nzt2 + (tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                    }
                }
                // MPI_Send(ffXOY, (nxt2 * nyt2), MPI_FLOAT, down, 3, MCW);
                // MPI_Recv(ffRecXOY, (nxt2 * nyt2), MPI_FLOAT, up, 3, MCW, &status);
                MPI_Sendrecv(ffXOY, nxt2 * nyt2, MPI_FLOAT, down, 3,
                             ffRecXOY, nxt2 * nyt2, MPI_FLOAT, up, 3, MCW, &status);
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        fftemp[Step4D * nxt2 * nyt2 * nzt2 + (nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = ffRecXOY[yy * nxt2 + xx];
                    }
                }
            }
            MPI_Barrier(MCW);

            memset(ffYOZ, 0, nzt * nyt * sizeof(float));
            memset(ffYOZ2, 0, nzt * nyt * sizeof(float));
            memset(ffXOZ, 0, nzt * nxt * sizeof(float));
            memset(ffXOZ2, 0, nzt * nxt * sizeof(float));
            memset(ffXOY, 0, nyt * nxt * sizeof(float));
            memset(ffXOY2, 0, nyt * nxt * sizeof(float));
            memset(ffRecYOZ, 0, nzt * nyt * sizeof(float));
            memset(ffRecYOZ2, 0, nzt * nyt * sizeof(float));
            memset(ffRecXOZ, 0, nzt * nxt * sizeof(float));
            memset(ffRecXOZ2, 0, nzt * nxt * sizeof(float));
            memset(ffRecXOY, 0, nyt * nxt * sizeof(float));
            memset(ffRecXOY2, 0, nyt * nxt * sizeof(float));

            athread_spawn(UpdateMidff1, 0);
            athread_join();

            int cx2[npop] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int cy2[npop] = {0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
            int cz2[npop] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};

            if (cx2[Step4D] == 1)
            {
                // yoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        ffYOZ2[zz * nyt2 + yy] = fftemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1]; // 右侧
                    }
                }
                // MPI_Send(ffYOZ2, (nyt2 * nzt2), MPI_FLOAT, right, 10, MCW);
                // MPI_Recv(ffRecYOZ2, (nyt2 * nzt2), MPI_FLOAT, left, 10, MCW, &status);
                MPI_Sendrecv(ffYOZ2, nyt2 * nzt2, MPI_FLOAT, right, 10,
                             ffRecYOZ2, nyt2 * nzt2, MPI_FLOAT, left, 10, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        fftemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)] = ffRecYOZ2[zz * nyt2 + yy];
                    }
                }
            }
            if (cx2[Step4D] == -1)
            {
                // yoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        ffYOZ[zz * nyt2 + yy] = fftemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)]; // 左侧
                    }
                }
                // MPI_Send(ffYOZ, (nyt2 * nzt2), MPI_FLOAT, left, 12, MCW);
                // MPI_Recv(ffRecYOZ, (nyt2 * nzt2), MPI_FLOAT, right, 12, MCW, &status);
                MPI_Sendrecv(ffYOZ, nyt2 * nzt2, MPI_FLOAT, left, 12,
                             ffRecYOZ, nyt2 * nzt2, MPI_FLOAT, right, 12, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        fftemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1] = ffRecYOZ[zz * nyt2 + yy];
                    }
                }
            }
            if (cy2[Step4D] == 1)
            {
                // xoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOZ2[zz * nxt2 + xx] = fftemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx]; // 前
                    }
                }
                // MPI_Send(ffXOZ2, (nxt2 * nzt2), MPI_FLOAT, forward, 19, MCW);
                // MPI_Recv(ffRecXOZ2, (nxt2 * nzt2), MPI_FLOAT, back, 19, MCW, &status);
                MPI_Sendrecv(ffXOZ2, nxt2 * nzt2, MPI_FLOAT, forward, 19,
                             ffRecXOZ2, nxt2 * nzt2, MPI_FLOAT, back, 19, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        fftemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx] = ffRecXOZ2[zz * nxt2 + xx];
                    }
                }
            }
            if (cy2[Step4D] == -1)
            {
                // xoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOZ[zz * nxt2 + xx] = fftemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx]; // 后侧
                    }
                }
                // MPI_Send(ffXOZ, (nxt2 * nzt2), MPI_FLOAT, back, 16, MCW);
                // MPI_Recv(ffRecXOZ, (nxt2 * nzt2), MPI_FLOAT, forward, 16, MCW, &status);
                MPI_Sendrecv(ffXOZ, nxt2 * nzt2, MPI_FLOAT, back, 16,
                             ffRecXOZ, nxt2 * nzt2, MPI_FLOAT, forward, 16, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        fftemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx] = ffRecXOZ[zz * nxt2 + xx];
                    }
                }
            }
            if (cz2[Step4D] == 1)
            {
                // xoy
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOY2[yy * nxt2 + xx] = fftemp[Step4D * nxt2 * nyt2 * nzt2 + (nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                    }
                }
                // MPI_Send(ffXOY2, (nxt2 * nyt2), MPI_FLOAT, up, 17, MCW);
                // MPI_Recv(ffRecXOY2, (nxt2 * nyt2), MPI_FLOAT, down, 17, MCW, &status);
                MPI_Sendrecv(ffXOY2, nxt2 * nyt2, MPI_FLOAT, up, 17,
                             ffRecXOY2, nxt2 * nyt2, MPI_FLOAT, down, 17, MCW, &status);
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        fftemp2[Step4D * nxt2 * nyt2 * nzt2 + (tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = ffRecXOY2[yy * nxt2 + xx];
                    }
                }
            }
            if (cz2[Step4D] == -1)
            {
                // xoy
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOY[yy * nxt2 + xx] = fftemp[Step4D * nxt2 * nyt2 * nzt2 + (tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                    }
                }
                // MPI_Send(ffXOY, (nxt2 * nyt2), MPI_FLOAT, down, 13, MCW);
                // MPI_Recv(ffRecXOY, (nxt2 * nyt2), MPI_FLOAT, up, 13, MCW, &status);
                MPI_Sendrecv(ffXOY, nxt2 * nyt2, MPI_FLOAT, down, 13,
                             ffRecXOY, nxt2 * nyt2, MPI_FLOAT, up, 13, MCW, &status);
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        fftemp2[Step4D * nxt2 * nyt2 * nzt2 + (nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = ffRecXOY[yy * nxt2 + xx];
                    }
                }
            }
            MPI_Barrier(MCW);

            memset(ffYOZ, 0, nzt * nyt * sizeof(float));
            memset(ffYOZ2, 0, nzt * nyt * sizeof(float));
            memset(ffXOZ, 0, nzt * nxt * sizeof(float));
            memset(ffXOZ2, 0, nzt * nxt * sizeof(float));
            memset(ffXOY, 0, nyt * nxt * sizeof(float));
            memset(ffXOY2, 0, nyt * nxt * sizeof(float));
            memset(ffRecYOZ, 0, nzt * nyt * sizeof(float));
            memset(ffRecYOZ2, 0, nzt * nyt * sizeof(float));
            memset(ffRecXOZ, 0, nzt * nxt * sizeof(float));
            memset(ffRecXOZ2, 0, nzt * nxt * sizeof(float));
            memset(ffRecXOY, 0, nyt * nxt * sizeof(float));
            memset(ffRecXOY2, 0, nyt * nxt * sizeof(float));
        }

        for (Step4D = 0; Step4D < npop; Step4D++)
        {
            athread_spawn(Updateff, 0);
            athread_join();
        }

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 1 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = ff[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = u_x[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         // printf("fftemp[%d] = %f\n", ai, fftemp2[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //         printf("ff[%d] = %f\n", ai, ff[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        //     // free(test);
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        athread_spawn(Updaterho, 0);
        athread_join();

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 1 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = ff[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = rho[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     printf("rho = %f\n", rho[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("feq = %f\n", feq[7 * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     printf("ff = %f\n", ff[7 * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     float ffall = 0;
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         ffall = ff[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm] + ffall;
        //         printf("ff[%d] = %f\n", ai, ff[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        //     printf("ffall = %f\n", ffall);
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        athread_spawn(UpdateUPot, 0);
        athread_join();

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 1 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = ff[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = ux[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     printf("ux = %f\n", ux[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("uy = %f\n", uy[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("uz = %f\n", uz[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("pp = %f\n", pp[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("psx = %f\n", psx[numm * nxt * nyt + numm * nxt + numm]);
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        athread_spawn(PotBoundaryCalP, 0);
        athread_join();

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 1 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = ff[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = psx[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     printf("psx = %f\n", psx[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("p = %f\n", p[numm * nxt * nyt + numm * nxt + numm]);
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        for (Step4D = 0; Step4D < npop; Step4D++)
        {
            athread_spawn(UpdateMidPot, 0);
            athread_join();

            int cx3[npop] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 0};
            int cy3[npop] = {0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1};
            int cz3[npop] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            if (cx3[Step4D] == -1)
            {
                // yoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        ffYOZ2[zz * nyt2 + yy] = psx[zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1];      // 右侧
                        fftempYOZ2[zz * nyt2 + yy] = obst[zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1]; // 右侧
                    }
                }
                // MPI_Send(ffYOZ2, (nyt2 * nzt2), MPI_FLOAT, right, 21, MCW);
                // MPI_Recv(ffRecYOZ2, (nyt2 * nzt2), MPI_FLOAT, left, 21, MCW, &status);

                // MPI_Send(fftempYOZ2, (nyt2 * nzt2), MPI_FLOAT, right, 121, MCW);
                // MPI_Recv(fftempRecYOZ2, (nyt2 * nzt2), MPI_FLOAT, left, 121, MCW, &status);

                MPI_Sendrecv(ffYOZ2, nyt2 * nzt2, MPI_FLOAT, right, 21,
                             ffRecYOZ2, nyt2 * nzt2, MPI_FLOAT, left, 21, MCW, &status);
                MPI_Sendrecv(fftempYOZ2, nyt2 * nzt2, MPI_FLOAT, right, 121,
                             fftempRecYOZ2, nyt2 * nzt2, MPI_FLOAT, left, 121, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)] = ffRecYOZ2[zz * nyt2 + yy];
                        spretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)] = fftempRecYOZ2[zz * nyt2 + yy];
                    }
                }
            }
            if (cx3[Step4D] == 1)
            {
                // yoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        ffYOZ[zz * nyt2 + yy] = psx[zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)];      // 左侧
                        fftempYOZ[zz * nyt2 + yy] = obst[zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)]; // 左侧
                    }
                }
                // MPI_Send(ffYOZ, (nyt2 * nzt2), MPI_FLOAT, left, 22, MCW);
                // MPI_Recv(ffRecYOZ, (nyt2 * nzt2), MPI_FLOAT, right, 22, MCW, &status);

                // MPI_Send(fftempYOZ, (nyt2 * nzt2), MPI_FLOAT, left, 122, MCW);
                // MPI_Recv(fftempRecYOZ, (nyt2 * nzt2), MPI_FLOAT, right, 122, MCW, &status);

                MPI_Sendrecv(ffYOZ, nyt2 * nzt2, MPI_FLOAT, left, 22,
                             ffRecYOZ, nyt2 * nzt2, MPI_FLOAT, right, 22, MCW, &status);
                MPI_Sendrecv(fftempYOZ, nyt2 * nzt2, MPI_FLOAT, left, 122,
                             fftempRecYOZ, nyt2 * nzt2, MPI_FLOAT, right, 122, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1] = ffRecYOZ[zz * nyt2 + yy];
                        spretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1] = fftempRecYOZ[zz * nyt2 + yy];
                    }
                }
            }
            if (cy3[Step4D] == -1)
            {
                // xoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOZ2[zz * nxt2 + xx] = psx[zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx];      // 前
                        fftempXOZ2[zz * nxt2 + xx] = obst[zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx]; // 前
                    }
                }
                // MPI_Send(ffXOZ2, (nxt2 * nzt2), MPI_FLOAT, forward, 29, MCW);
                // MPI_Recv(ffRecXOZ2, (nxt2 * nzt2), MPI_FLOAT, back, 29, MCW, &status);

                // MPI_Send(fftempXOZ2, (nxt2 * nzt2), MPI_FLOAT, forward, 129, MCW);
                // MPI_Recv(fftempRecXOZ2, (nxt2 * nzt2), MPI_FLOAT, back, 129, MCW, &status);

                MPI_Sendrecv(ffXOZ2, nxt2 * nzt2, MPI_FLOAT, forward, 29,
                             ffRecXOZ2, nxt2 * nzt2, MPI_FLOAT, back, 29, MCW, &status);
                MPI_Sendrecv(fftempXOZ2, nxt2 * nzt2, MPI_FLOAT, forward, 129,
                             fftempRecXOZ2, nxt2 * nzt2, MPI_FLOAT, back, 129, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx] = ffRecXOZ2[zz * nxt2 + xx];
                        spretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx] = fftempRecXOZ2[zz * nxt2 + xx];
                    }
                }
            }
            if (cy3[Step4D] == 1)
            {
                // xoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOZ[zz * nxt2 + xx] = psx[zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx];      // 后侧
                        fftempXOZ[zz * nxt2 + xx] = obst[zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx]; // 后侧
                    }
                }
                // MPI_Send(ffXOZ, (nxt2 * nzt2), MPI_FLOAT, back, 26, MCW);
                // MPI_Recv(ffRecXOZ, (nxt2 * nzt2), MPI_FLOAT, forward, 26, MCW, &status);

                // MPI_Send(fftempXOZ, (nxt2 * nzt2), MPI_FLOAT, back, 126, MCW);
                // MPI_Recv(fftempRecXOZ, (nxt2 * nzt2), MPI_FLOAT, forward, 126, MCW, &status);

                MPI_Sendrecv(ffXOZ, nxt2 * nzt2, MPI_FLOAT, back, 26,
                             ffRecXOZ, nxt2 * nzt2, MPI_FLOAT, forward, 26, MCW, &status);
                MPI_Sendrecv(fftempXOZ, nxt2 * nzt2, MPI_FLOAT, back, 126,
                             fftempRecXOZ, nxt2 * nzt2, MPI_FLOAT, forward, 126, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx] = ffRecXOZ[zz * nxt2 + xx];
                        spretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx] = fftempRecXOZ[zz * nxt2 + xx];
                    }
                }
            }
            if (cz3[Step4D] == -1)
            {
                // xoy
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOY2[yy * nxt2 + xx] = psx[(nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                        fftempXOY2[yy * nxt2 + xx] = obst[(nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                    }
                }
                // MPI_Send(ffXOY2, (nxt2 * nyt2), MPI_FLOAT, up, 27, MCW);
                // MPI_Recv(ffRecXOY2, (nxt2 * nyt2), MPI_FLOAT, down, 27, MCW, &status);

                // MPI_Send(fftempXOY2, (nxt2 * nyt2), MPI_FLOAT, up, 127, MCW);
                // MPI_Recv(fftempRecXOY2, (nxt2 * nyt2), MPI_FLOAT, down, 127, MCW, &status);

                MPI_Sendrecv(ffXOY2, nxt2 * nyt2, MPI_FLOAT, up, 27,
                             ffRecXOY2, nxt2 * nyt2, MPI_FLOAT, down, 27, MCW, &status);
                MPI_Sendrecv(fftempXOY2, nxt2 * nyt2, MPI_FLOAT, up, 127,
                             fftempRecXOY2, nxt2 * nyt2, MPI_FLOAT, down, 127, MCW, &status);
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + (tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = ffRecXOY2[yy * nxt2 + xx];
                        spretemp[Step4D * nxt2 * nyt2 * nzt2 + (tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = fftempRecXOY2[yy * nxt2 + xx];
                    }
                }
            }
            if (cz3[Step4D] == 1)
            {
                // xoy
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOY[yy * nxt2 + xx] = psx[(tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                        fftempXOY[yy * nxt2 + xx] = obst[(tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                    }
                }
                // MPI_Send(ffXOY, (nxt2 * nyt2), MPI_FLOAT, down, 23, MCW);
                // MPI_Recv(ffRecXOY, (nxt2 * nyt2), MPI_FLOAT, up, 23, MCW, &status);

                // MPI_Send(fftempXOY, (nxt2 * nyt2), MPI_FLOAT, down, 123, MCW);
                // MPI_Recv(fftempRecXOY, (nxt2 * nyt2), MPI_FLOAT, up, 123, MCW, &status);

                MPI_Sendrecv(ffXOY, nxt2 * nyt2, MPI_FLOAT, down, 23,
                             ffRecXOY, nxt2 * nyt2, MPI_FLOAT, up, 23, MCW, &status);
                MPI_Sendrecv(fftempXOY, nxt2 * nyt2, MPI_FLOAT, down, 123,
                             fftempRecXOY, nxt2 * nyt2, MPI_FLOAT, up, 123, MCW, &status);
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + (nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = ffRecXOY[yy * nxt2 + xx];
                        spretemp[Step4D * nxt2 * nyt2 * nzt2 + (nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = fftempRecXOY[yy * nxt2 + xx];
                    }
                }
            }
            MPI_Barrier(MCW);

            athread_spawn(UpdateMidPot1, 0);
            athread_join();

            int cx4[npop] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int cy4[npop] = {0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
            int cz4[npop] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};

            if (cx4[Step4D] == -1)
            {
                // yoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        ffYOZ2[zz * nyt2 + yy] = psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1];   // 右侧
                        fftempYOZ2[zz * nyt2 + yy] = spretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1]; // 右侧
                    }
                }
                // MPI_Send(ffYOZ2, (nyt2 * nzt2), MPI_FLOAT, right, 31, MCW);
                // MPI_Recv(ffRecYOZ2, (nyt2 * nzt2), MPI_FLOAT, left, 31, MCW, &status);
                // MPI_Send(fftempYOZ2, (nyt2 * nzt2), MPI_FLOAT, right, 131, MCW);
                // MPI_Recv(fftempRecYOZ2, (nyt2 * nzt2), MPI_FLOAT, left, 131, MCW, &status);

                MPI_Sendrecv(ffYOZ2, nyt2 * nzt2, MPI_FLOAT, right, 31,
                             ffRecYOZ2, nyt2 * nzt2, MPI_FLOAT, left, 31, MCW, &status);
                MPI_Sendrecv(fftempYOZ2, nyt2 * nzt2, MPI_FLOAT, right, 131,
                             fftempRecYOZ2, nyt2 * nzt2, MPI_FLOAT, left, 131, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        psxpretemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)] = ffRecYOZ2[zz * nyt2 + yy];
                        spretemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)] = fftempRecYOZ2[zz * nyt2 + yy];
                    }
                }
            }
            if (cx4[Step4D] == 1)
            {
                // yoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        ffYOZ[zz * nyt2 + yy] = psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)];   // 左侧
                        fftempYOZ[zz * nyt2 + yy] = spretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + (tran - 1)]; // 左侧
                    }
                }
                // MPI_Send(ffYOZ, (nyt2 * nzt2), MPI_FLOAT, left, 32, MCW);
                // MPI_Recv(ffRecYOZ, (nyt2 * nzt2), MPI_FLOAT, right, 32, MCW, &status);
                // MPI_Send(fftempYOZ, (nyt2 * nzt2), MPI_FLOAT, left, 132, MCW);
                // MPI_Recv(fftempRecYOZ, (nyt2 * nzt2), MPI_FLOAT, right, 132, MCW, &status);

                MPI_Sendrecv(ffYOZ, nyt2 * nzt2, MPI_FLOAT, left, 32,
                             ffRecYOZ, nyt2 * nzt2, MPI_FLOAT, right, 32, MCW, &status);
                MPI_Sendrecv(fftempYOZ, nyt2 * nzt2, MPI_FLOAT, left, 132,
                             fftempRecYOZ, nyt2 * nzt2, MPI_FLOAT, right, 132, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (yy = 0; yy < nyt2; yy++)
                    {
                        psxpretemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1] = ffRecYOZ[zz * nyt2 + yy];
                        spretemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + yy * nxt2 + nxt2 - 1] = fftempRecYOZ[zz * nyt2 + yy];
                    }
                }
            }
            if (cy4[Step4D] == -1)
            {
                // xoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOZ2[zz * nxt2 + xx] = psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx];   // 前
                        fftempXOZ2[zz * nxt2 + xx] = spretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx]; // 前
                    }
                }
                // MPI_Send(ffXOZ2, (nxt2 * nzt2), MPI_FLOAT, forward, 39, MCW);
                // MPI_Recv(ffRecXOZ2, (nxt2 * nzt2), MPI_FLOAT, back, 39, MCW, &status);
                // MPI_Send(fftempXOZ2, (nxt2 * nzt2), MPI_FLOAT, forward, 139, MCW);
                // MPI_Recv(fftempRecXOZ2, (nxt2 * nzt2), MPI_FLOAT, back, 139, MCW, &status);

                MPI_Sendrecv(ffXOZ2, nxt2 * nzt2, MPI_FLOAT, forward, 39,
                             ffRecXOZ2, nxt2 * nzt2, MPI_FLOAT, back, 39, MCW, &status);
                MPI_Sendrecv(fftempXOZ2, nxt2 * nzt2, MPI_FLOAT, forward, 139,
                             fftempRecXOZ2, nxt2 * nzt2, MPI_FLOAT, back, 139, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        psxpretemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx] = ffRecXOZ2[zz * nxt2 + xx];
                        spretemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx] = fftempRecXOZ2[zz * nxt2 + xx];
                    }
                }
            }
            if (cy4[Step4D] == 1)
            {
                // xoz
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOZ[zz * nxt2 + xx] = psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx];   // 后侧
                        fftempXOZ[zz * nxt2 + xx] = spretemp[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (tran - 1) * nxt2 + xx]; // 后侧
                    }
                }
                // MPI_Send(ffXOZ, (nxt2 * nzt2), MPI_FLOAT, back, 36, MCW);
                // MPI_Recv(ffRecXOZ, (nxt2 * nzt2), MPI_FLOAT, forward, 36, MCW, &status);
                // MPI_Send(fftempXOZ, (nxt2 * nzt2), MPI_FLOAT, back, 136, MCW);
                // MPI_Recv(fftempRecXOZ, (nxt2 * nzt2), MPI_FLOAT, forward, 136, MCW, &status);

                MPI_Sendrecv(ffXOZ, nxt2 * nzt2, MPI_FLOAT, back, 36,
                             ffRecXOZ, nxt2 * nzt2, MPI_FLOAT, forward, 36, MCW, &status);
                MPI_Sendrecv(fftempXOZ, nxt2 * nzt2, MPI_FLOAT, back, 136,
                             fftempRecXOZ, nxt2 * nzt2, MPI_FLOAT, forward, 136, MCW, &status);
                for (zz = 0; zz < nzt2; zz++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        psxpretemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx] = ffRecXOZ[zz * nxt2 + xx];
                        spretemp2[Step4D * nxt2 * nyt2 * nzt2 + zz * nxt2 * nyt2 + (nyt2 - 1) * nxt2 + xx] = fftempRecXOZ[zz * nxt2 + xx];
                    }
                }
            }
            if (cz4[Step4D] == -1)
            {
                // xoy
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOY2[yy * nxt2 + xx] = psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + (nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                        fftempXOY2[yy * nxt2 + xx] = spretemp[Step4D * nxt2 * nyt2 * nzt2 + (nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                    }
                }
                // MPI_Send(ffXOY2, (nxt2 * nyt2), MPI_FLOAT, up, 37, MCW);
                // MPI_Recv(ffRecXOY2, (nxt2 * nyt2), MPI_FLOAT, down, 37, MCW, &status);
                // MPI_Send(fftempXOY2, (nxt2 * nyt2), MPI_FLOAT, up, 137, MCW);
                // MPI_Recv(fftempRecXOY2, (nxt2 * nyt2), MPI_FLOAT, down, 137, MCW, &status);

                MPI_Sendrecv(ffXOY2, nxt2 * nyt2, MPI_FLOAT, up, 37,
                             ffRecXOY2, nxt2 * nyt2, MPI_FLOAT, down, 37, MCW, &status);
                MPI_Sendrecv(fftempXOY2, nxt2 * nyt2, MPI_FLOAT, up, 137,
                             fftempRecXOY2, nxt2 * nyt2, MPI_FLOAT, down, 137, MCW, &status);
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        psxpretemp2[Step4D * nxt2 * nyt2 * nzt2 + (tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = ffRecXOY2[yy * nxt2 + xx];
                        spretemp2[Step4D * nxt2 * nyt2 * nzt2 + (tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = fftempRecXOY2[yy * nxt2 + xx];
                    }
                }
            }
            if (cz4[Step4D] == 1)
            {
                // xoy
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        ffXOY[yy * nxt2 + xx] = psxpretemp[Step4D * nxt2 * nyt2 * nzt2 + (tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                        fftempXOY[yy * nxt2 + xx] = spretemp[Step4D * nxt2 * nyt2 * nzt2 + (tran - 1) * nxt2 * nyt2 + yy * nxt2 + xx];
                    }
                }
                // MPI_Send(ffXOY, (nxt2 * nyt2), MPI_FLOAT, down, 33, MCW);
                // MPI_Recv(ffRecXOY, (nxt2 * nyt2), MPI_FLOAT, up, 33, MCW, &status);
                // MPI_Send(fftempXOY, (nxt2 * nyt2), MPI_FLOAT, down, 133, MCW);
                // MPI_Recv(fftempRecXOY, (nxt2 * nyt2), MPI_FLOAT, up, 133, MCW, &status);

                MPI_Sendrecv(ffXOY, nxt2 * nyt2, MPI_FLOAT, down, 33,
                             ffRecXOY, nxt2 * nyt2, MPI_FLOAT, up, 33, MCW, &status);
                MPI_Sendrecv(fftempXOY, nxt2 * nyt2, MPI_FLOAT, down, 133,
                             fftempRecXOY, nxt2 * nyt2, MPI_FLOAT, up, 133, MCW, &status);
                for (yy = 0; yy < nyt2; yy++)
                {
                    for (xx = 0; xx < nxt2; xx++)
                    {
                        psxpretemp2[Step4D * nxt2 * nyt2 * nzt2 + (nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = ffRecXOY[yy * nxt2 + xx];
                        spretemp2[Step4D * nxt2 * nyt2 * nzt2 + (nzt2 - 1) * nxt2 * nyt2 + yy * nxt2 + xx] = fftempRecXOY[yy * nxt2 + xx];
                    }
                }
            }
            MPI_Barrier(MCW);
        }

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 1 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = psxpretemp2[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = obst[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         printf("psxpretemp2[%d] = %f\n", ai, psxpretemp2[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //         printf("spretemp2[%d] = %f\n", ai, spretemp2[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         printf("psxpre[%d] = %f\n", ai, psxpre[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         printf("spre[%d] = %f\n", ai, spre[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        athread_spawn(UpdatePot, 0);
        athread_join();

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 1 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = psxpre[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = obst[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         printf("psxpretemp2[%d] = %f\n", ai, psxpretemp2[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //         printf("spretemp2[%d] = %f\n", ai, spretemp2[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         printf("psxpre[%d] = %f\n", ai, psxpre[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        //     for (int ai = 0; ai < 19; ai++)
        //     {
        //         printf("spre[%d] = %f\n", ai, spre[ai * nxt * nyt * nzt + numm * nxt * nyt + numm * nxt + numm]);
        //     }
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        athread_spawn(CalF, 0);
        athread_join();

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 1 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = spre[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = Fmz2[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     printf("smx_pre1 = %f\n", smx_pre1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("smy_pre1 = %f\n", smy_pre1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("smz_pre1 = %f\n", smz_pre1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmx1 = %f\n", Fmx1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmy1 = %f\n", Fmy1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmz1 = %f\n", Fmz1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmx2 = %f\n", Fmx2[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmy2 = %f\n", Fmy2[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmz2 = %f\n", Fmz2[numm * nxt * nyt + numm * nxt + numm]);
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        athread_spawn(BoundaryF, 0);
        athread_join();

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 0 && myid == 0)
        // {
        //     // test = space3d(NZ, NY, NX);
        //     // test4d = space4d(npop, NZ, NY, NX);
        //     // for (int z = 0; z < NZ; z++)
        //     // {
        //     //     for (int y = 0; y < NY; y++)
        //     //     {
        //     //         for (int x = 0; x < NX; x++)
        //     //         {
        //     //             for (int i = 0; i < npop; i++)
        //     //             {
        //     //                 test4d[i][z][y][x] = ff[i * NX * NY * NZ + z * NX * NY + y * NX + x];
        //     //             }
        //     //         }
        //     //     }
        //     // }
        //     // write_4d_data_to_file(test4d, npop, NZ, NY, NX);
        //     // free_space3d(test, NZ, NY);
        //     // free_space4d(test4d, npop, NZ, NY);
        //     printf("smx_pre1 = %f\n", smx_pre1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("smy_pre1 = %f\n", smy_pre1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("smz_pre1 = %f\n", smz_pre1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmx1 = %f\n", Fmx1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmy1 = %f\n", Fmy1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmz1 = %f\n", Fmz1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmx2 = %f\n", Fmx2[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmy2 = %f\n", Fmy2[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmz2 = %f\n", Fmz2[numm * nxt * nyt + numm * nxt + numm]);
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        athread_spawn(UpdateF, 0);
        athread_join();

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 1 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 for (int i = 0; i < npop; i++)
        //                 {
        //                     test4d[i][z][y][x] = spre[i * nxt * nyt * nzt + z * nxt * nyt + y * nxt + x];
        //                 }
        //                 test[z][y][x] = F1sx[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        //     printf("****************************************************************\n");
        //     printf("F1sx = %f\n", F1sx[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("F1sy = %f\n", F1sy[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("F1sz = %f\n", F1sz[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmx1 = %f\n", Fmx1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmy1 = %f\n", Fmy1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmz1 = %f\n", Fmz1[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmx2 = %f\n", Fmx2[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmy2 = %f\n", Fmy2[numm * nxt * nyt + numm * nxt + numm]);
        //     printf("Fmz2 = %f\n", Fmz2[numm * nxt * nyt + numm * nxt + numm]);
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        // //------------------------------------------------------测试----------------------------------------------------
        // if (ll == 9 && myid == 0)
        // {
        //     test = space3d(nzt, nyt, nxt);
        //     test4d = space4d(npop, nzt, nyt, nxt);
        //     for (int z = 0; z < nzt; z++)
        //     {
        //         for (int y = 0; y < nyt; y++)
        //         {
        //             for (int x = 0; x < nxt; x++)
        //             {
        //                 test[z][y][x] = rho[z * nxt * nyt + y * nxt + x];
        //             }
        //         }
        //     }
        //     write_4d_data_to_file(test4d, npop, nzt, nyt, nxt);
        //     char filename[50];
        //     sprintf(filename, "test.txt");
        //     write_data_to_file(test, nzt, nyt, nxt, filename);
        //     free_space3d(test, nzt, nyt);
        //     free_space4d(test4d, npop, nzt, nyt);
        // }
        // //------------------------------------------------------测试----------------------------------------------------

        //------------------------------------------------------迭代结束----------------------------------------------------
    }
    MPI_Barrier(MCW);
    finish = clock();
    printf("myid  = %d   %f seconds\n", myid, (double)(finish - start) / CLOCKS_PER_SEC);

    free(ffYOZ);
    free(ffYOZ2);
    free(ffXOZ);
    free(ffXOZ2);
    free(ffXOY);
    free(ffXOY2);
    free(ffRecYOZ);
    free(ffRecYOZ2);
    free(ffRecXOZ);
    free(ffRecXOZ2);
    free(ffRecXOY);
    free(ffRecXOY2);
    free(fftempYOZ);
    free(fftempYOZ2);
    free(fftempXOZ);
    free(fftempXOZ2);
    free(fftempXOY);
    free(fftempXOY2);
    free(fftempRecYOZ);
    free(fftempRecYOZ2);
    free(fftempRecXOZ);
    free(fftempRecXOZ2);
    free(fftempRecXOY);
    free(fftempRecXOY2);

    free(obst);
    free(rhoexp);
    free(ff);
    free(feq);
    free(tempt);
    free(ux);
    free(uy);
    free(uz);
    free(u_x);
    free(u_y);
    free(u_z);
    free(F1sx);
    free(F1sy);
    free(F1sz);
    free(Fmx1);
    free(Fmy1);
    free(Fmz1);
    free(Fmx2);
    free(Fmy2);
    free(Fmz2);
    free(fftemp);
    free(fftemp2);
    free(pp);
    free(psx);
    free(p);
    free(psxpretemp);
    free(spretemp);
    free(psxpretemp2);
    free(spretemp2);
    free(psxpre);
    free(spre);
    free(smx_pre1);
    free(smy_pre1);
    free(smz_pre1);

    // if (myid == 0)
    // {
    //     test = space3d(nzt, nyt, nxt);
    //     for (int z = 0; z < nzt; z++)
    //     {
    //         for (int y = 0; y < nyt; y++)
    //         {
    //             for (int x = 0; x < nxt; x++)
    //             {
    //                 test[z][y][x] = rho[z * nxt * nyt + y * nxt + x];
    //             }
    //         }
    //     }
    //     char filename[50];
    //     sprintf(filename, "test.txt");
    //     write_data_to_file(test, nzt, nyt, nxt, filename);
    //     free_space3d(test, nzt, nyt);
    //     printf("xx\n");
    // }
    for (int zz = 0; zz < nzt; zz++)
    {
        for (int yy = 0; yy < nyt; yy++)
        {
            for (int xx = 0; xx < nxt; xx++)
            {
                int gobleIndex = (coordz * nzt + zz) * NX * NY + (coordy * nyt + yy) * NX + (coordx * nxt + xx);
                rhogoble[gobleIndex] = rho[zz * nxt * nyt + yy * nxt + xx];
            }
        }
    }

    size_t chunk_size = 100 * 100 * 100; // 定义每块的大小
    size_t total_size = NX * NY * NZ;
    size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;

    // 分块进行 Reduce 操作
    for (size_t i = 0; i < num_chunks; i++)
    {
        size_t start = i * chunk_size;
        size_t count = (start + chunk_size <= total_size) ? chunk_size : (total_size - start);

        int mpi_error = MPI_Reduce(&rhogoble[start], &rhol[start], count, MPI_FLOAT, MPI_SUM, 0, MCW);
        if (mpi_error != MPI_SUCCESS)
        {
            char error_string[BUFSIZ];
            int length_of_error_string;
            MPI_Error_string(mpi_error, error_string, &length_of_error_string);
            fprintf(stderr, "Process %zu: MPI_Reduce error in chunk %zu: %s\n", myid, i, error_string);
            MPI_Abort(MCW, mpi_error);
        }

        MPI_Barrier(MCW);
        // if (myid == 0)
        // {
        //     printf("Root process: Completed chunk %zu of MPI_Reduce.\n", i);
        // }
    }

    // 后处理
    if (myid == 0)
    {
        printf("myid0 enter");
        double q = 0.0;
        for (int k = 0; k < NZ; k++)
        {
            for (int j = 0; j < NY; j++)
            {
                for (int i = 0; i < NX; i++)
                {
                    q += rhol[k * NX * NY + j * NX + i];
                }
            }
        }

        for (int k = 0; k < NZ; k++)
        {
            for (int j = 0; j < NY; j++)
            {
                for (int i = 0; i < NX; i++)
                {
                    if (dataJudge[k * NX * NY + j * NX + i] != 0.0f)
                    {
                        rhol[k * NX * NY + j * NX + i] = 0.01;
                    }
                }
            }
        }

        int s = 0;

        for (int k = 0; k < NZ; k++)
        {
            for (int j = 0; j < NY; j++)
            {
                for (int i = 0; i < NX; i++)
                {
                    if (rhol[k * NX * NY + j * NX + i] < ((rho_h + rho_l) / 2.0) && dataJudge[k * NX * NY + j * NX + i] != 1)
                    {
                        s++;
                        data11[k][j][i] = 2;
                        // printf("1\n");
                    }
                    else
                    {
                        data11[k][j][i] = 3;
                    }
                }
            }
        }

        float *rock = (float *)malloc(NZ * NY * NX * sizeof(float));
        memset(rock, 0, NZ * NY * NX * sizeof(float));
        for (int iz = 0; iz < NZ; iz++)
        {
            for (int iy = 0; iy < NY; iy++)
            {
                for (int ix = 0; ix < NX; ix++)
                {
                    rock[iz * NX * NY + iy * NX + ix] = data11[iz][iy][ix];

                    if (rock[iz * NX * NY + iy * NX + ix] == 2)
                        DATA1D[iz * NX * NY + iy * NX + ix] = 7; // 7是水  1是油

                    DATA3D[iz][iy][ix] = DATA1D[iz * NX * NY + iy * NX + ix];
                }
            }
        }

        for (int i = 1; i <= 7; i++)
        {
            int t = i;
            int num = countElements1D(DATA1D, NZ, NY, NX, t);
            int totalElements = NZ * NY * NX;
            float P = (float)num / totalElements;
            printf("Component %d ---- %f\n", i, P);
        }

        int numw = countElements1D(DATA1D, NZ, NY, NX, 7); // water
        int numo = countElements1D(DATA1D, NZ, NY, NX, 1); // oil
        printf("numw = %d, numo = %d\n", numw, numo);
        float sw = 100 * (float)numw / ((float)numw + (float)numo);
        printf("Water saturation ---- %f\n", sw);

        // //输出TXT文件
        // char filename[50];
        // sprintf(filename, "%f_%f.txt", sw, rho_ratio);
        // clock_t start1, finish1;
        // start1 = clock();
        // write_data_to_file(DATA3D, NZ, NY, NX, filename);
        // finish1 = clock();
        // printf("TXT file write %f seconds\n", (float)(finish1 - start1) / CLOCKS_PER_SEC);

        // //输出二进制文件
        int totalSize = NZ * NY * NX;
        char filename11[50];
        sprintf(filename11, "%f_%fBIN.bin", sw, rho_ratio);
        clock_t start5, finish5;
        start5 = clock();
        write_data_to_file2BIN111(DATA1D, totalSize, filename11);
        finish5 = clock();
        printf("Binary file write %f seconds\n", (float)(finish5 - start5) / CLOCKS_PER_SEC);

        // //------------------------------------------------------测试----------------------------------------------------
        // float *testre = (float *)malloc(sizeof(float *) * NX * NY * NZ);
        // memset(testre, 0, NZ * NX * NY * sizeof(float));
        // char *filename00 = "./0.092986_1.900000BIN.bin";
        // FILE *file00 = fopen(filename00, "rb");
        // if (file00 == NULL)
        // {
        //     fprintf(stderr, "no open file0\n");
        //     return 1;
        // }
        // // 调用函数
        // readCompressedBinary(testre, totalSize, filename00);
        // float para;
        // for (int xx = 0; xx < totalSize; xx++)
        // {
        //     para = DATA1D[xx] - testre[xx];
        //     if (para != 0)
        //     {
        //         printf("error != 0\n");
        //     }
        // }
        // printf("completed test\n");
        // free(testre);
        // //------------------------------------------------------测试----------------------------------------------------
        free(rock);
    }

    free(rho);
    free(rhol);
    free(rhogoble);
    free_space3d(data11, NZ, NY);
    free(dataJudge);
    free(DATA1D);
    free_space3d(DATA3D, NZ, NY);
    athread_halt();
    MPI_Finalize();
    return 0;
}

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

float ****space4d(int n1, int n2, int n3, int n4)
{
    float ****a;
    int i, j, k;

    a = (float ****)malloc(sizeof(float ***) * n1);
    for (i = 0; i < n1; i++)
    {
        a[i] = (float ***)malloc(sizeof(float **) * n2);
        for (j = 0; j < n2; j++)
        {
            a[i][j] = (float **)malloc(sizeof(float *) * n3);
            for (k = 0; k < n3; k++)
            {
                a[i][j][k] = (float *)malloc(sizeof(float) * n4);
            }
        }
    }

    for (i = 0; i < n1; i++)
        for (j = 0; j < n2; j++)
            for (k = 0; k < n3; k++)
                for (int ll = 0; ll < n4; ll++)
                {
                    a[i][j][k][ll] = 0.0f;
                }

    return a;
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

void free_space4d(float ****a, int nr, int ny, int nc)
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

Index3D *find_nonzero_elements(float *array, int *Boundray, int NZ, int NY, int NX)
{
    int max_elements = NZ * NY * NX;
    Index3D *indices = (Index3D *)malloc(max_elements * sizeof(Index3D));
    int index = 0;
    for (int z = 0; z < NZ; z++)
        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++)
            {
                if (array[z * NX * NY + y * NX + x] != 0.0f)
                {
                    indices[index].z = z;
                    indices[index].y = y;
                    indices[index].x = x;
                    index++;
                }
            }
    indices = (Index3D *)realloc(indices, index * sizeof(Index3D));
    *Boundray = index;
    return indices;
}

void zero3dArray(float ***array, int nr, int ny, int nc)
{
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            memset(array[i][j], 0, nc * sizeof(float));
        }
    }
}

// 计算 rock 中等于 t 的元素数量
int countElements(float ***rock, int lz, int ly, int lx, float t)
{
    int count = 0;
    for (int i = 0; i < lz; i++)
    {
        for (int j = 0; j < ly; j++)
        {
            for (int k = 0; k < lx; k++)
            {
                if (rock[i][j][k] == t)
                {
                    count++;
                }
            }
        }
    }
    return count;
}

int countElements1D(float *rock, int lz, int ly, int lx, float t)
{
    int count = 0;
    int allnumber = lz * ly * lx;
    for (int number = 0; number < allnumber; number++)
    {
        if (rock[number] == t)
        {
            count++;
        }
    }
    return count;
}

// 将二进制数据写入文件—三维
void wfile3d(char filename[], float ***data, int nr, int ny, int nc)
{
    int i, j, k;
    FILE *fp = fopen(filename, "wt");
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

void wfile1d(char filename[], float *data, int nz, int ny, int nx)
{
    int i;
    FILE *fp = fopen(filename, "wt");
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
void write_data_to_file(float ***data, int nz, int ny, int nx, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < nz; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nx; k++)
            {
                fprintf(file, "%d %d %d %f\n", i, j, k, data[i][j][k]);
            }
        }
    }

    fclose(file);
}

// 将数据写入二进制文件
void write_data_to_file2BIN(float *data, int totalSize, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        perror("无法打开文件");
        return;
    }

    int buffer = 0;
    int bufferIndex = 0;

    for (int i = 0; i < totalSize; i++)
    {
        int value = (int)data[i];

        if (value < 1 || value > 7)
        {
            fprintf(stderr, "数组中的元素超出范围 (1-7): %d\n", value);
            fclose(file);
            return;
        }
        buffer |= (value << (32 - (bufferIndex + 1) * 3));
        bufferIndex++;

        if (bufferIndex == 10)
        {
            fwrite(&buffer, sizeof(int), 1, file);
            buffer = 0;
            bufferIndex = 0;
        }
    }

    if (bufferIndex > 0)
    {
        fwrite(&buffer, sizeof(int), 1, file);
    }
    fclose(file);
}

// 将数据写入二进制文件
void write_data_to_file2BIN111(float *data, int totalSize, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        perror("can't open file");
        return;
    }

    int buffer = 0;
    int bufferIndex = 0;
    int G = 0;
    int *A = (int *)calloc(totalSize / 10, sizeof(int));
    if (!A)
    {
        perror("memory allocate error\n");
        fclose(file);
        return;
    }

    for (int i = 0; i < totalSize; i++)
    {
        int value = (int)data[i];

        if (value < 1 || value > 7)
        {
            fprintf(stderr, "The element in the array is out of range (1-7): %d\n", value);
            fclose(file);
            return;
        }
        buffer |= (value << (32 - (bufferIndex + 1) * 3));
        bufferIndex++;

        if (bufferIndex == 10)
        {
            // fwrite(&buffer, sizeof(int), 1, file);
            A[G] = buffer;
            G++;
            buffer = 0;
            bufferIndex = 0;
        }
    }

    fwrite(A, sizeof(int), totalSize / 10, file);

    if (bufferIndex > 0)
    {
        fwrite(&buffer, sizeof(int), 1, file);
    }
    fclose(file);

    free(A);
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

// 将四维数据写入文件
void write_4d_data_to_file(float ****data, int nh, int nz, int ny, int nx)
{
    FILE *file = fopen("data4Dtempt11.txt", "w");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int h = 0; h < nh; h++)
    {
        for (int i = 0; i < nz; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                for (int k = 0; k < nx; k++)
                {
                    fprintf(file, "%d %d %d %d %f\n", h, i, j, k, data[h][i][j][k]);
                }
            }
        }
    }

    fclose(file);
}
